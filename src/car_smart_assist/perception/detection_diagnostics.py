"""Optional locked-point diagnostics; never changes AP, NMS or model selection."""
from __future__ import annotations

from collections import Counter
import numpy as np

SIZE_BUCKETS = ("tiny_lt16", "small_16_32", "medium_32_96", "large_ge96")


def size_bucket(area: float) -> str:
    if not np.isfinite(area) or area < 0:
        raise ValueError("Box area must be finite and nonnegative")
    return SIZE_BUCKETS[sum(area >= edge * edge for edge in (16, 32, 96))]


def _boxes(value, count):
    value = np.asarray(value, dtype=np.float64).reshape((-1, 4))
    if len(value) != count or not np.isfinite(value).all():
        raise ValueError("Box count/coordinates are invalid")
    if np.any(value[:, 2:] < value[:, :2]):
        raise ValueError("Box corners must be ordered")
    return value


def _iou(targets, predictions):
    start = np.maximum(targets[:, None, :2], predictions[None, :, :2])
    end = np.minimum(targets[:, None, 2:], predictions[None, :, 2:])
    intersection = np.maximum(end - start, 0).prod(axis=-1)
    a = (targets[:, 2:] - targets[:, :2]).prod(axis=-1)
    b = (predictions[:, 2:] - predictions[:, :2]).prod(axis=-1)
    union = a[:, None] + b[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def diagnose_frame(gt_boxes, gt_classes, pred_boxes, pred_classes, scores, *, input_areas):
    """IoU-descending one-to-one same-class matches; ties use GT then prediction index."""
    gt_classes = np.asarray(gt_classes)
    pred_classes = np.asarray(pred_classes)
    for labels in (gt_classes, pred_classes):
        if labels.ndim != 1 or not np.isfinite(labels).all() or np.any(labels < 0) or np.any(labels != labels.astype(int)):
            raise ValueError("Class IDs must be nonnegative integers")
    targets, predictions = _boxes(gt_boxes, len(gt_classes)), _boxes(pred_boxes, len(pred_classes))
    scores = np.asarray(scores, dtype=np.float64)
    areas = np.asarray(input_areas, dtype=np.float64)
    if scores.shape != (len(predictions),) or not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("Scores must be finite probabilities")
    if areas.shape != (len(targets),) or not np.isfinite(areas).all() or np.any(areas < 0):
        raise ValueError("Input-space areas must match targets")
    overlap = _iou(targets, predictions)
    candidates = [
        (float(overlap[g, p]), g, p)
        for g, p in zip(*np.nonzero((overlap >= 0.5) & (gt_classes[:, None] == pred_classes[None, :])))
    ]
    used_gt, used_pred, matches = set(), set(), []
    for iou, g, p in sorted(candidates, key=lambda row: (-row[0], row[1], row[2])):
        if g not in used_gt and p not in used_pred:
            used_gt.add(g)
            used_pred.add(p)
            matches.append({"gt": int(g), "prediction": int(p), "iou": iou})
    native_areas = (targets[:, 2:] - targets[:, :2]).prod(axis=-1)
    truth = [
        {"class_id": int(label), "box": box.tolist(), "matched": i in used_gt,
         "native_size": size_bucket(native_areas[i]), "input_size": size_bucket(areas[i]),
         "native_area": float(native_areas[i]), "input_area": float(areas[i])}
        for i, (label, box) in enumerate(zip(gt_classes, targets))
    ]
    proposals = []
    for p, (label, box, score) in enumerate(zip(pred_classes, predictions, scores)):
        same = overlap[gt_classes == label, p]
        same_iou = float(same.max()) if len(same) else 0.0
        any_iou = float(overlap[:, p].max()) if len(targets) else 0.0
        if p in used_pred:
            kind = "matched"
        elif same_iou >= 0.5:
            kind = "duplicate_overlap"
        elif any_iou >= 0.5:
            kind = "wrong_class_overlap"
        elif same_iou >= 0.1:
            kind = "localization_overlap"
        else:
            kind = "other_unmatched"
        proposals.append({"class_id": int(label), "box": box.tolist(), "score": float(score),
                          "kind": kind, "max_same_class_iou": same_iou, "max_any_class_iou": any_iou})
    return {"targets": truth, "predictions": proposals, "matches": matches}


def _counts(targets, predictions):
    tp = sum(row["matched"] for row in targets)
    support, predicted = len(targets), len(predictions)
    return {
        "tp": tp, "fp": predicted - tp, "fn": support - tp, "support": support,
        "precision": tp / predicted if predicted else None,
        "recall": tp / support if support else None,
        "f1": 2 * tp / (support + predicted) if support + predicted else None,
        "prediction_categories": dict(Counter(row["kind"] for row in predictions)),
        "size_recall": {
            space: {
                bucket: {
                    "support": sum(row[space + "_size"] == bucket for row in targets),
                    "matched": sum(row[space + "_size"] == bucket and row["matched"] for row in targets),
                    "recall": (
                        sum(row[space + "_size"] == bucket and row["matched"] for row in targets)
                        / sum(row[space + "_size"] == bucket for row in targets)
                    ) if any(row[space + "_size"] == bucket for row in targets) else None,
                } for bucket in SIZE_BUCKETS
            } for space in ("native", "input")
        },
    }


class DetectionDiagnostics:
    def __init__(self, names):
        self.names = list(names)
        self.frames = []

    def add(self, *, image, source, condition, gt_boxes, gt_classes, pred_boxes, pred_classes, scores, input_areas, supervised_classes=None):
        frame = diagnose_frame(gt_boxes, gt_classes, pred_boxes, pred_classes, scores, input_areas=input_areas)
        if any(row["class_id"] >= len(self.names) for key in ("targets", "predictions") for row in frame[key]):
            raise ValueError("Class ID outside configured names")
        coverage = list(range(len(self.names))) if supervised_classes is None else list(supervised_classes)
        if any(type(i) is not int or not 0 <= i < len(self.names) for i in coverage):
            raise ValueError("Invalid supervised class coverage")
        if any(row["class_id"] not in coverage for key in ("targets", "predictions") for row in frame[key]):
            raise ValueError("Diagnostics require predictions and targets filtered to supervised classes")
        self.frames.append({"image": str(image), "source": str(source), "condition": str(condition),
                            "supervised_classes": coverage, **frame})

    def report(self):
        def summarize(frames, label=None):
            targets = [r for frame in frames for r in frame["targets"] if label is None or r["class_id"] == label]
            predictions = [r for frame in frames for r in frame["predictions"] if label is None or r["class_id"] == label]
            return {"image_count": len(frames), **_counts(targets, predictions)}
        return {
            "schema_version": 1,
            "scope": "Locked operating point; source-covered classes and annotated ROI only",
            "matching": "Same class IoU>=0.5; descending IoU, ties GT then prediction index; one-to-one greedy",
            "limitations": [
                "Diagnostic matching is separate from official AP/operating counts and may differ on ambiguous matches.",
                "Size buckets use box area, not side length; native and actual network-input spaces are separate.",
                "These are recall buckets, not COCO AP-small/medium/large.",
                "Unmatched categories describe overlap, not causal error attribution or whole-image false positives.",
                "Conditions are existing dataset metadata, not independently annotated co-occurring weather.",
            ],
            "overall": summarize(self.frames),
            "by_class": {name: summarize([frame for frame in self.frames if i in frame["supervised_classes"]], i)
                         for i, name in enumerate(self.names)},
            **{
                "by_" + group: {
                    value: summarize([frame for frame in self.frames if frame[group] == value])
                    for value in sorted({frame[group] for frame in self.frames})
                } for group in ("source", "condition")
            },
            "frames": self.frames,
        }


def diagnostic_validator(base, collector):
    """Capture only inputs already transformed and filtered by the owning validator."""
    class DiagnosticValidator(base):
        def _prepare_batch(self, si, batch):
            result = super()._prepare_batch(si, batch)
            result["diagnostic_image"] = batch["im_file"][si]
            boxes = batch["bboxes"][batch["batch_idx"] == si]
            h, w = batch["img"].shape[-2:]
            result["diagnostic_input_areas"] = boxes[:, 2] * w * boxes[:, 3] * h
            return result

        def _process_batch(self, preds, batch):
            result = super()._process_batch(preds, batch)
            row = batch.get("joint_row", {})
            def array(value):
                return value.detach().cpu().numpy()
            collector.add(
                image=batch["diagnostic_image"], source=row.get("dataset", "unknown"),
                condition=row.get("condition", "unknown"),
                gt_boxes=array(batch["bboxes"]), gt_classes=array(batch["cls"]),
                pred_boxes=array(preds["bboxes"]), pred_classes=array(preds["cls"]),
                scores=array(preds["conf"]), input_areas=array(batch["diagnostic_input_areas"]),
                supervised_classes=row.get("supervised_classes"),
            )
            return result
    return DiagnosticValidator
