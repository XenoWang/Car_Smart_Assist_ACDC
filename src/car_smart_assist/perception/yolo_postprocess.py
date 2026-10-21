"""项目内标准 detect 后处理及严格校准身份，不修改第三方函数。"""

from __future__ import annotations

import hashlib
import json
import math
from numbers import Real
from pathlib import Path

import torch

from car_smart_assist.config.detection import (
    YOLO_AGNOSTIC_NMS,
    YOLO_AP_MODE,
    YOLO_CALIBRATION_SCHEMA_VERSION,
    YOLO_LOCKED_MODE,
    YOLO_MAX_NMS,
    YOLO_MAX_WH,
    YOLO_POSTPROCESS_VERSION,
    YOLO_PROPOSAL_MODE,
    YOLO_SWEEP_CONFIDENCE,
)


def ordered_class_names(names):
    if isinstance(names, dict):
        if set(names) != set(range(len(names))):
            raise ValueError("Detection class IDs must be contiguous from zero")
        names = [names[index] for index in range(len(names))]
    names = list(names)
    if (
        not names
        or not all(isinstance(name, str) and name for name in names)
        or len(set(names)) != len(names)
    ):
        raise ValueError("Detection class names must be nonempty and unique")
    return names


def threshold_values(names, confidence, class_thresholds=None):
    names = ordered_class_names(names)
    class_thresholds = class_thresholds or {}
    if set(class_thresholds) - set(names):
        raise ValueError("Detection thresholds contain unknown classes")
    raw_values = [confidence, *(class_thresholds.get(name, confidence) for name in names)]
    if any(isinstance(value, bool) or not isinstance(value, Real) for value in raw_values):
        raise ValueError("Detection confidence thresholds must be numeric, not boolean")
    values = [float(value) for value in raw_values[1:]]
    if not math.isfinite(confidence) or not 0 < confidence <= 1:
        raise ValueError("Detection confidence thresholds must be in (0, 1]")
    if any(not math.isfinite(value) or not 0 < value <= 1 for value in values):
        raise ValueError("Detection confidence thresholds must be in (0, 1]")
    return values


def postprocess_detection(
    predictions, *, nc, mode, confidence, class_thresholds,
    names, iou, max_det, agnostic=False, end2end=False, task="detect",
):
    """原始最高类资格 → max_nms → NMS → max_det；proposal不截数量。"""
    if mode not in (YOLO_LOCKED_MODE, YOLO_PROPOSAL_MODE, YOLO_AP_MODE):
        raise ValueError(f"Unsupported detection postprocess mode: {mode}")
    if task != "detect" or end2end:
        raise ValueError("Only standard raw detect output is supported")
    if isinstance(predictions, (list, tuple)):
        if not predictions:
            raise ValueError("Empty detection output container")
        predictions = predictions[0]
    if (
        not isinstance(predictions, torch.Tensor)
        or predictions.ndim != 3
        or predictions.shape[1] != 4 + nc
    ):
        raise ValueError("Detection output must be raw [B, 4 + nc, N]")
    if nc != len(ordered_class_names(names)):
        raise ValueError("Detection output class count differs from names")
    if not torch.isfinite(predictions).all():
        raise ValueError("Detection raw boxes/scores must be finite")
    from torchvision.ops import nms

    # 高类别偏移在FP16下会抹掉小框坐标；资格比较与NMS都至少用FP32。
    work = (
        predictions.float()
        if predictions.dtype in (torch.float16, torch.bfloat16)
        else predictions
    )
    values = (
        threshold_values(names, confidence, class_thresholds)
        if mode == YOLO_LOCKED_MODE
        else [YOLO_SWEEP_CONFIDENCE] * nc
    )
    thresholds = work.new_tensor(values)
    outputs = []
    for image in work.transpose(1, 2):
        if not len(image):
            outputs.append(work.new_zeros((0, 6)))
            continue
        scores, labels = image[:, 4:].max(1)
        keep = scores >= thresholds[labels]
        xywh = image[keep, :4]
        half_size = xywh[:, 2:] / 2
        boxes = torch.cat((xywh[:, :2] - half_size, xywh[:, :2] + half_size), 1)
        scores, labels = scores[keep], labels[keep]
        if mode != YOLO_PROPOSAL_MODE and len(scores) > YOLO_MAX_NMS:
            selected = scores.argsort(descending=True)[:YOLO_MAX_NMS]
            boxes, scores, labels = boxes[selected], scores[selected], labels[selected]
        if len(scores):
            offsets = labels[:, None] * (0 if agnostic else YOLO_MAX_WH)
            selected = nms(boxes + offsets, scores, iou)
            if mode != YOLO_PROPOSAL_MODE:
                selected = selected[:max_det]
            outputs.append(
                torch.cat((boxes, scores[:, None], labels[:, None].to(scores.dtype)), 1)[selected]
            )
        else:
            outputs.append(work.new_zeros((0, 6)))
    return outputs


def calibration_path(root: Path, cfg, output_dir=None):
    configured = cfg.get("calibration_file")
    if not isinstance(configured, str) or not configured.strip():
        raise ValueError("Detection calibration_file must be configured")
    base = root / configured
    parent = root / output_dir if output_dir is not None else base.parent
    return parent / YOLO_POSTPROCESS_VERSION / base.name


def calibration_identity(root: Path, cfg, names, checkpoint=None, manifest=None):
    names = ordered_class_names(names)
    if manifest is None:
        data = root / cfg["data"]
        manifest = json.loads((data.parent / "manifest.json").read_text(encoding="utf-8"))
    manifest_names = [
        row["name"] if isinstance(row, dict) else row for row in manifest["classes"]
    ]
    if manifest_names != names:
        raise ValueError("Detection model/data class IDs or names differ")
    args = cfg["inference"]
    return {
        "schema_version": YOLO_CALIBRATION_SCHEMA_VERSION,
        "postprocess_version": YOLO_POSTPROCESS_VERSION,
        "mode": YOLO_LOCKED_MODE,
        "calibration_selection_mode": YOLO_PROPOSAL_MODE,
        "ap_mode": YOLO_AP_MODE,
        "sweep_confidence": YOLO_SWEEP_CONFIDENCE,
        "checkpoint_sha256": hashlib.sha256(
            (checkpoint or root / cfg["checkpoint"]).read_bytes()
        ).hexdigest(),
        "dataset_signature": manifest["signature"],
        "classes": [{"id": index, "name": name} for index, name in enumerate(names)],
        "imgsz": args["imgsz"],
        "nms_iou": args["iou"],
        "max_det": args["max_det"],
        "max_nms": YOLO_MAX_NMS,
        "max_wh": YOLO_MAX_WH,
        "agnostic_nms": YOLO_AGNOSTIC_NMS,
        "class_selection": "original_best_class_only",
        "threshold_comparison": ">=",
    }


def validate_calibration(payload, identity, names):
    if not isinstance(payload, dict) or any(
        payload.get(key) != value for key, value in identity.items()
    ):
        raise ValueError(
            "Detection calibration identity/settings changed or format is obsolete; "
            "recalibrate first"
        )
    confidence = payload["confidence_threshold"]
    if isinstance(confidence, bool) or not isinstance(confidence, Real):
        raise ValueError("Detection calibration confidence must be numeric, not boolean")
    confidence = float(confidence)
    thresholds = payload["class_confidence_thresholds"]
    if not isinstance(thresholds, dict) or set(thresholds) != set(ordered_class_names(names)):
        raise ValueError("Detection calibration must cover every class")
    threshold_values(names, confidence, thresholds)
    if not math.isfinite(confidence) or not 0 < confidence <= 1:
        raise ValueError("Detection calibration confidence must be in (0, 1]")
    if payload.get("split") != "calibration" or payload.get("passes") != [
        YOLO_PROPOSAL_MODE, YOLO_LOCKED_MODE
    ]:
        raise ValueError("Detection calibration requires completed proposal and locked passes")
    sources = payload.get("class_threshold_sources")
    if not isinstance(sources, dict) or set(sources) != set(ordered_class_names(names)):
        raise ValueError("Detection calibration must identify every class threshold source")
    return payload


def load_calibration(
    root: Path, cfg, names, *, output_dir=None, checkpoint=None, manifest=None
):
    path = calibration_path(root, cfg, output_dir)
    payload = json.loads(path.read_text(encoding="utf-8"))
    return validate_calibration(
        payload, calibration_identity(root, cfg, names, checkpoint, manifest), names
    )


def detection_predictor_class(confidence, class_thresholds):
    from ultralytics.models.yolo.detect import DetectionPredictor
    from ultralytics.nn.autobackend import AutoBackend
    from ultralytics.utils import ops

    class ProjectDetectionPredictor(DetectionPredictor):
        def setup_model(self, model, verbose: bool = True):
            """使用项目已解析设备，避免改写进程的CUDA可见设备。"""
            device = str(self.args.device)
            device = torch.device(f"cuda:{device}" if device.isdecimal() else device)
            self.model = AutoBackend(
                weights=model or self.args.model,
                device=device,
                dnn=self.args.dnn,
                data=self.args.data,
                fp16=self.args.half,
                batch=self.args.batch,
                fuse=True,
                verbose=verbose,
            )
            self.device = self.model.device
            self.args.half = self.model.fp16
            if hasattr(self.model, "imgsz") and not getattr(self.model, "dynamic", False):
                self.args.imgsz = self.model.imgsz
            self.model.eval()

        def postprocess(self, preds, img, orig_imgs, **kwargs):
            outputs = postprocess_detection(
                preds,
                nc=len(self.model.names),
                names=self.model.names,
                mode=YOLO_LOCKED_MODE,
                confidence=confidence,
                class_thresholds=class_thresholds,
                iou=self.args.iou,
                max_det=self.args.max_det,
                agnostic=self.args.agnostic_nms,
                end2end=getattr(self.model, "end2end", False),
                task=self.args.task,
            )
            if not isinstance(orig_imgs, list):
                orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)
            return self.construct_results(outputs, img, orig_imgs)

    return ProjectDetectionPredictor
