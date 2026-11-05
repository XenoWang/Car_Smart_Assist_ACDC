"""Profile frozen TAL supervision on the configured joint detector's training split."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from car_smart_assist.config.detection import load_yolo_config
from car_smart_assist.perception.detection_diagnostics import size_bucket


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def save(path, payload):
    with path.open("x", encoding="utf-8") as out:
        json.dump(payload, out, ensure_ascii=False, indent=2, allow_nan=False)


def assignment_rows(inside, proposed, assigned_indices, foreground, scores, lengths):
    """Reduce actual assignment masks; background indices are meaningless."""
    n_gt, n_anchor = inside.shape
    if sum(lengths) != n_anchor or proposed.shape != inside.shape:
        raise ValueError("Feature level boundaries differ from native assignment")
    fg = foreground.bool()
    if not torch.isfinite(scores).all() or (scores < 0).any():
        raise ValueError("Invalid native assignment quality")
    if (assigned_indices[fg] < 0).any() or (assigned_indices[fg] >= n_gt).any():
        raise ValueError("Foreground points to an unknown GT")
    hit = torch.arange(n_gt, device=inside.device)[:, None].eq(assigned_indices[None]) & fg
    values = (inside, proposed, hit, hit & (scores > 0))
    columns = torch.stack([torch.stack([part.sum(1) for part in mask.split(lengths, dim=1)], dim=1)
                           for mask in values], dim=1).cpu().tolist()
    qualities = (hit * scores).sum(1).cpu().tolist()
    rows = []
    for (eligible, pre, post, effective), mass in zip(columns, qualities, strict=True):
        state = ("effective" if sum(effective) else "assigned_zero_quality" if sum(post) else
                 "lost_at_conflict" if sum(pre) else "no_proposal" if sum(eligible) else "no_center")
        rows.append({"state": state, "eligible": eligible, "proposed": pre, "assigned": post,
                     "effective": effective, "quality_sum": mass})
    return rows


def add_summary(groups, row, target):
    for group in ("all", "source/" + row["dataset"], "condition/" + row["condition"],
                  "class/" + target["class"], "condition_class/" + row["condition"] + "/" + target["class"]):
        counter = groups[group][target["size"]]
        counter["gt"] += 1
        counter[target["state"]] += 1
        for field in ("eligible", "proposed", "assigned", "effective"):
            for i, n in enumerate(target.get(field, [])):
                counter[f"{field}_P{i + 3}"] += n
        counter["quality_sum"] += target.get("quality_sum", 0.0)


class AssignmentObserver:
    """Observe the native assigner call before and after conflict resolution."""
    def __init__(self, assigner):
        self.assigner = assigner
        self.original = assigner.get_pos_mask
        self.lengths = []
        self.proposals = None
        self.rows = None
        self.gt = None

    def proposals_for(self, *args):
        mask, alignment, overlaps = self.original(*args)
        self.proposals = mask[0].bool().clone()
        return mask, alignment, overlaps

    def completed(self, module, inputs, result):
        _, _, centers, labels, boxes, valid = inputs
        if len(boxes) != 1 or not valid.bool().all():
            raise ValueError("Expected one image with unpadded GT indices")
        self.gt = boxes[0].cpu().numpy(), labels[0, :, 0].long().cpu().numpy()
        if not boxes.shape[1]:
            self.rows = []
            return
        if self.proposals is None:
            raise RuntimeError("Actual native pre-conflict mask was not captured")
        inside = module.select_candidates_in_gts(centers, boxes)[0].bool()
        _, _, scores, foreground, indices = result
        self.rows = assignment_rows(inside, self.proposals, indices[0], foreground[0],
                                    scores[0].sum(-1), self.lengths)
        self.proposals = None


def run(output):
    from ultralytics import YOLO
    from ultralytics.cfg import get_cfg
    from ultralytics.data import dataset as vendor_data
    from ultralytics.data import utils as data_utils
    from ultralytics.data.augment import RandomPerspective
    from ultralytics.utils.torch_utils import init_seeds
    from car_smart_assist.perception.joint_yolo import JointDetectionLoss
    from car_smart_assist.perception.joint_yolo_backend import JointDataset

    if not torch.cuda.is_available():
        raise RuntimeError("This measurement requires CUDA:0")
    cfg = load_yolo_config(ROOT / "configs/model/yolo_detection.yaml")
    weight = ROOT / cfg["checkpoint"]
    weight_hash = digest(weight)
    args = get_cfg(overrides={k: v for k, v in cfg["train"].items() if k != "resume_extra_epochs"})
    init_seeds(args.seed, deterministic=True)
    # Use the trainer's metadata defaults while suppressing its plotting-only font lookup.
    with patch.object(data_utils, "check_font", return_value=None):
        doc = data_utils.check_det_dataset(str(ROOT / cfg["data"]), autodownload=False)
    manifest_path = Path(doc["joint_manifest"])
    manifest = json.loads(manifest_path.read_bytes())
    base = manifest_path.parent
    train = {str((base / r["image"]).resolve()): r for r in manifest["images"] if r["split"] == "train"}
    listing = base / doc["train"]
    paths = [(listing.parent / line).resolve() for line in listing.read_text(encoding="utf-8").splitlines() if line.strip()]
    if (len(paths) != len(train) or {str(p) for p in paths} != set(train)
            or any(p.suffix.lower() != ".png" or p.with_suffix(".npy").exists() for p in paths)):
        raise ValueError("Training list must contain each declared PNG once, without pixel-cache overrides")
    # Rebuild from the PNG/label pairs. The native writer also sets the in-memory version.
    def save_local_cache(prefix, original_path, records, version):
        data_utils.save_dataset_cache_file(prefix, output / "labels.cache", records, version)

    with patch.object(vendor_data, "load_dataset_cache_file", side_effect=FileNotFoundError), \
            patch.object(vendor_data, "save_dataset_cache_file", save_local_cache):
        dataset = JointDataset(img_path=str(listing), imgsz=args.imgsz, batch_size=1,
                               augment=True, hyp=args, rect=False, stride=32, pad=0.0, data=doc, task="detect")
    if len(dataset) != len(train) or {str(Path(p).resolve()) for p in dataset.im_files} != set(train):
        raise ValueError("Incomplete native training dataset")
    network = YOLO(str(weight)).model.to("cuda:0").float().eval()
    network.args = args
    strides = [int(s) for s in network.stride.tolist()]
    if strides != [8, 16, 32] or network.names != doc["names"]:
        raise ValueError("Unexpected checkpoint strides or class order")
    criterion = JointDetectionLoss(network, manifest["known_class_count"], cfg["joint_foreground_exclusive"])
    observer = AssignmentObserver(criterion.base.assigner)
    candidate_filter = RandomPerspective.box_candidates
    filter_calls = []

    def capture_filter(*args, **kwargs):
        kept = candidate_filter(*args, **kwargs)
        filter_calls.append((kwargs["box2"].T.copy(), kept.copy()))
        return kept

    groups = defaultdict(lambda: defaultdict(Counter))
    total_gt = 0
    with ExitStack() as scope:
        scope.enter_context(patch.object(RandomPerspective, "box_candidates", staticmethod(capture_filter)))
        scope.enter_context(patch.object(observer.assigner, "get_pos_mask", observer.proposals_for))
        scope.callback(observer.assigner.register_forward_hook(observer.completed).remove)
        scope.enter_context(torch.inference_mode())
        stream = scope.enter_context((output / "train_targets.jsonl").open("x", encoding="utf-8"))
        for index, name in enumerate(dataset.im_files):
            row = train[str(Path(name).resolve())]
            filter_calls.clear()
            observer.rows = observer.gt = None
            sample = dataset[index]
            if len(filter_calls) != 1:
                raise RuntimeError("Expected exactly one native geometry filter per input")
            boxes, kept = filter_calls[0]
            classes = dataset.labels[index]["cls"].reshape(-1).astype(int)
            if (len(boxes) != len(classes) or len(classes) != len(row["boxes"])
                    or sample["img"].shape != (3, args.imgsz, args.imgsz)):
                raise ValueError("Native prefilter GT or input geometry changed: " + name)
            batch = dataset.collate_fn([sample])
            predictions = network(batch["img"].to("cuda:0", torch.float32) / 255)
            features = predictions[1]
            observer.lengths = [f.shape[-1] * f.shape[-2] for f in features]
            criterion(predictions, batch)
            if observer.rows is None or len(observer.rows) != int(kept.sum()):
                raise ValueError("Transformed GT and measured assignments disagree: " + name)
            actual_boxes, actual_classes = observer.gt
            if (not np.allclose(boxes[kept], actual_boxes, atol=0.001, rtol=0)
                    or not np.array_equal(classes[kept], actual_classes)):
                raise ValueError("GT coordinates or ordering changed after filtering: " + name)
            records = []
            surviving = iter(observer.rows)
            for gt_id, (box, cls, keep) in enumerate(zip(boxes, classes, kept, strict=True)):
                xy = box.tolist()
                area = (xy[2] - xy[0]) * (xy[3] - xy[1])
                rec = {"gt_index": gt_id, "input_xyxy": xy, "class": doc["names"][int(cls)],
                       "size": size_bucket(area), **(next(surviving) if keep else {"state": "transform_dropped"})}
                add_summary(groups, row, rec)
                records.append(rec)
            total_gt += len(records)
            frame = {k: row[k] for k in ("image", "dataset", "condition", "group")}
            stream.write(json.dumps({**frame, "targets": records}, allow_nan=False) + "\n")
    if digest(weight) != weight_hash:
        raise ValueError("Checkpoint identity changed during profiling")
    save(output / "coverage.json", {"split": "train", "images": len(train), "raw_gt": total_gt,
         "groups": groups, "strides": strides, "seed": args.seed, "imgsz": args.imgsz,
         "checkpoint_sha256": weight_hash, "dataset_signature": manifest["signature"],
         "device": torch.cuda.get_device_name(0), "precision": "fp32",
         "model_mode": "eval; training transforms; frozen weights; batch=1; no optimizer"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    output = parser.parse_args().output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    os.environ["YOLO_CONFIG_DIR"] = str(output / "settings")
    os.environ["YOLO_OFFLINE"] = "true"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    run(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
