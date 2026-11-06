"""Create immutable training-view recipes from a completed train-only coverage run."""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

from PIL import Image
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from car_smart_assist.config.detection import load_yolo_config
from car_smart_assist.perception.detection_crops import enclosing_window


def write_json(path, payload):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coverage-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=80, help="Training epochs per arm")
    opts = parser.parse_args()
    if opts.epochs < 1:
        parser.error("--epochs must be positive")
    run, output = opts.coverage_run.resolve(), opts.output_dir.resolve()
    cfg = load_yolo_config(ROOT / "configs/model/yolo_detection.yaml")
    base = (ROOT / cfg["data"]).parent
    manifest = json.loads((base / "manifest.json").read_bytes())
    expected = {r["image"]: r for r in manifest["images"] if r["split"] == "train"}
    terminal = json.loads((run / "terminal.json").read_bytes())
    coverage = json.loads((run / "profile/coverage.json").read_bytes())
    weight = cfg["checkpoint"]
    initial = hashlib.sha256((ROOT / weight).read_bytes()).hexdigest()
    if (terminal["status"] != "COMPLETED" or terminal["changed_files"] or coverage["split"] != "train"
            or coverage["dataset_signature"] != manifest["signature"] or coverage["checkpoint_sha256"] != initial):
        raise ValueError("Only audited train coverage from the specified starting weights is admissible")
    windows, seen = {}, set()
    eligible, selected, removed, scales = Counter(), Counter(), Counter(), Counter()
    with (run / "profile/train_targets.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            key = record["image"]
            if key in seen or key not in expected:
                raise ValueError("Unexpected training frame: " + key)
            seen.add(key)
            reasons = [t for t in record["targets"] if t["size"] == "tiny_lt16" and
                       t["state"] in {"transform_dropped", "no_center"}]
            if not reasons:
                continue
            row = expected[key]
            group = row["condition"]
            eligible[group] += 1
            with Image.open(base / key) as image:
                size = image.size
            window = enclosing_window(size, [b[1:] for b in row["boxes"]])
            if window is None:
                continue
            windows[key] = {"size": list(size), "xyxy": window}
            selected[group] += 1
            removed[group] += len(reasons)
            factor = size[0] / (window[2] - window[0])
            scales["2x" if factor > 1.75 else "1.5x"] += 1
    if seen != set(expected) or not windows:
        raise ValueError("Incomplete source coverage or no admissible crops")
    output.mkdir(parents=True, exist_ok=False)
    plan = output / "view_plan.json"
    write_json(plan, {"version": 1, "imgsz": cfg["train"]["imgsz"], "dataset_signature": manifest["signature"],
                      "initial_sha256": initial, "windows": windows})
    identity = hashlib.sha256(plan.read_bytes()).hexdigest()
    for mode in ("full", "crop"):
        arm = deepcopy(cfg)
        # The paired test relocates these isolated output roots again for every new attempt.
        arm.update(model=weight, project=str(output / "models"), name=mode, device="cuda:0",
                   checkpoint=str(output / "models" / mode / "weights/best.pt"),
                   calibration_file=str(output / "calibration" / mode / "calibration.json"),
                   training_view={"mode": mode, "plan": str(plan), "sha256": identity})
        # Avoid unequal AMP overflow skips; both new runs use the same FP32 recipe.
        arm["train"].update(epochs=opts.epochs, patience=0, amp=False)
        with (output / (mode + ".yaml")).open("x", encoding="utf-8") as stream:
            yaml.safe_dump({"yolo_detection": arm}, stream, allow_unicode=True, sort_keys=False)
    summary = {"epochs_per_arm": opts.epochs, "train_images": len(seen), "planned_crops": len(windows), "full_views_in_crop_arm": len(seen)-len(windows),
               "candidate_frames_by_condition": eligible, "planned_by_condition": selected,
               "targeted_geometry_misses_in_selected_frames": removed, "magnification": scales,
               "plan_sha256": identity, "initial_weight": weight, "initial_sha256": initial}
    write_json(output / "pair_recipe.json", summary)
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
