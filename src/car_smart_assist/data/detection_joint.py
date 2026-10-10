"""把 ACDC replay 和 Lost & Found 障碍物框一起用，标注只覆盖一部分。"""

from __future__ import annotations

import hashlib
import itertools
import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from car_smart_assist.data.detection import (
    _label_text,
    _materialize_image,
    normalize_box,
    validate_detection_export,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def obstacle_boxes(instance_ids: np.ndarray, train_ids: np.ndarray, foreground: int = 2):
    if instance_ids.shape != train_ids.shape or instance_ids.ndim != 2:
        raise ValueError("Lost & Found instance/semantic dimensions differ")
    height, width = train_ids.shape
    foreground_pixels = train_ids == foreground
    boxes = []
    for instance in np.unique(instance_ids[foreground_pixels]):
        if instance < 1000:
            raise ValueError("Obstacle pixels do not have an instance ID")
        y, x = np.where((instance_ids == instance) & foreground_pixels)
        boxes.append(
            normalize_box(
                [
                    int(x.min()),
                    int(y.min()),
                    int(x.max() - x.min() + 1),
                    int(y.max() - y.min() + 1),
                ],
                width,
                height,
            )
        )
    return tuple(boxes)


def split_development_locations(rows, ratios, seed):
    groups = sorted({row["group"] for row in rows})
    names = tuple(ratios)
    if names != ("train", "val", "calibration") or len(groups) < 3:
        raise ValueError("Lost & Found development needs three disjoint location splits")
    values = np.asarray(list(ratios.values()), dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any() or not np.isclose(values.sum(), 1):
        raise ValueError("Development ratios must be positive and sum to one")
    vectors = np.array(
        [
            [
                sum(row["group"] == group for row in rows),
                sum(len(row["boxes"]) for row in rows if row["group"] == group),
            ]
            for group in groups
        ]
    )
    candidates = [
        item for item in itertools.product(range(3), repeat=len(groups)) if len(set(item)) == 3
    ]
    random.Random(seed).shuffle(candidates)
    best, score = None, float("inf")
    for candidate in candidates:
        counts = np.array([vectors[np.array(candidate) == index].sum(axis=0) for index in range(3)])
        error = float(((counts / np.maximum(counts.sum(axis=0), 1) - values[:, None]) ** 2).sum())
        if error < score:
            best, score = candidate, error
    assignments = {group: names[index] for group, index in zip(groups, best, strict=True)}
    return [{**row, "split": assignments[row["group"]]} for row in rows]


def prepare_joint_detection(root: Path, cfg: dict) -> dict:
    if cfg["group_strategy"] != "location":
        raise ValueError("Joint Lost & Found export requires location grouping")
    if not np.isclose(sum(cfg["source_weights"].values()), 1) or any(
        not np.isfinite(value) or value <= 0 for value in cfg["source_weights"].values()
    ):
        raise ValueError("Validation source weights must be positive and sum to one")
    if any(
        not np.isfinite(value) or value <= 0
        for value in cfg["classification_loss_weights"].values()
    ):
        raise ValueError("Classification loss source weights must be positive")
    base_path = root / cfg["acdc_manifest"]
    base = json.loads(base_path.read_text(encoding="utf-8"))
    validate_detection_export(base_path.parent)
    if base["seed"] != cfg["seed"]:
        raise ValueError("Joint and ACDC split seeds differ")
    output = (root / cfg["output_root"]).resolve()
    if not output.is_relative_to((root / "data/processed").resolve()):
        raise ValueError("Joint export must be under data/processed")
    classes = [*base["classes"], cfg["additional_class"]]
    obstacle_class = len(base["classes"])
    rows = [
        {
            **row,
            "dataset": "acdc",
            "source_relative": "acdc/" + row["source_relative"],
            "group": "acdc:" + row["group"],
            "roi_mask": None,
            "supervised_classes": list(range(obstacle_class)),
        }
        for row in base["images"]
    ]
    dataset = (root / cfg["lost_and_found_root"]).resolve()
    cleaning = json.loads((root / cfg["cleaning_manifest"]).read_text(encoding="utf-8"))
    invalid = {Path(path).resolve() for path in cleaning["invalid"]}
    deferred = []
    candidates = []
    for image in sorted((dataset / "leftImg8bit").rglob("*_leftImg8bit.png")):
        relative = image.relative_to(dataset / "leftImg8bit")
        split, location = relative.parts[:2]
        if split not in ("train", "test"):
            raise ValueError("Unexpected Lost & Found source split")
        prefix = relative.name.removesuffix("_leftImg8bit.png")
        mask = dataset / "gtCoarse" / relative.parent / (prefix + "_gtCoarse_labelTrainIds.png")
        instances = mask.with_name(prefix + "_gtCoarse_instanceIds.png")
        if any(path.resolve() in invalid for path in (image, mask, instances)):
            deferred.append(str(relative))
            continue
        if not mask.is_file() or not instances.is_file():
            raise FileNotFoundError(f"Missing annotation pair: {relative}")
        candidates.append((image, mask, instances, split, location, relative))

    def convert(item):
        image, mask, instances, split, location, relative = item
        with Image.open(mask) as value:
            train_ids = np.asarray(value)
        with Image.open(instances) as value:
            instance_ids = np.asarray(value)
        height, width = train_ids.shape
        with Image.open(image) as value:
            if value.size != (width, height):
                raise ValueError(f"Image/mask dimensions differ: {relative}")
        return {
            "dataset": "lost_and_found",
            "source": str(image),
            "source_relative": "lost_and_found/" + relative.as_posix(),
            "official_split": split,
            "split": "test" if split == "test" else "development",
            "group": "lost_and_found:" + location,
            "sequence": location,
            "condition": "lost_and_found",
            "sha256": sha256(image),
            "roi_mask": str(mask),
            "roi_sha256": sha256(mask),
            "instance_sha256": sha256(instances),
            "supervised_classes": [obstacle_class],
            "boxes": [
                (obstacle_class, *box)
                for box in obstacle_boxes(instance_ids, train_ids, cfg["obstacle_train_id"])
            ],
        }

    with ThreadPoolExecutor(max_workers=cfg["integrity_workers"]) as executor:
        laf = list(executor.map(convert, candidates))
    training_locations = {row["group"] for row in laf if row["official_split"] == "train"}
    test_locations = {row["group"] for row in laf if row["official_split"] == "test"}
    if training_locations & test_locations:
        raise ValueError("Official train/test share locations")
    rows += split_development_locations(
        [row for row in laf if row["official_split"] == "train"],
        cfg["development_ratios"],
        cfg["seed"],
    )
    rows += [row for row in laf if row["official_split"] == "test"]
    rows.sort(key=lambda row: (row["split"], row["source_relative"]))
    signature = hashlib.sha256(
        json.dumps(
            {"config": cfg, "base_signature": base["signature"], "images": rows}, sort_keys=True
        ).encode()
    ).hexdigest()
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text())["signature"] != signature:
        raise ValueError("Joint dataset changed; choose a new output_root")
    output.mkdir(parents=True, exist_ok=True)
    for row in rows:
        image = Path("images") / row["split"] / row["source_relative"]
        label = (Path("labels") / row["split"] / row["source_relative"]).with_suffix(".txt")
        _materialize_image(Path(row["source"]), output / image, cfg["image_mode"], row["sha256"])
        text = _label_text(row["boxes"])
        destination = output / label
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_file() and destination.read_text() != text:
            raise ValueError("Existing joint labels changed")
        destination.write_text(text, encoding="utf-8")
        row.update(image=image.as_posix(), label=label.as_posix())
    for split in ("train", "val", "calibration", "test"):
        (output / f"{split}.txt").write_text(
            "".join("./" + row["image"] + "\n" for row in rows if row["split"] == split)
        )
    manifest = {
        "format_version": 2,
        "signature": signature,
        "seed": cfg["seed"],
        "classes": classes,
        "joint_supervision": True,
        "known_class_count": obstacle_class,
        "valid_roi_ids": cfg["valid_roi_ids"],
        "source_weights": cfg["source_weights"],
        "classification_loss_weights": cfg["classification_loss_weights"],
        "images": rows,
        "deferred_corrupt": deferred,
        "base_signature": base["signature"],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    data = {
        "path": output.as_posix(),
        "train": "train.txt",
        "val": "val.txt",
        "test": "test.txt",
        "names": {i: item["name"] for i, item in enumerate(classes)},
        "joint_manifest": str(manifest_path),
    }
    for name, value in (
        ("data.yaml", data),
        ("calibration.yaml", {**data, "val": "calibration.txt"}),
    ):
        (output / name).write_text(yaml.safe_dump(value), encoding="utf-8")
    summary = {
        "signature": signature,
        "counts": validate_detection_export(output),
        "sources": {
            source: {
                split: {
                    "images": sum(
                        row["dataset"] == source and row["split"] == split for row in rows
                    ),
                    "boxes": sum(
                        len(row["boxes"])
                        for row in rows
                        if row["dataset"] == source and row["split"] == split
                    ),
                }
                for split in ("train", "val", "calibration", "test")
            }
            for source in ("acdc", "lost_and_found")
        },
        "classes": classes,
        "deferred_corrupt": deferred,
    }
    report = root / cfg["report_root"]
    report.mkdir(parents=True, exist_ok=True)
    (report / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
