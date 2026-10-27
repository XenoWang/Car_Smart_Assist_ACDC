"""按场景分组、互不重叠地准备 ACDC 检测数据，输出 Ultralytics YOLO 格式。"""

from __future__ import annotations

import hashlib
import itertools
import json
import logging
import math
import os
import random
import re
import shutil
import tempfile
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import numpy as np
import yaml

from car_smart_assist.data.preprocessing import probe_image_integrity


@dataclass(frozen=True)
class DetectionImage:
    relative_path: str
    source: Path
    width: int
    height: int
    condition: str
    sequence: str
    group: str
    labels: tuple[tuple[float, ...], ...]
    annotation_ids: tuple[int, ...]
    crowd: bool = False
    sha256: str = ""


def load_detection_data_config(path: Path) -> dict[str, Any]:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))["detection_data"]
    ratios = cfg["split_ratios"]
    if tuple(ratios) != ("train", "val", "calibration", "test"):
        raise ValueError("split_ratios must define train/val/calibration/test in that order")
    if any(isinstance(v, bool) or not math.isfinite(v) or v <= 0 for v in ratios.values()):
        raise ValueError("Split ratios must be positive finite numbers")
    if not math.isclose(sum(ratios.values()), 1.0, abs_tol=1e-9):
        raise ValueError("Split ratios must sum to 1")
    if type(cfg["seed"]) is not int or type(cfg["integrity_workers"]) is not int:
        raise ValueError("seed and integrity_workers must be integers")
    if cfg["integrity_workers"] < 1:
        raise ValueError("integrity_workers must be positive")
    ids = [item["source_id"] for item in cfg["classes"]]
    names = [item["name"] for item in cfg["classes"]]
    if not ids or len(set(ids)) != len(ids) or len(set(names)) != len(names):
        raise ValueError("Detection classes must have unique source IDs and names")
    if cfg["crowd_policy"] != "exclude_image":
        raise ValueError("This YOLO exporter requires crowd_policy=exclude_image")
    if cfg["image_mode"] not in ("hardlink", "copy"):
        raise ValueError("image_mode must be hardlink or copy")
    if cfg["group_strategy"] not in ("recording_family", "sequence"):
        raise ValueError("group_strategy must be recording_family or sequence")
    return cfg


def normalize_box(bbox: list[float], width: int, height: int) -> tuple[float, ...]:
    """COCO 像素 xywh -> YOLO 归一化 xywh；保留有效的小框/截断框。"""
    if len(bbox) != 4 or width <= 0 or height <= 0:
        raise ValueError("Invalid image dimensions or bbox length")
    x, y, w, h = (float(value) for value in bbox)
    if not all(math.isfinite(value) for value in (x, y, w, h)) or w <= 0 or h <= 0:
        raise ValueError(f"Invalid bbox: {bbox}")
    x1, y1 = max(0.0, x), max(0.0, y)
    x2, y2 = min(float(width), x + w), min(float(height), y + h)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"BBox lies entirely outside the image: {bbox}")
    return (
        (x1 + x2) / (2 * width),
        (y1 + y2) / (2 * height),
        (x2 - x1) / width,
        (y2 - y1) / height,
    )


def recording_group(sequence: str, strategy: str) -> str:
    if strategy == "recording_family":
        match = re.fullmatch(r"(?:GOPR|GP\d{2})(\d{4})", sequence)
        if match:
            return "gopro:" + match.group(1)
    return "sequence:" + sequence


def read_detection_annotations(
    project_root: Path, cfg: dict[str, Any]
) -> tuple[list[DetectionImage], dict[str, str], dict[str, int]]:
    acdc = (project_root / cfg["acdc_root"]).resolve()
    rgb_root = (acdc / "rgb_anon").resolve()
    class_map = {item["source_id"]: index for index, item in enumerate(cfg["classes"])}
    expected_names = {item["source_id"]: item["name"] for item in cfg["classes"]}
    records: list[DetectionImage] = []
    paths: set[str] = set()
    source_hashes: dict[str, str] = {}
    counts: Counter = Counter()
    for relative_annotation in cfg["annotation_files"]:
        path = acdc / relative_annotation
        payload = path.read_bytes()
        source_hashes[relative_annotation] = hashlib.sha256(payload).hexdigest()
        document = json.loads(payload)
        actual_names = {row["id"]: row["name"] for row in document["categories"]}
        if actual_names != expected_names:
            raise ValueError(f"Category IDs/names do not match configured mapping: {path}")
        image_rows = {row["id"]: row for row in document["images"]}
        if len(image_rows) != len(document["images"]):
            raise ValueError(f"Duplicate image IDs in {path}")
        annotations: dict[int, list[dict]] = defaultdict(list)
        annotation_ids: set[int] = set()
        for ann in document["annotations"]:
            if ann["id"] in annotation_ids or ann["image_id"] not in image_rows:
                raise ValueError(f"Duplicate/orphan annotation in {path}: {ann['id']}")
            annotation_ids.add(ann["id"])
            if ann["category_id"] not in class_map or ann.get("iscrowd", 0) not in (0, 1):
                raise ValueError(f"Invalid annotation class/crowd flag: {ann['id']}")
            annotations[ann["image_id"]].append(ann)
        for image_id, row in image_rows.items():
            relative = PurePosixPath(row["file_name"])
            if relative.is_absolute() or ".." in relative.parts or len(relative.parts) != 4:
                raise ValueError(f"Unexpected ACDC image path: {relative}")
            condition, official_split, sequence, _ = relative.parts
            if condition not in ("fog", "night", "rain", "snow") or official_split not in (
                "train",
                "val",
            ):
                raise ValueError(
                    f"Only labeled adverse train/val images can be prepared: {relative}"
                )
            source = (rgb_root / str(relative)).resolve()
            if not source.is_relative_to(rgb_root) or not source.is_file():
                raise FileNotFoundError(f"Image missing or outside ACDC root: {source}")
            if str(relative) in paths:
                raise ValueError(f"Image occurs in both annotation sources: {relative}")
            paths.add(str(relative))
            labels: list[tuple[float, ...]] = []
            anns = sorted(annotations[image_id], key=lambda item: item["id"])
            for ann in anns:
                box = normalize_box(ann["bbox"], row["width"], row["height"])
                labels.append((class_map[ann["category_id"]], *box))
                counts["source_boxes"] += 1
                counts["crowd_boxes"] += int(ann.get("iscrowd", 0))
                counts["source_boxes_below_20px"] += int(ann["bbox"][3] < 20)
                x, y, w, h = ann["bbox"]
                counts["clipped_boxes"] += int(
                    x < 0 or y < 0 or x + w > row["width"] or y + h > row["height"]
                )
            records.append(
                DetectionImage(
                    relative_path=str(relative),
                    source=source,
                    width=row["width"],
                    height=row["height"],
                    condition=condition,
                    sequence=sequence,
                    group=recording_group(sequence, cfg["group_strategy"]),
                    labels=tuple(labels),
                    annotation_ids=tuple(ann["id"] for ann in anns),
                    crowd=any(ann.get("iscrowd", 0) for ann in anns),
                )
            )
    return sorted(records, key=lambda item: item.relative_path), source_hashes, dict(counts)


def _verify_record(record: DetectionImage) -> tuple[DetectionImage, str | None]:
    result = probe_image_integrity(record.source)
    if not result["ok"]:
        if not result.get("corrupt", False):
            raise OSError(f"Image read incomplete; cannot exclude as corrupt: {record.source}: {result['error']}")
        return record, result["error"]
    if (result["width"], result["height"]) != (record.width, record.height):
        raise ValueError(f"Image/annotation dimensions disagree: {record.source}")
    digest = hashlib.sha256()
    with record.source.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return replace(record, sha256=digest.hexdigest()), None


def _merge_duplicate_groups(records: list[DetectionImage]) -> list[DetectionImage]:
    """把字节完全相同的图像分到同一组，不删除它们。"""
    parent = {record.group: record.group for record in records}

    def find(group):
        while parent[group] != group:
            parent[group] = parent[parent[group]]
            group = parent[group]
        return group

    hashes: dict[str, DetectionImage] = {}
    for record in records:
        if record.sha256 in hashes:
            previous = hashes[record.sha256]
            if previous.condition != record.condition:
                raise ValueError(
                    "Identical RGB bytes occur under different weather labels; review before splitting"
                )
            a, b = find(record.group), find(previous.group)
            parent[max(a, b)] = min(a, b)
        else:
            hashes[record.sha256] = record
    return [replace(record, group=find(record.group)) for record in records]


def stratified_group_split(
    records: list[DetectionImage], ratios: dict[str, float], seed: int, class_count: int
) -> dict[str, list[DetectionImage]]:
    """按种子做整段录制划分，让图像和框类别尽量均衡。"""
    split_names = tuple(ratios)
    target = np.asarray(list(ratios.values()))
    rng = random.Random(seed)
    by_group: dict[str, list[DetectionImage]] = defaultdict(list)
    for record in records:
        by_group[record.group].append(record)
    weather_groups: dict[str, list[str]] = defaultdict(list)
    vectors = {}
    for name, rows in sorted(by_group.items()):
        conditions = {record.condition for record in rows}
        if len(conditions) != 1:
            raise ValueError(f"Recording group crosses weather labels: {name}")
        weather_groups[rows[0].condition].append(name)
        vector = np.zeros(1 + class_count)
        vector[0] = len(rows)
        for record in rows:
            for box in record.labels:
                vector[1 + int(box[0])] += 1
        vectors[name] = vector

    def score(stats):
        totals = stats.sum(axis=1, keepdims=True)
        fractions = np.divide(stats, totals, out=np.zeros_like(stats), where=totals > 0)
        error = (fractions - target[None, :, None]) ** 2
        result = 4 * error[:, :, 0].sum(axis=1) + error[:, :, 1:].mean(axis=(1, 2))
        result += 0.5 * (stats[:, 0, 1:] == 0).sum(axis=1)
        result += 0.03 * (stats[:, 1:, 1:] == 0).sum(axis=(1, 2))
        return result

    candidates = {}
    choices = {}
    for condition, groups in sorted(weather_groups.items()):
        if len(groups) < len(split_names):
            raise ValueError(
                f"{condition} only has {len(groups)} recording groups; cannot cover four splits without leakage"
            )
        rng.shuffle(groups)
        if len(groups) <= 8:
            assignments = [
                row for row in itertools.product(range(4), repeat=len(groups)) if len(set(row)) == 4
            ]
        else:
            assignments = []
            for _ in range(10000):
                row = tuple(rng.choices(range(4), weights=target, k=len(groups)))
                if len(set(row)) == 4:
                    assignments.append(row)
            if not assignments:
                row = list(range(4))
                row.extend(rng.choices(range(4), weights=target, k=len(groups) - 4))
                rng.shuffle(row)
                assignments.append(tuple(row))
        rng.shuffle(assignments)
        assignment_array = np.asarray(assignments, dtype=np.int8)
        selectors = assignment_array[:, :, None] == np.arange(4)
        counts = np.einsum(
            "kgs,gd->ksd", selectors, np.asarray([vectors[group] for group in groups])
        )
        candidates[condition] = (groups, assignment_array, counts)
        choices[condition] = int(np.argmin(score(counts)))
    for _ in range(6):
        changed = False
        conditions = sorted(candidates)
        rng.shuffle(conditions)
        for condition in conditions:
            counts = candidates[condition][2]
            other = sum(
                (candidates[c][2][choices[c]] for c in candidates if c != condition),
                np.zeros_like(counts[0]),
            )
            scores = score(counts + other[None]) + 0.25 * score(counts)
            selected = int(np.argmin(scores))
            changed |= selected != choices[condition]
            choices[condition] = selected
        if not changed:
            break
    partitions = {name: [] for name in split_names}
    for condition, (groups, assignments, _) in candidates.items():
        for group, assignment in zip(groups, assignments[choices[condition]], strict=True):
            partitions[split_names[int(assignment)]].extend(by_group[group])
    return {
        name: sorted(rows, key=lambda item: item.relative_path) for name, rows in partitions.items()
    }


def _label_text(labels: tuple[tuple[float, ...], ...]) -> str:
    return "".join(
        f"{int(row[0])} " + " ".join(f"{value:.10f}" for value in row[1:]) + "\n" for row in labels
    )


def _materialize_image(source: Path, destination: Path, mode: str, digest: str) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if hashlib.sha256(destination.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Existing exported image differs from source: {destination}")
        return "existing"
    if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
        raise ValueError(f"Source image content changed: {source}")
    if mode == "hardlink":
        try:
            os.link(source, destination)
            return "hardlink"
        except OSError:
            pass
    shutil.copy2(source, destination)
    return "copy"


def _empty_output_identity(output: Path):
    if output.is_symlink() or getattr(output, "is_junction", lambda: False)():
        raise ValueError("Detection output must be a real directory, not a symbolic link or junction")
    if output.exists() and output.is_dir() and not any(output.iterdir()):
        current = output.lstat()
        return current.st_dev, current.st_ino, getattr(current, "st_birthtime_ns", None)
    return None


def _publish_detection_staging(staging: Path, output: Path, *, empty_identity=None) -> None:
    """Publish once, tolerating brief Windows locks without replacing another export."""
    if staging.parent != output.parent:
        raise ValueError("Detection staging and output must share a parent directory")
    identity = staging.lstat()
    if empty_identity is not None:
        from car_smart_assist.data.joint_paths import remove_original_empty_output

        remove_original_empty_output(output, empty_identity)
    for attempt in range(6):
        if output.exists() or output.is_symlink():
            raise ValueError("Detection output appeared during preparation; choose a new output_root")
        current = staging.lstat()
        if staging.is_symlink() or (current.st_dev, current.st_ino) != (identity.st_dev, identity.st_ino):
            raise ValueError("Detection staging identity changed; refusing publication")
        try:
            staging.rename(output)
            return
        except OSError as exc:
            if getattr(exc, "winerror", None) not in (5, 32, 33) or attempt == 5:
                raise
            logging.warning(
                "Windows temporarily denied detection publication (error %s); retry %d/5",
                exc.winerror, attempt + 1,
            )
            time.sleep(0.05 * 2 ** attempt)


def validate_detection_export(
    output: Path, verify_hashes: bool = False, *, lost_and_found_root=None
) -> dict[str, int]:
    from car_smart_assist.data.joint_paths import resolve_joint_manifest

    manifest = resolve_joint_manifest(
        json.loads((output / "manifest.json").read_text(encoding="utf-8")), lost_and_found_root
    )
    groups: dict[str, str] = {}
    hashes: dict[str, str] = {}
    files: set[str] = set()
    counts = Counter()
    expected_lists: dict[str, list[str]] = defaultdict(list)
    for row in manifest["images"]:
        split = row["split"]
        for key, registry in (("group", groups), ("sha256", hashes)):
            value = row[key]
            if value in registry and registry[value] != split:
                raise ValueError(f"{key} leakage across splits: {value}")
            registry[value] = split
        if row["image"] in files:
            raise ValueError("Duplicate exported image")
        files.add(row["image"])
        image_path = output / row["image"]
        label_path = output / row["label"]
        if not image_path.is_file() or not label_path.is_file():
            raise FileNotFoundError(f"Missing image or label: {image_path}")
        if label_path.read_text(encoding="utf-8") != _label_text(
            tuple(tuple(box) for box in row["boxes"])
        ):
            raise ValueError(f"Label contents no longer match converted annotations: {label_path}")
        if verify_hashes and hashlib.sha256(image_path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError(f"Exported image content changed: {image_path}")
        if manifest.get("joint_supervision"):
            known = manifest["known_class_count"]
            expected = list(range(known)) if row["dataset"] == "acdc" else [known]
            if row["supervised_classes"] != expected:
                raise ValueError("Joint annotation coverage changed")
            if any(int(box[0]) not in expected for box in row["boxes"]):
                raise ValueError("Joint labels contain an unsupported class for this source")
            if row["dataset"] == "lost_and_found":
                roi = Path(row["roi_mask"])
                instances = roi.with_name(roi.name.removesuffix("_labelTrainIds.png") + "_instanceIds.png")
                if not roi.name.endswith("_gtCoarse_labelTrainIds.png"):
                    raise ValueError("Joint ROI does not follow the Lost & Found naming contract")
                for path, key in ((roi, "roi_sha256"), (instances, "instance_sha256")):
                    if not path.is_file():
                        raise FileNotFoundError(f"Missing obstacle supervision: {path}; supply explicit current joint-data config after relocation")
                    if verify_hashes and hashlib.sha256(path.read_bytes()).hexdigest() != row[key]:
                        raise ValueError(f"Joint supervision contents changed: {path}")
                if lost_and_found_root is not None:
                    source = Path(row["source"])
                    if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != row["sha256"]:
                        raise ValueError(f"Mapped joint source is missing or changed: {source}")
        expected_lists[split].append("./" + row["image"])
        counts[split] += 1
    for split, expected in expected_lists.items():
        actual = (output / f"{split}.txt").read_text(encoding="utf-8").splitlines()
        if actual != expected:
            raise ValueError(f"Image list does not match fixed split: {split}")
    return dict(counts)


def prepare_detection_data(project_root: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    requested_output = project_root / cfg["output_root"]
    if requested_output.is_symlink() or getattr(requested_output, "is_junction", lambda: False)():
        raise ValueError("Detection output must be a real directory, not a symbolic link or junction")
    output = requested_output.resolve()
    empty_identity = _empty_output_identity(output)
    processed_root = (project_root / "data/processed").resolve()
    if output == processed_root or not output.is_relative_to(processed_root):
        raise ValueError("Detection export must be under data/processed")
    official = json.loads(
        (project_root / cfg["acdc_root"] / cfg["official_test_index"]).read_text(encoding="utf-8")
    )
    if (
        not isinstance(official, dict) or not isinstance(official.get("images"), list)
        or official.get("annotations", []) != []
    ):
        raise ValueError("Official inference-only index must contain images and no annotations")
    rgb_root = (project_root / cfg["acdc_root"] / "rgb_anon").resolve()
    official_paths = []
    seen_official = set()
    for row in official["images"]:
        file_name = row.get("file_name") if isinstance(row, dict) else None
        if not isinstance(file_name, str) or not file_name.strip() or "\x00" in file_name:
            raise ValueError("Official inference image file_name must be a nonempty relative path")
        relative = PurePosixPath(file_name.replace("\\", "/"))
        windows = PureWindowsPath(file_name)
        if (
            not relative.parts or relative.is_absolute() or windows.drive or windows.root
            or ".." in relative.parts
        ):
            raise ValueError(f"Invalid official inference image path: {file_name}")
        path = (rgb_root / str(relative)).resolve()
        if not path.is_relative_to(rgb_root) or not path.is_file():
            raise ValueError(f"Official inference image missing or outside rgb_anon: {file_name}")
        if path in seen_official:
            raise ValueError(f"Duplicate official inference image: {file_name}")
        seen_official.add(path)
        official_paths.append(path)
    records, source_hashes, annotation_counts = read_detection_annotations(project_root, cfg)
    logging.info("Checking %d labeled RGB images with shared integrity checks", len(records))
    eligible, excluded = [], []
    with ThreadPoolExecutor(max_workers=cfg["integrity_workers"]) as executor:
        for record, error in executor.map(_verify_record, records):
            if error:
                excluded.append(
                    {"path": record.relative_path, "reason": "corrupt_image", "detail": error}
                )
            elif record.crowd:
                excluded.append(
                    {
                        "path": record.relative_path,
                        "reason": "deferred_crowd_image",
                        "annotation_ids": list(record.annotation_ids),
                    }
                )
            else:
                eligible.append(record)
    if not eligible:
        raise ValueError(f"No eligible detection images: {len(records)} checked, {len(excluded)} excluded")
    eligible = _merge_duplicate_groups(eligible)
    partitions = stratified_group_split(
        eligible, cfg["split_ratios"], cfg["seed"], len(cfg["classes"])
    )
    fingerprint_data = {
        "config": cfg,
        "annotations": source_hashes,
        "images": [
            (split, row.relative_path, row.sha256, row.group, row.labels)
            for split, rows in partitions.items()
            for row in rows
        ],
    }
    signature = hashlib.sha256(json.dumps(fingerprint_data, sort_keys=True).encode()).hexdigest()
    exported = []
    split_lists = {}
    split_stats = {}
    names = {index: row["name"] for index, row in enumerate(cfg["classes"])}
    for split, rows in partitions.items():
        image_list = []
        objects = Counter()
        weather = Counter()
        for row in rows:
            image_relative = PurePosixPath("images") / split / row.relative_path
            label_relative = (PurePosixPath("labels") / split / row.relative_path).with_suffix(
                ".txt"
            )
            image_list.append("./" + str(image_relative))
            weather[row.condition] += 1
            objects.update(names[int(box[0])] for box in row.labels)
            exported.append(
                {
                    "split": split,
                    "image": str(image_relative),
                    "label": str(label_relative),
                    "source": str(row.source),
                    "source_relative": row.relative_path,
                    "condition": row.condition,
                    "sequence": row.sequence,
                    "group": row.group,
                    "sha256": row.sha256,
                    "boxes": row.labels,
                }
            )
        split_lists[f"{split}.txt"] = "\n".join(image_list) + "\n"
        split_stats[split] = {
            "images": len(rows),
            "groups": len({row.group for row in rows}),
            "actual_ratio": len(rows) / len(eligible),
            "weather_images": dict(weather),
            "boxes": sum(objects.values()),
            "class_boxes": {name: objects[name] for name in names.values()},
        }
    dataset = {
        "path": output.as_posix(),
        "train": "train.txt",
        "val": "val.txt",
        "test": "test.txt",
        "names": names,
    }
    manifest = {
        "format_version": 1,
        "signature": signature,
        "seed": cfg["seed"],
        "group_strategy": cfg["group_strategy"],
        "target_ratios": cfg["split_ratios"],
        "source_annotation_sha256": source_hashes,
        "classes": cfg["classes"],
        "images": exported,
        "excluded": excluded,
        "official_inference_only_count": len(official_paths),
    }
    split_document = {
        "seed": cfg["seed"],
        "signature": signature,
        "splits": {
            split: [row.relative_path for row in rows] for split, rows in partitions.items()
        },
    }
    documents = {
        "manifest.json": json.dumps(manifest, ensure_ascii=False, indent=2),
        "split.json": json.dumps(split_document, indent=2),
        "deferred_samples.json": json.dumps(excluded, ensure_ascii=False, indent=2),
    }
    yaml_documents = {
        "data.yaml": dataset,
        "calibration.yaml": {**dataset, "val": "calibration.txt"},
    }
    official_text = "\n".join(path.as_posix() for path in official_paths) + "\n"

    def validate_current(directory):
        for name, expected in documents.items():
            if json.loads((directory / name).read_text(encoding="utf-8")) != json.loads(expected):
                raise ValueError(f"Prepared metadata differs from current inputs: {name}")
        for name, expected in yaml_documents.items():
            if yaml.safe_load((directory / name).read_text(encoding="utf-8")) != expected:
                raise ValueError(f"Prepared dataset YAML differs: {name}")
        if (directory / "official_inference.txt").read_text(encoding="utf-8") != official_text:
            raise ValueError("Prepared official inference list differs from the current index")
        return validate_detection_export(directory, verify_hashes=True)

    materialization = Counter()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        try:
            validated = validate_current(output)
        except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as exc:
            raise ValueError(
                f"Existing detection export is incomplete or differs from current inputs: {exc}; "
                "use a new output_root rather than overwriting existing content"
            ) from exc
        materialization["existing"] = len(eligible)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
        staging_identity = staging.stat()
        try:
            for split, rows in partitions.items():
                for row in rows:
                    image_relative = PurePosixPath("images") / split / row.relative_path
                    label_relative = (PurePosixPath("labels") / split / row.relative_path).with_suffix(".txt")
                    materialization[
                        _materialize_image(
                            row.source, staging / str(image_relative), cfg["image_mode"], row.sha256
                        )
                    ] += 1
                    label_path = staging / str(label_relative)
                    label_path.parent.mkdir(parents=True, exist_ok=True)
                    label_path.write_text(_label_text(row.labels), encoding="utf-8")
            for name, content in {**split_lists, **documents, "official_inference.txt": official_text}.items():
                (staging / name).write_text(content, encoding="utf-8")
            for name, document in yaml_documents.items():
                (staging / name).write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
            validated = validate_current(staging)
            _publish_detection_staging(staging, output, empty_identity=empty_identity)
        finally:
            if staging.exists() or staging.is_symlink():
                current = staging.lstat()
                if (
                    staging.is_symlink() or staging.resolve() != staging
                    or staging == processed_root or not staging.is_relative_to(processed_root)
                    or (current.st_dev, current.st_ino) != (staging_identity.st_dev, staging_identity.st_ino)
                ):
                    raise RuntimeError(f"Refusing cleanup of changed or unsafe staging directory: {staging}")
                shutil.rmtree(staging)
    summary = {
        "dataset": "ACDC instance detection",
        "seed": cfg["seed"],
        "signature": signature,
        "labeled_source_images": len(records),
        "exported_images": len(eligible),
        "deferred_crowd_images": sum(row["reason"] == "deferred_crowd_image" for row in excluded),
        "corrupt_images": sum(row["reason"] == "corrupt_image" for row in excluded),
        "annotation_counts": annotation_counts,
        "exported_small_boxes_below_20px": sum(
            box[4] * row.height < 20 for row in eligible for box in row.labels
        ),
        "splits": split_stats,
        "target_ratios": cfg["split_ratios"],
        "image_materialization": dict(materialization),
        "validated_split_counts": validated,
        "official_inference_only_images": len(official_paths),
        "dataset_yaml": str(output / "data.yaml"),
        "notes": [
            "Calibration is isolated from epoch selection; test is held for final reporting.",
            "Exact ratios may be infeasible when whole recording families must stay together.",
            "Crowd images are deferred, not deleted; small valid boxes and genuine adverse images are retained.",
            "Hardlinked images share raw bytes: exported RGB files must be treated as read-only.",
        ],
    }
    report_root = project_root / cfg["report_root"]
    report_root.mkdir(parents=True, exist_ok=True)
    (report_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# ACDC YOLO detection data",
        "",
        f"Seed: {cfg['seed']}; exported: {len(eligible)}; deferred crowd: {summary['deferred_crowd_images']}.",
        "",
        "| Split | Images | Groups | Boxes | Actual ratio |",
        "|---|---:|---:|---:|---:|",
    ]
    lines.extend(
        f"| {split} | {stats['images']} | {stats['groups']} | {stats['boxes']} | {stats['actual_ratio']:.2%} |"
        for split, stats in split_stats.items()
    )
    lines.extend(["", "| Class | train | val | calibration | test |", "|---|---:|---:|---:|---:|"])
    lines.extend(
        "| "
        + name
        + " | "
        + " | ".join(str(stats["class_boxes"][name]) for stats in split_stats.values())
        + " |"
        for name in names.values()
    )
    lines.extend(["", *summary["notes"]])
    (report_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary
