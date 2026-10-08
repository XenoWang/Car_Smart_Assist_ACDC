"""Meaningful regression checks for box conversion, crowd deferral and four-way leakage."""

from __future__ import annotations

import copy
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from car_smart_assist.data.detection import (
    load_detection_data_config,
    normalize_box,
    prepare_detection_data,
    recording_group,
    validate_detection_export,
)


def test_coco_to_yolo_geometry_and_legal_truncation():
    assert normalize_box([10, 20, 100, 50], 200, 100) == pytest.approx((0.3, 0.45, 0.5, 0.5))
    assert normalize_box([-10, 90, 30, 20], 200, 100) == pytest.approx((0.05, 0.95, 0.1, 0.1))
    assert normalize_box([0, 0, 1, 1], 200, 100)[2] > 0
    for bbox in ([0, 0, 0, 3], [0, 0, float("nan"), 3], [250, 10, 20, 10]):
        with pytest.raises(ValueError):
            normalize_box(bbox, 200, 100)


def test_gopro_chapters_are_one_recording_group():
    assert recording_group("GOPR0475", "recording_family") == recording_group(
        "GP020475", "recording_family"
    )
    assert recording_group("GOPR0475", "sequence") != recording_group("GP020475", "sequence")


def write_detection_fixture(root: Path):
    cfg = copy.deepcopy(load_detection_data_config(Path("configs/data/acdc_detection.yaml")))
    cfg["classes"] = [{"source_id": 26, "name": "car"}]
    cfg["integrity_workers"] = 1
    cfg["acdc_root"] = "data/raw/acdc"
    acdc = root / cfg["acdc_root"]
    gt = acdc / "gt_detection"
    gt.mkdir(parents=True)
    number = 0
    raw_hashes = {}
    for source_split in ("train", "val"):
        rows, anns = [], []
        for condition_index, condition in enumerate(("fog", "night", "rain", "snow")):
            for group in range(4):
                sequence = f"{'GOPR' if source_split == 'train' else 'GP01'}{condition_index * 10 + group:04d}"
                relative = f"{condition}/{source_split}/{sequence}/{sequence}_frame_{number:06d}_rgb_anon.png"
                path = acdc / "rgb_anon" / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(np.full((20, 30, 3), number + 10, np.uint8)).save(path)
                raw_hashes[path] = hashlib.sha256(path.read_bytes()).hexdigest()
                rows.append({"id": number, "file_name": relative, "width": 30, "height": 20})
                # A valid negative frame must remain an empty label file.
                if number != 2:
                    anns.append(
                        {
                            "id": number,
                            "image_id": number,
                            "category_id": 26,
                            "iscrowd": 0,
                            "bbox": [1, 2, 4, 3],
                        }
                    )
                number += 1
        # Add a crowd image with an additional normal object: the entire image must be deferred.
        relative = f"fog/{source_split}/GOPR0000/crowd{source_split}_rgb_anon.png"
        path = acdc / "rgb_anon" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.full((20, 30, 3), number + 10, np.uint8)).save(path)
        rows.append({"id": number, "file_name": relative, "width": 30, "height": 20})
        anns.extend(
            [
                {
                    "id": number,
                    "image_id": number,
                    "category_id": 26,
                    "iscrowd": 1,
                    "bbox": [1, 1, 20, 15],
                },
                {
                    "id": number + 1000,
                    "image_id": number,
                    "category_id": 26,
                    "iscrowd": 0,
                    "bbox": [1, 2, 4, 3],
                },
            ]
        )
        number += 1
        (gt / f"instancesonly_{source_split}_gt_detection.json").write_text(
            json.dumps(
                {"images": rows, "annotations": anns, "categories": [{"id": 26, "name": "car"}]}
            ),
            encoding="utf-8",
        )
    inference = acdc / "rgb_anon/fog/test/GOPR9999/unlabeled_rgb_anon.png"
    inference.parent.mkdir(parents=True)
    Image.fromarray(np.zeros((20, 30, 3), np.uint8)).save(inference)
    (gt / "instancesonly_test_image_info.json").write_text(
        json.dumps(
            {
                "images": [{"file_name": "fog/test/GOPR9999/unlabeled_rgb_anon.png"}],
                "annotations": [],
            }
        ),
        encoding="utf-8",
    )
    return cfg, raw_hashes


def test_prepare_is_reproducible_and_preserves_raw_and_global_random_state(tmp_path):
    cfg, raw_hashes = write_detection_fixture(tmp_path)
    random_before = random.getstate()
    numpy_before = np.random.get_state()  # noqa: NPY002 - check legacy global state is preserved
    first = prepare_detection_data(tmp_path, cfg)
    output = tmp_path / cfg["output_root"]
    first_split = (output / "split.json").read_bytes()
    second = prepare_detection_data(tmp_path, cfg)
    assert first_split == (output / "split.json").read_bytes()
    assert first["signature"] == second["signature"]
    assert first["deferred_crowd_images"] == 2
    assert first["exported_images"] == 32
    assert first["corrupt_images"] == 0
    assert first["official_inference_only_images"] == 1
    assert sum(validate_detection_export(output, verify_hashes=True).values()) == 32
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert any(not row["boxes"] for row in manifest["images"])
    assert all("crowd" not in row["image"] for row in manifest["images"])
    for split in ("train", "val", "calibration", "test"):
        assert set(first["splits"][split]["weather_images"]) == {"fog", "night", "rain", "snow"}
    for path, digest in raw_hashes.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    assert random.getstate() == random_before
    numpy_after = np.random.get_state()  # noqa: NPY002 - compare the same global RNG state
    assert numpy_before[0] == numpy_after[0]
    assert np.array_equal(numpy_before[1], numpy_after[1])
    assert numpy_before[2:] == numpy_after[2:]


def test_modified_label_and_split_list_are_detected(tmp_path):
    cfg, _ = write_detection_fixture(tmp_path)
    prepare_detection_data(tmp_path, cfg)
    output = tmp_path / cfg["output_root"]
    row = json.loads((output / "manifest.json").read_text(encoding="utf-8"))["images"][0]
    label = output / row["label"]
    original = label.read_text(encoding="utf-8")
    label.write_text("0 0.5 0.5 1 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Label contents"):
        validate_detection_export(output)
    label.write_text(original, encoding="utf-8")
    (output / "train.txt").write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="Image list"):
        validate_detection_export(output)
