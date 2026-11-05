"""Inspect observable training pixels, targets, coverage masks and recipe identities."""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from car_smart_assist.perception.detection_crops import enclosing_window, load_windows, project_boxes


def test_windows_enclose_all_labels_or_use_whole_frame():
    assert enclosing_window((100, 80), [[.5, .5, .1, .1]]) == [25, 20, 75, 60]
    assert enclosing_window((100, 80), [[.5, .5, .6, .1]]) == [16, 13, 83, 67]
    assert enclosing_window((100, 80), [[.5, .5, .9, .1]]) is None
    assert enclosing_window((100, 80), []) is None
    with pytest.raises(ValueError, match="truncate"):
        project_boxes([[.1, .1, .1, .1]], (100, 80), [25, 20, 75, 60])


def test_verified_plan_cannot_select_validation_or_cross_model_view(tmp_path):
    key = "images/train/a.png"
    plan = {"version": 1, "dataset_signature": "dataset", "imgsz": 960,
            "windows": {key: {"size": [100, 80], "xyxy": [25, 20, 75, 60]}}}
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    cfg = {"train": {"imgsz": 960}, "training_view": {"mode": "crop", "plan": path.name,
           "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}}
    manifest = {"joint_supervision": True, "signature": "dataset",
                "images": [{"split": "train", "image": key, "boxes": [[8, .5, .5, .1, .1]]}]}
    assert load_windows(tmp_path, cfg, manifest) == plan["windows"]
    cfg["training_view"]["mode"] = "full"
    assert load_windows(tmp_path, cfg, manifest) == {}
    manifest["images"][0]["split"] = "val"
    with pytest.raises(ValueError, match="training-only"):
        load_windows(tmp_path, cfg, manifest)
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        load_windows(tmp_path, cfg, manifest)


def test_source_pixels_boxes_and_sparse_roi_use_the_same_crop(tmp_path):
    from car_smart_assist.perception.joint_yolo_backend import JointDataset

    frame, mask = tmp_path / "pixels.png", tmp_path / "roi.png"
    gray = np.arange(64, dtype=np.uint8).reshape(8, 8)
    Image.fromarray(np.stack([gray] * 3, axis=-1)).save(frame)
    roi = np.zeros((8, 8), dtype=np.uint8)
    roi[1, 2] = 1
    roi[7, 7] = 1
    Image.fromarray(roi).save(mask)
    ds = JointDataset.__new__(JointDataset)
    ds.im_files = [str(frame)]
    ds.labels = [dict(im_file=str(frame), normalized=True, bbox_format="xywh", segments=[],
                      bboxes=np.array([[.5, .375, .25, .25]], dtype=np.float32))]
    ds.imgsz, ds.cv2_flag, ds.augment = 4, 1, True
    ds.update_labels_info = lambda label: label
    ds.transforms = lambda label: {"img": torch.as_tensor(label["img"]).permute(2, 0, 1), "bboxes": label["bboxes"]}
    ds.training_windows = {"train": {"size": [8, 8], "xyxy": [2, 1, 6, 5]}}
    ds.index = {str(frame.resolve()): {"image": "train", "roi_mask": str(mask), "dataset": "lost_and_found",
                "supervised_classes": [8], "boxes": [[8, .5, .375, .25, .25]]}}
    ds.manifest = {"valid_roi_ids": [1, 2], "classes": list(range(9)),
                   "classification_loss_weights": {"lost_and_found": 1.5}}
    actual = ds[0]
    np.testing.assert_array_equal(actual["img"][0], [[10, 11, 12, 13], [18, 19, 20, 21],
                                                     [26, 27, 28, 29], [34, 35, 36, 37]])
    np.testing.assert_array_equal(actual["bboxes"], [[.5, .5, .5, .5]])
    np.testing.assert_array_equal(actual["obstacle_roi"], [[1,0,0,0], [0,1,1,0], [0,1,1,0], [0,0,0,0]])
    assert actual["class_supervision"].tolist() == [False] * 8 + [True]
    assert actual["paired_crop_applied"] is True and actual["source_loss_weight"].item() == 1.5
    np.testing.assert_array_equal(ds.labels[0]["bboxes"], [[.5, .375, .25, .25]])


def test_validation_dataset_refuses_training_windows():
    from car_smart_assist.perception.joint_yolo_backend import JointDataset
    with pytest.raises(ValueError, match="restricted to training"):
        JointDataset(training_windows={"frame": {}}, augment=False)
