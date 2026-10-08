"""Adapter contracts: RGB, original-image coordinates and explicit capability limits."""

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from car_smart_assist.perception.detection_yolo import YoloDetectionPredictor


class ArrayResult:
    def __init__(self, values):
        self.values = values

    def cpu(self):
        return self

    def tolist(self):
        return self.values


def config():
    return {
        "device": "cpu",
        "checkpoint": "best.pt",
        "calibration_file": "calibration.json",
        "inference": {"imgsz": 640, "confidence": 0.25, "iou": 0.5, "max_det": 100},
    }


@pytest.mark.parametrize("empty", [False, True])
def test_rgb_boxes_and_unimplemented_obstacle_capability(tmp_path, empty):
    received = {}

    def predict(**kwargs):
        received.update(kwargs)
        return [
            SimpleNamespace(
                names={0: "car", 1: "truck"},
                boxes=SimpleNamespace(
                    xyxy=ArrayResult([] if empty else [[10, 20, 70, 45]]),
                    conf=ArrayResult([] if empty else [0.8]),
                    cls=ArrayResult([] if empty else [1]),
                ),
            )
        ]

    adapter = YoloDetectionPredictor(SimpleNamespace(predict=predict), config(), tmp_path)
    rgb = np.zeros((60, 90, 3), dtype=np.uint8)
    rgb[..., 0] = 255
    result = adapter.predict(rgb)
    assert received["source"].getpixel((0, 0)) == (255, 0, 0)
    assert received["source"].size == (90, 60)
    assert result.object_detection_available
    assert result.to_dict()["object_detection_classes"] == ["car", "truck"]
    assert not result.road_obstacle_detection_available
    if empty:
        assert result.objects == []
    else:
        obj = result.objects[0]
        assert obj.category == "truck"
        assert obj.bbox == (10, 20, 70, 45)
        assert obj.distance_m is None
        assert obj.direction.value == "unknown"


def test_calibration_bound_to_checkpoint(tmp_path):
    (tmp_path / "best.pt").write_bytes(b"weights")
    calibration = tmp_path / "calibration.json"
    calibration.write_text(
        json.dumps(
            {
                "checkpoint_sha256": hashlib.sha256(b"weights").hexdigest(),
                "confidence_threshold": 0.31,
                "nms_iou": 0.5,
                "imgsz": 640,
                "max_det": 100,
            }
        )
    )
    adapter = YoloDetectionPredictor(None, config(), tmp_path)
    assert adapter.confidence == 0.31
    changed_cfg = config()
    changed_cfg["inference"]["iou"] = 0.7
    with pytest.raises(ValueError, match="settings changed"):
        YoloDetectionPredictor(None, changed_cfg, tmp_path)
    (tmp_path / "best.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="different weights"):
        YoloDetectionPredictor(None, config(), tmp_path)


def test_reject_non_rgb_uint8(tmp_path):
    adapter = YoloDetectionPredictor(None, config(), tmp_path)
    with pytest.raises(ValueError, match="RGB uint8"):
        adapter.predict(np.zeros((20, 30), dtype=np.uint8))
    with pytest.raises(ValueError, match="RGB uint8"):
        adapter.predict(np.zeros((20, 30, 3), dtype=np.float32))
