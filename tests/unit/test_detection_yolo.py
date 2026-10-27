"""适配器的约定：RGB、原图坐标，以及明确的能力限制。"""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from car_smart_assist.perception.detection_yolo import YoloDetectionPredictor
from car_smart_assist.perception.yolo_postprocess import calibration_identity, calibration_path


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
        "data": "data/data.yaml",
        "inference": {"imgsz": 640, "confidence": 0.25, "iou": 0.5, "max_det": 100},
    }


def _write_valid_calibration(root, cfg):
    checkpoint = root / cfg["checkpoint"]
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"weights")
    data_dir = root / cfg["data"]
    data_dir.parent.mkdir(parents=True, exist_ok=True)
    manifest = {"classes": ["car", "truck"], "signature": "fixture-data"}
    (data_dir.parent / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    identity = calibration_identity(root, cfg, {0: "car", 1: "truck"}, checkpoint, manifest)
    payload = {
        **identity,
        "split": "calibration",
        "passes": [identity["calibration_selection_mode"], identity["mode"]],
        "confidence_threshold": 0.31,
        "class_confidence_thresholds": {"car": 0.31, "truck": 0.31},
        "class_threshold_sources": {"car": "fixture", "truck": "fixture"},
    }
    path = calibration_path(root, cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _model(predict):
    return SimpleNamespace(names={0: "car", 1: "truck"}, predictor=object(), predict=predict)


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

    cfg = config()
    _write_valid_calibration(tmp_path, cfg)
    model = _model(predict)
    adapter = YoloDetectionPredictor(model, cfg, tmp_path)
    assert model.predictor is None  # cached default predictor is cleared at adapter setup
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
    cfg = config()
    _write_valid_calibration(tmp_path, cfg)
    adapter = YoloDetectionPredictor(_model(lambda **kwargs: []), cfg, tmp_path)
    assert adapter.confidence == 0.31
    changed_cfg = config()
    changed_cfg["inference"]["iou"] = 0.7
    with pytest.raises(ValueError, match="identity/settings changed"):
        YoloDetectionPredictor(_model(lambda **kwargs: []), changed_cfg, tmp_path)
    (tmp_path / "best.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="identity/settings changed"):
        YoloDetectionPredictor(_model(lambda **kwargs: []), config(), tmp_path)


def test_shared_image_contract_accepts_gray_and_unit_float_and_rejects_bad_inputs_before_yolo(tmp_path):
    cfg = config()
    _write_valid_calibration(tmp_path, cfg)
    sources = []

    def predict(**kwargs):
        sources.append(np.asarray(kwargs["source"]).copy())
        boxes = SimpleNamespace(
            xyxy=ArrayResult([]), conf=ArrayResult([]), cls=ArrayResult([]),
        )
        return [SimpleNamespace(boxes=boxes, names={0: "car", 1: "truck"})]

    adapter = YoloDetectionPredictor(_model(predict), cfg, tmp_path)
    adapter.predict(np.full((20, 30), 85, dtype=np.uint8))
    adapter.predict(np.full((20, 30, 3), 0.5, dtype=np.float32))
    assert len(sources) == 2
    np.testing.assert_array_equal(sources[0], np.full((20, 30, 3), 85, np.uint8))
    np.testing.assert_array_equal(sources[1], np.full((20, 30, 3), 128, np.uint8))
    with pytest.raises(ValueError, match="通道数"):
        adapter.predict(np.zeros((20, 30, 2), dtype=np.uint8))
    with pytest.raises(ValueError, match="value_range"):
        adapter.predict(np.full((20, 30, 3), 128.0, dtype=np.float32))
    assert len(sources) == 2


def test_installed_predictor_postprocess_calls_shared_helper(monkeypatch):
    import car_smart_assist.perception.yolo_postprocess as postprocess_module

    from car_smart_assist.perception.yolo_postprocess import detection_predictor_class

    calls = []
    monkeypatch.setattr(
        postprocess_module,
        "postprocess_detection",
        lambda *args, **kwargs: calls.append((args, kwargs)) or [],
    )
    predictor_type = detection_predictor_class(0.3, {"car": 0.3})
    predictor = predictor_type.__new__(predictor_type)
    predictor.model = SimpleNamespace(names={0: "car"}, end2end=False)
    predictor.args = SimpleNamespace(
        iou=0.5, max_det=100, agnostic_nms=False, task="detect"
    )
    monkeypatch.setattr(predictor_type, "construct_results", lambda self, *args: args[0])

    result = predictor.postprocess(object(), object(), [])

    assert result == []
    assert len(calls) == 1
    assert calls[0][1]["mode"] == "locked"
    assert calls[0][1]["class_thresholds"] == {"car": 0.3}
