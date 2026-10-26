"""共享YOLO后处理与校准身份的安全边界。"""

from __future__ import annotations

import hashlib

import pytest
import torch

from car_smart_assist.config.detection import YOLO_LOCKED_MODE, YOLO_MAX_WH
from car_smart_assist.perception.yolo_postprocess import (
    calibration_identity,
    calibration_path,
    load_calibration,
    postprocess_detection,
    validate_calibration,
)


def _raw(rows: list[list[float]]) -> torch.Tensor:
    """rows are [cx, cy, w, h, class0, ...]."""
    return torch.tensor(rows, dtype=torch.float32).T.unsqueeze(0)


def test_locked_threshold_uses_original_best_class_and_includes_boundary():
    predictions = _raw(
        [
            [20, 20, 10, 10, 0.8, 0.2],  # equality at class-0 threshold survives
            [40, 40, 10, 10, 0.79, 0.78],  # no fallback to eligible class 1
            [60, 60, 10, 10, 0.1, 0.7],  # eligible class 1
        ]
    )

    output = postprocess_detection(
        predictions,
        nc=2,
        mode=YOLO_LOCKED_MODE,
        confidence=0.5,
        class_thresholds={"car": 0.8, "truck": 0.7},
        names={0: "car", 1: "truck"},
        iou=0.5,
        max_det=10,
    )[0]

    assert output[:, 4].tolist() == pytest.approx([0.8, 0.7])
    assert output[:, 5].tolist() == [0.0, 1.0]


def test_ineligible_crowded_class_cannot_use_the_nms_quantity_budget():
    rows = [[float(index * 20), 20.0, 8.0, 8.0, 0.5, 0.1] for index in range(101)]
    # road_obstacle must be the original highest class; class-0=.4 would be truck/car.
    rows.append([20.0, 60.0, 8.0, 8.0, 0.05, 0.4])
    predictions = _raw(rows)

    output = postprocess_detection(
        predictions, nc=2, mode=YOLO_LOCKED_MODE, confidence=0.03,
        class_thresholds={"car": 0.877, "road_obstacle": 0.031},
        names=["car", "road_obstacle"], iou=0.5, max_det=100,
    )[0]

    assert output.shape == (1, 6)
    assert output[0, 4].item() == pytest.approx(0.4)
    assert output[0, 5].item() == 1


def test_nms_hand_calculated_overlap_classes_order_and_max_det():
    # Independent geometry oracle: identical 10x10 boxes have IoU=1; separated
    # boxes have IoU=0. Same-class suppresses, class-aware keeps other classes,
    # agnostic suppresses cross-class duplicates, and max_det keeps top scores.
    rows = [
        [20, 20, 10, 10, 0.90, 0.10],
        [20, 20, 10, 10, 0.80, 0.10],
        [20, 20, 10, 10, 0.10, 0.85],
        [50, 20, 10, 10, 0.70, 0.10],
        [80, 20, 10, 10, 0.10, 0.95],
    ]
    predictions = _raw(rows)

    def run(*, agnostic, max_det):
        return postprocess_detection(
            predictions, nc=2, mode=YOLO_LOCKED_MODE, confidence=0.1,
            class_thresholds={"car": 0.1, "truck": 0.1},
            names=["car", "truck"], iou=0.5, max_det=max_det,
            agnostic=agnostic,
        )[0]

    class_aware = run(agnostic=False, max_det=10)
    assert class_aware[:, 4].tolist() == pytest.approx([0.95, 0.90, 0.85, 0.70])
    assert class_aware[:, 5].tolist() == [1.0, 0.0, 1.0, 0.0]
    assert class_aware[:, :4].tolist() == [
        [75.0, 15.0, 85.0, 25.0], [15.0, 15.0, 25.0, 25.0],
        [15.0, 15.0, 25.0, 25.0], [45.0, 15.0, 55.0, 25.0],
    ]

    agnostic = run(agnostic=True, max_det=10)
    assert agnostic[:, 4].tolist() == pytest.approx([0.95, 0.90, 0.70])
    assert agnostic[:, 5].tolist() == [1.0, 0.0, 0.0]
    top_two = run(agnostic=False, max_det=2)
    assert top_two[:, 4].tolist() == pytest.approx([0.95, 0.90])


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_fp16_class_offset_keeps_nearby_class8_boxes_distinct_without_mutating_input(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA device coverage requested but CUDA is unavailable")
    names = [f"class_{index}" for index in range(9)]
    rows = []
    for cx, score in ((11.0, 0.9), (21.0, 0.8)):
        rows.append([cx, 11.0, 20.0, 20.0, *([0.0] * 8), score])
    predictions = torch.tensor(rows, dtype=torch.float16, device=device).T.unsqueeze(0)
    original = predictions.clone()

    # Reproduce the old FP16 offset arithmetic: adding class-8's 61440 offset
    # quantizes both distinct boxes to identical coordinates at FP16 precision.
    half_boxes = torch.tensor(
        [[1.0, 1.0, 21.0, 21.0], [11.0, 1.0, 31.0, 21.0]],
        dtype=torch.float16,
        device=device,
    )
    class_offsets = torch.full((2, 1), 8, dtype=torch.float16, device=device) * YOLO_MAX_WH
    old_shifted = half_boxes + class_offsets
    assert torch.equal(old_shifted[0], old_shifted[1])

    result = postprocess_detection(
        predictions, nc=9, mode=YOLO_LOCKED_MODE, confidence=0.1,
        class_thresholds={"class_8": 0.1}, names=names, iou=0.5, max_det=10,
    )[0]
    float32_reference = postprocess_detection(
        predictions.float(), nc=9, mode=YOLO_LOCKED_MODE, confidence=0.1,
        class_thresholds={"class_8": 0.1}, names=names, iou=0.5, max_det=10,
    )[0]
    expected_boxes = torch.tensor(
        [[1.0, 1.0, 21.0, 21.0], [11.0, 1.0, 31.0, 21.0]],
        dtype=torch.float32,
        device=device,
    )

    assert result.dtype == torch.float32
    assert result.shape == (2, 6)
    torch.testing.assert_close(result, float32_reference)
    torch.testing.assert_close(result[:, :4], expected_boxes)
    assert torch.equal(predictions, original)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_fp16_score_below_python_threshold_does_not_round_up_for_comparison(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA device coverage requested but CUDA is unavailable")
    predictions = torch.tensor(
        [[[20.0], [20.0], [10.0], [10.0], [0.8]]], dtype=torch.float16, device=device
    )
    original = predictions.clone()
    rounded_threshold = predictions.new_tensor([0.8])[0]
    assert predictions[0, 4, 0] == rounded_threshold
    assert predictions[0, 4, 0].item() == pytest.approx(0.7998046875)

    result = postprocess_detection(
        predictions, nc=1, mode=YOLO_LOCKED_MODE, confidence=0.8,
        class_thresholds={"car": 0.8}, names=["car"], iou=0.5, max_det=10,
    )[0]

    assert result.shape == (0, 6)
    assert torch.equal(predictions, original)


def test_empty_output_and_invalid_raw_shapes_are_explicit():
    output = postprocess_detection(
        torch.empty((2, 6, 0)), nc=2, mode=YOLO_LOCKED_MODE,
        confidence=0.5, class_thresholds={}, names=["car", "truck"],
        iou=0.5, max_det=10,
    )
    assert len(output) == 2
    assert all(item.shape == (0, 6) for item in output)
    with pytest.raises(ValueError, match="raw \\[B, 4 \\+ nc, N\\]"):
        postprocess_detection(
            torch.zeros((1, 6, 4)).transpose(1, 2), nc=2,
            mode=YOLO_LOCKED_MODE, confidence=0.5, class_thresholds={},
            names=["car", "truck"], iou=0.5, max_det=10,
        )


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), True, "0.5"])
def test_raw_scores_and_thresholds_reject_invalid_numeric_values(invalid):
    if invalid is True or invalid == "0.5":
        thresholds = {"car": invalid}
        predictions = _raw([[20, 20, 10, 10, 0.8, 0.2]])
    else:
        thresholds = {}
        predictions = _raw([[20, 20, 10, 10, invalid, 0.2]])
    with pytest.raises(ValueError):
        postprocess_detection(
            predictions, nc=2, mode=YOLO_LOCKED_MODE, confidence=0.5,
            class_thresholds=thresholds, names=["car", "truck"],
            iou=0.5, max_det=10,
        )


def test_calibration_path_versions_and_legacy_identity_fails_closed(tmp_path):
    cfg = {
        "calibration_file": "reports/calibration.json",
        "inference": {"imgsz": 640, "iou": 0.5, "max_det": 10},
    }
    path = calibration_path(tmp_path, cfg)
    assert path == tmp_path / "reports" / "best_class_threshold_v1" / "calibration.json"
    assert calibration_path(
        tmp_path, cfg, "isolated-run/detection"
    ) == tmp_path / "isolated-run" / "detection" / "best_class_threshold_v1" / "calibration.json"
    # 旧格式校准文件必须位于带版本号的真实路径上，才会被 load_calibration 读到；
    # 否则加载器先因文件缺失报 OSError，旧的“身份缺失”拒绝逻辑根本没有被验证。
    path.parent.mkdir(parents=True)
    path.write_text('{"confidence_threshold": 0.5}', encoding="utf-8")
    checkpoint = tmp_path / "weights.pt"
    checkpoint.write_bytes(b"fixture weights")
    with pytest.raises(ValueError, match="identity/settings changed or format is obsolete"):
        load_calibration(
            tmp_path, cfg, ["car"], checkpoint=checkpoint,
            manifest={"classes": ["car"], "signature": "fixture"},
        )


def test_calibration_identity_binds_checkpoint_dataset_and_postprocess(tmp_path):
    checkpoint = tmp_path / "weights.pt"
    checkpoint_bytes = b"known checkpoint bytes\x00v1"
    checkpoint.write_bytes(checkpoint_bytes)
    cfg = {
        "checkpoint": "weights.pt", "data": "data.yaml",
        "inference": {"imgsz": 640, "iou": 0.5, "max_det": 10},
    }
    manifest = {"classes": ["car", "truck"], "signature": "dataset-sha256-4c1"}
    identity = calibration_identity(tmp_path, cfg, ["car", "truck"], checkpoint, manifest)
    expected = {
        "schema_version": 2,
        "postprocess_version": "best_class_threshold_v1",
        "mode": "locked",
        "calibration_selection_mode": "proposal_sweep",
        "ap_mode": "ap_sweep",
        "sweep_confidence": 0.001,
        "checkpoint_sha256": hashlib.sha256(checkpoint_bytes).hexdigest(),
        "dataset_signature": "dataset-sha256-4c1",
        "classes": [{"id": 0, "name": "car"}, {"id": 1, "name": "truck"}],
        "imgsz": 640,
        "nms_iou": 0.5,
        "max_det": 10,
        "max_nms": 30000,
        "max_wh": 7680,
        "agnostic_nms": False,
        "class_selection": "original_best_class_only",
        "threshold_comparison": ">=",
    }
    assert set(identity) == set(expected)
    for key, value in expected.items():
        assert identity[key] == value
    for key, wrong in (
        ("checkpoint_sha256", "0" * 64),
        ("dataset_signature", "other-dataset"),
        ("classes", [{"id": 0, "name": "truck"}, {"id": 1, "name": "car"}]),
        ("imgsz", 320),
        ("nms_iou", 0.6),
        ("max_det", 11),
    ):
        tampered = dict(expected)
        tampered[key] = wrong
        with pytest.raises(ValueError, match="identity/settings changed"):
            validate_calibration(tampered, expected, ["car", "truck"])


@pytest.mark.parametrize("requested,expected", [
    ("cpu", "cpu"), ("0", "cuda:0"), ("1", "cuda:1"),
    ("cuda:2", "cuda:2"), (torch.device("cuda:1"), "cuda:1"),
])
@pytest.mark.parametrize("dynamic", [False, True])
def test_predictor_setup_preserves_visible_devices_and_logical_indices(
    monkeypatch, requested, expected, dynamic
):
    import os
    from types import SimpleNamespace

    from ultralytics.engine import predictor as base_predictor
    from ultralytics.nn import autobackend
    from car_smart_assist.perception.yolo_postprocess import detection_predictor_class

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5")
    captured = {}
    evaluations = []

    def forbidden_select(*args, **kwargs):
        raise AssertionError("project setup must not call the visibility-mutating device selector")

    def backend(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            device=kwargs["device"], fp16=True, imgsz=640, dynamic=dynamic,
            eval=lambda: evaluations.append(True),
        )

    monkeypatch.setattr(base_predictor, "select_device", forbidden_select)
    monkeypatch.setattr(autobackend, "AutoBackend", backend)
    predictor_type = detection_predictor_class(0.25, {})
    instance = predictor_type.__new__(predictor_type)
    instance.args = SimpleNamespace(
        device=requested, model="unused.pt", dnn=False, data="dataset.yaml",
        half=False, batch=3, imgsz=960,
    )
    model = object()
    instance.setup_model(model, verbose=False)
    assert captured == {
        "weights": model, "device": torch.device(expected), "dnn": False,
        "data": "dataset.yaml", "fp16": False, "batch": 3, "fuse": True,
        "verbose": False,
    }
    assert instance.device == torch.device(expected)
    assert instance.args.half is True
    assert instance.args.imgsz == (960 if dynamic else 640)
    assert evaluations == [True]
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "4,5"
