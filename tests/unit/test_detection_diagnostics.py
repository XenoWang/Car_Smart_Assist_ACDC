"""Oracles: drawn rectangles, exact intersections, coverage rules and size boundaries."""
import json

import numpy as np
import pytest

from car_smart_assist.perception.detection_diagnostics import (
    DetectionDiagnostics, diagnose_frame, diagnostic_validator, size_bucket,
)


def test_hand_drawn_errors_and_two_coordinate_size_buckets():
    collector = DetectionDiagnostics(["car", "obstacle"])
    collector.add(
        image="scene.png", source="acdc", condition="rain",
        gt_boxes=[[0, 0, 10, 10], [30, 0, 50, 20], [100, 0, 140, 40]],
        gt_classes=[0, 0, 1],
        pred_boxes=[[0, 0, 10, 10], [0, 0, 10, 10], [100, 0, 140, 40],
                    [40, 0, 60, 20], [200, 0, 210, 10]],
        pred_classes=[0, 0, 0, 0, 1], scores=[0.9, 0.8, 0.7, 0.6, 0.5],
        input_areas=[25, 100, 400],
    )
    report = collector.report()
    counts = report["overall"]
    assert (counts["tp"], counts["fp"], counts["fn"], counts["support"]) == (1, 4, 2, 3)
    assert counts["precision"] == pytest.approx(0.2)
    assert counts["recall"] == pytest.approx(1 / 3)
    assert counts["f1"] == pytest.approx(0.25)
    assert [r["kind"] for r in report["frames"][0]["predictions"]] == [
        "matched", "duplicate_overlap", "wrong_class_overlap", "localization_overlap", "other_unmatched",
    ]
    # 20*20 overlap with a 10-pixel horizontal shift: intersection 200, union 600.
    assert report["frames"][0]["predictions"][3]["max_same_class_iou"] == pytest.approx(1 / 3)
    assert counts["size_recall"]["native"]["tiny_lt16"] == {"support": 1, "matched": 1, "recall": 1.0}
    assert counts["size_recall"]["input"]["tiny_lt16"] == {"support": 2, "matched": 1, "recall": 0.5}
    assert report["by_source"]["acdc"]["fn"] == 2
    assert report["by_condition"]["rain"]["fp"] == 4
    assert report["by_class"]["obstacle"]["recall"] == 0
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("area,bucket", [
    (0, "tiny_lt16"), (255, "tiny_lt16"), (256, "small_16_32"),
    (1023, "small_16_32"), (1024, "medium_32_96"), (9215, "medium_32_96"), (9216, "large_ge96"),
])
def test_size_boundary_areas(area, bucket):
    assert size_bucket(area) == bucket


def test_half_overlap_is_inclusive_and_higher_iou_wins_over_confidence():
    half = diagnose_frame([[0, 0, 2, 2]], [0], [[0, 0, 1, 2]], [0], [0.2], input_areas=[4])
    assert half["matches"] == [{"gt": 0, "prediction": 0, "iou": 0.5}]
    competing = diagnose_frame([[0, 0, 2, 2]], [0], [[0, 0, 1, 2], [0, 0, 2, 2]],
                               [0, 0], [0.99, 0.1], input_areas=[4])
    assert competing["matches"] == [{"gt": 0, "prediction": 1, "iou": 1.0}]


def test_tied_rectangles_are_one_to_one_and_deterministic():
    data = diagnose_frame([[0, 0, 2, 2]] * 2, [0, 0], [[0, 0, 2, 2]] * 3,
                          [0, 0, 0], [0.5] * 3, input_areas=[4, 4])
    assert [(r["gt"], r["prediction"]) for r in data["matches"]] == [(0, 0), (1, 1)]
    assert data["predictions"][2]["kind"] == "duplicate_overlap"


def test_translation_preserves_matches_and_counts():
    boxes = np.array([[0, 0, 10, 10], [30, 0, 40, 10]], dtype=float)
    before = diagnose_frame(boxes, [0, 1], boxes[:1], [0], [0.8], input_areas=[100, 100])
    after = diagnose_frame(boxes + 150, [0, 1], boxes[:1] + 150, [0], [0.8], input_areas=[100, 100])
    assert before["matches"] == after["matches"] == [{"gt": 0, "prediction": 0, "iou": 1.0}]
    assert [r["matched"] for r in after["targets"]] == [True, False]


def test_empty_truth_is_null_and_uncovered_classes_do_not_gain_image_exposure():
    collector = DetectionDiagnostics(["car", "obstacle"])
    collector.add(image="empty.png", source="laf", condition="unknown", gt_boxes=[], gt_classes=[],
                  pred_boxes=[[0, 0, 2, 2]], pred_classes=[1], scores=[0.9], input_areas=[],
                  supervised_classes=[1])
    report = collector.report()
    assert report["overall"]["fp"] == 1
    assert report["overall"]["recall"] is None
    assert report["overall"]["size_recall"]["native"]["tiny_lt16"]["recall"] is None
    assert report["by_class"]["car"]["image_count"] == 0
    assert report["by_class"]["obstacle"]["image_count"] == 1
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("change,message", [
    ({"scores": [float("nan")]}, "Scores"),
    ({"scores": [1.1]}, "Scores"),
    ({"input_areas": []}, "Input-space areas"),
    ({"input_areas": [-1]}, "Input-space areas"),
    ({"pred_classes": [-1]}, "Class IDs"),
    ({"pred_classes": [0.5]}, "Class IDs"),
    ({"gt_boxes": [[2, 0, 1, 2]]}, "corners"),
    ({"pred_boxes": [[0, 0, float("inf"), 2]]}, "coordinates"),
])
def test_invalid_geometry_and_probability_are_rejected(change, message):
    args = dict(gt_boxes=[[0, 0, 2, 2]], gt_classes=[0], pred_boxes=[[0, 0, 2, 2]],
                pred_classes=[0], scores=[0.9], input_areas=[4])
    args.update(change)
    with pytest.raises(ValueError, match=message):
        diagnose_frame(**args)


def test_real_validator_filters_roi_and_unknown_classes_before_diagnostics(tmp_path, monkeypatch):
    import torch
    from PIL import Image
    from types import SimpleNamespace
    monkeypatch.setenv("YOLO_CONFIG_DIR", str(tmp_path / "settings"))
    from car_smart_assist.perception.joint_yolo_backend import JointValidator

    image_path = str((tmp_path / "image.png").resolve())
    roi = np.zeros((100, 100), dtype=np.uint8)
    roi[:50, :50] = 1
    mask_path = tmp_path / "roi.png"
    Image.fromarray(roi).save(mask_path)
    collector = DetectionDiagnostics(["car", "obstacle"])
    validator = object.__new__(diagnostic_validator(JointValidator, collector))
    validator.device = torch.device("cpu")
    validator.args = SimpleNamespace(single_cls=False)
    validator.niou = 10
    validator.iouv = torch.linspace(0.5, 0.95, 10)
    validator.manifest = {"valid_roi_ids": [1]}
    validator.index = {image_path: {
        "dataset": "lost_and_found", "condition": "laf", "supervised_classes": [1],
        "roi_mask": str(mask_path), "boxes": [[1, 0.15, 0.15, 0.1, 0.1]],
    }}
    batch = {"batch_idx": torch.tensor([0]), "cls": torch.tensor([[1.0]]),
             "bboxes": torch.tensor([[0.15, 0.15, 0.1, 0.1]]),
             "ori_shape": [(100, 100)], "img": torch.zeros(1, 3, 64, 64),
             "ratio_pad": [((0.64, 0.64), (0, 0))], "im_file": [image_path]}
    prepared = validator._prepare_batch(0, batch)
    prediction = {"bboxes": torch.tensor([[6.4, 6.4, 12.8, 12.8], [40., 40., 45., 45.],
                                          [6.4, 6.4, 12.8, 12.8]]),
                  "cls": torch.tensor([1., 1., 0.]), "conf": torch.tensor([0.8, 0.9, 0.7])}
    filtered = validator._prepare_pred(prediction, prepared)
    official = validator._process_batch(filtered, prepared)
    assert official["tp"].shape == (1, 10)
    assert official["tp"].all()
    report = collector.report()
    assert (report["overall"]["tp"], report["overall"]["fp"], report["overall"]["fn"]) == (1, 0, 0)
    assert len(report["frames"][0]["predictions"]) == 1
    assert report["frames"][0]["targets"][0]["native_area"] == pytest.approx(100, abs=0.001)
    assert report["frames"][0]["targets"][0]["input_area"] == pytest.approx(40.96, abs=0.001)
