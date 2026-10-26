"""部分标注不能把没标出的类别或区域当作背景来训练。"""

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from car_smart_assist.data.detection_joint import obstacle_boxes, split_development_locations
from car_smart_assist.perception.joint_yolo import (
    MaskedClassificationLoss,
    preserve_classification_outputs,
)


def test_instance_boxes_use_only_obstacle_train_ids_and_keep_small_objects():
    semantic = np.full((10, 20), 255, dtype=np.uint8)
    instances = np.zeros((10, 20), dtype=np.uint16)
    semantic[2:4, 5:8] = 2
    instances[2:4, 5:8] = 5000
    semantic[6:8, 10:12] = 0
    instances[6:8, 10:12] = 31000
    boxes = obstacle_boxes(instances, semantic)
    assert boxes == ((0.325, 0.3, 0.15, 0.2),)


def test_location_split_reproducible_and_disjoint_without_global_rng_changes():
    rows = [
        {"group": f"location{i}", "boxes": [()] * (i + 1), "source": f"{i}/{frame}"}
        for i in range(4)
        for frame in range(3)
    ]
    ratios = {"train": 0.5, "val": 0.25, "calibration": 0.25}
    a = split_development_locations(rows, ratios, 42)
    b = split_development_locations(rows, ratios, 42)
    assert a == b
    assert {row["split"] for row in a} == set(ratios)
    for location in {row["group"] for row in rows}:
        assert len({row["split"] for row in a if row["group"] == location}) == 1


def test_unannotated_class_and_roi_gradients_are_zero():
    predictions = torch.zeros(2, 3, 3, requires_grad=True)
    targets = torch.zeros_like(predictions)
    loss = MaskedClassificationLoss()
    # ACDC 监督旧类别；Lost & Found 在已知 ROI 内只监督障碍物类。
    loss.valid = torch.tensor(
        [
            [[True, True, False]] * 3,
            [[False, False, True], [False, False, False], [False, False, False]],
        ]
    )
    loss.weights = torch.ones(2, 1, 1)
    targets[1, 2, 2] = 1  # 稀疏 ROI 之外的正样本分配仍要参与训练。
    loss(predictions, targets).sum().backward()
    gradient = predictions.grad
    assert torch.count_nonzero(gradient[0, :, 2]) == 0
    assert torch.count_nonzero(gradient[1, :2, :2]) == 0
    assert torch.all(gradient[1, 2, :2] > 0)  # 标注的货物不是交通类别正样本。
    assert gradient[1, 1, 2] == 0
    assert gradient[1, 0, 2] > 0 and gradient[1, 2, 2] < 0


def test_masked_weighted_bce_matches_hand_value_and_gradient():
    predictions = torch.tensor(
        [[[0.0, math.log(3.0)], [2.0, -2.0]]], requires_grad=True
    )
    targets = torch.tensor([[[1.0, 0.0], [0.0, 0.0]]])
    loss = MaskedClassificationLoss()
    loss.valid = torch.tensor([[[True, False], [False, False]]])
    loss.weights = torch.tensor([[[2.0]]])
    actual = loss(predictions, targets)
    # unreduced BCE保留[batch,anchor,class]；已监督项分别为2ln2和2ln4，
    # 第二anchor被mask。逐元素合同与独立总和都检查，不能误作类维已求和。
    expected = torch.tensor([[[2 * math.log(2), 2 * math.log(4)], [0.0, 0.0]]])
    torch.testing.assert_close(actual, expected)
    assert actual.sum().item() == pytest.approx(2 * math.log(8))
    actual.sum().backward()
    gradient = predictions.grad
    # dBCE/dlogit=sigmoid(logit)-target；加权后分别为 -1、1.5，屏蔽位置为 0。
    torch.testing.assert_close(gradient, torch.tensor([[[-1.0, 1.5], [0.0, 0.0]]]))


def test_old_output_rows_preserved_when_adding_class():
    def model(classes):
        head = SimpleNamespace(
            nc=classes, cv3=[torch.nn.Sequential(torch.nn.Conv2d(4, classes, 1)) for _ in range(3)]
        )
        return SimpleNamespace(model=[head], names={i: f"class_{i}" for i in range(classes)})

    old, new = model(2), model(3)
    new_rows = [layer[-1].weight[2].clone() for layer in new.model[-1].cv3]
    assert preserve_classification_outputs(old, new, 2, ["class_0", "class_1"]) == 3
    for index, (a, b) in enumerate(zip(old.model[-1].cv3, new.model[-1].cv3, strict=True)):
        torch.testing.assert_close(a[-1].weight, b[-1].weight[:2])
        torch.testing.assert_close(a[-1].bias, b[-1].bias[:2])
        torch.testing.assert_close(new_rows[index], b[-1].weight[2])


def test_validation_uses_deployment_single_class_selection():
    from car_smart_assist.perception.joint_yolo_backend import JointValidator

    validator = JointValidator.__new__(JointValidator)
    validator.args = SimpleNamespace(
        conf=0.001, iou=0.5, single_cls=False, agnostic_nms=False, max_det=10,
        task="detect",
    )
    validator.names = {0: "car", 1: "truck", 2: "bus"}
    validator.end2end = False
    predictions = torch.tensor([[[5.0], [5.0], [4.0], [4.0], [0.8], [0.7], [0.1]]])
    result = validator.postprocess(predictions)
    assert result[0]["cls"].tolist() == [0.0]


def test_training_ap_validation_passes_shared_sweep_contract(monkeypatch):
    from car_smart_assist.perception import joint_yolo_backend as backend
    from car_smart_assist.perception.joint_yolo_backend import DeploymentValidator

    captured = {}

    def fake_postprocess(predictions, **kwargs):
        captured.update(kwargs)
        return [torch.empty((0, 6))]

    monkeypatch.setattr(backend, "postprocess_detection", fake_postprocess)
    validator = DeploymentValidator.__new__(DeploymentValidator)
    validator.args = SimpleNamespace(
        conf=0.001, iou=0.5, single_cls=False, agnostic_nms=False, max_det=100,
        task="detect",
    )
    validator.names = {0: "car", 1: "truck", 2: "bus"}
    validator.end2end = False
    validator.postprocess(torch.empty((1, 7, 1)))
    assert captured == {
        "nc": 3,
        "names": validator.names,
        "mode": "ap_sweep",
        "confidence": 0.001,
        "class_thresholds": None,
        "iou": 0.5,
        "agnostic": False,
        "max_det": 100,
        "end2end": False,
        "task": "detect",
    }


def test_validation_ignores_unannotated_classes_and_void_regions(tmp_path):
    from PIL import Image

    from car_smart_assist.perception.joint_yolo_backend import JointValidator

    mask = np.full((10, 20), 255, np.uint8)
    mask[4:] = 1
    path = tmp_path / "roi.png"
    Image.fromarray(mask).save(path)
    validator = JointValidator.__new__(JointValidator)
    validator.args = SimpleNamespace(single_cls=False)
    validator.manifest = {"valid_roi_ids": [1, 2]}
    batch = {
        "imgsz": (10, 20),
        "ori_shape": (10, 20),
        "ratio_pad": ((1.0, 1.0), (0.0, 0.0)),
        "joint_row": {"supervised_classes": [8], "roi_mask": str(path), "boxes": []},
    }
    predictions = {
        "bboxes": torch.tensor([[4.0, 5.0, 6.0, 7.0], [5.0, 6.0, 8.0, 8.0], [3.0, 1.0, 4.0, 2.0]]),
        "cls": torch.tensor([0.0, 8.0, 8.0]),
        "conf": torch.tensor([0.8, 0.9, 0.7]),
        "extra": torch.empty(3, 0),
    }
    result = validator._prepare_pred(predictions, batch)
    assert result["cls"].tolist() == [8.0]
    assert result["conf"].tolist() == pytest.approx([0.9])


def test_locked_operating_stats_are_exact_and_macro_excludes_missing_gt(monkeypatch):
    from car_smart_assist.perception import joint_yolo_backend as backend

    monkeypatch.setattr(backend.DetectionValidator, "get_stats", lambda self: "base-stats")
    validator = backend.DeploymentValidator.__new__(backend.DeploymentValidator)
    validator.names = {0: "car", 1: "truck"}
    validator.postprocess_mode = "locked"
    validator.metrics = SimpleNamespace(
        stats={
            "target_cls": [np.array([0, 0])],
            "pred_cls": [np.array([0, 0, 1])],
            "tp": [np.array([[True], [False], [False]])],
        }
    )

    result = validator.get_stats()

    assert result == "base-stats"
    operating = validator.metrics.operating_iou50
    assert operating["per_class"]["car"] == {
        "class_id": 0, "tp": 1, "fp": 1, "fn": 1, "support": 2,
        "precision": 0.5, "recall": 0.5, "f1": 0.5, "has_ground_truth": True,
    }
    assert operating["per_class"]["truck"]["tp"] == 0
    assert operating["per_class"]["truck"]["fp"] == 1
    assert operating["per_class"]["truck"]["fn"] == 0
    assert operating["per_class"]["truck"]["has_ground_truth"] is False
    assert operating["macro"] == {"precision": 0.5, "recall": 0.5, "f1": 0.5}
