"""Partial labels must not teach missing classes/regions as background."""

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
    # ACDC supervises old classes; Lost & Found only obstacle class in known ROI.
    loss.valid = torch.tensor(
        [
            [[True, True, False]] * 3,
            [[False, False, True], [False, False, False], [False, False, False]],
        ]
    )
    loss.weights = torch.ones(2, 1, 1)
    targets[1, 2, 2] = 1  # Positive assignment outside sparse ROI must still train.
    loss(predictions, targets).sum().backward()
    gradient = predictions.grad
    assert torch.count_nonzero(gradient[0, :, 2]) == 0
    assert torch.count_nonzero(gradient[1, :2, :2]) == 0
    assert torch.all(gradient[1, 2, :2] > 0)  # Labelled cargo is not a traffic-category positive.
    assert gradient[1, 1, 2] == 0
    assert gradient[1, 0, 2] > 0 and gradient[1, 2, 2] < 0


def test_old_output_rows_preserved_when_adding_class():
    def model(classes):
        head = SimpleNamespace(
            nc=classes, cv3=[torch.nn.Sequential(torch.nn.Conv2d(4, classes, 1)) for _ in range(3)]
        )
        return SimpleNamespace(model=[head])

    old, new = model(2), model(3)
    new_rows = [layer[-1].weight[2].clone() for layer in new.model[-1].cv3]
    assert preserve_classification_outputs(old, new, 2) == 3
    for index, (a, b) in enumerate(zip(old.model[-1].cv3, new.model[-1].cv3, strict=True)):
        torch.testing.assert_close(a[-1].weight, b[-1].weight[:2])
        torch.testing.assert_close(a[-1].bias, b[-1].bias[:2])
        torch.testing.assert_close(new_rows[index], b[-1].weight[2])


def test_validation_uses_deployment_single_class_selection():
    from car_smart_assist.perception.joint_yolo_backend import DeploymentValidator

    validator = DeploymentValidator.__new__(DeploymentValidator)
    validator.args = SimpleNamespace(
        conf=0.25, iou=0.5, single_cls=False, agnostic_nms=False, max_det=10
    )
    validator.end2end = False
    predictions = torch.tensor([[[5.0], [5.0], [4.0], [4.0], [0.8], [0.7], [0.1]]])
    result = validator.postprocess(predictions)
    assert result[0]["cls"].tolist() == [0.0]


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
