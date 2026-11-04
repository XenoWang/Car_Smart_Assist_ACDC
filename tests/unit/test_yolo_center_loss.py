"""Independent bounded-distance values/gradients, units, configuration and C."""
import math

import pytest
import torch

from car_smart_assist.config.detection import center_loss_config
from car_smart_assist.perception.yolo_center_loss import (
    CenterAssistedBboxLoss, center_distance_loss, require_center_recipe,
    train_center_recipe, validate_center_recipe,
)


def recipe(weight=.1, scale=16.):
    return {"version": 1, "enabled": True, "weight": weight, "normalization": "train_mean_sqrt_area",
            "scale_pixels": scale, "ground_truth_count": 2, "imgsz": 64,
            "training_data_signature": "synthetic-training"}


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_euclidean_value_and_analytic_gradient(dtype):
    # D=sqrt(3^2+4^2)=5, C=5, target-score weight2/denominator4.
    predicted = torch.tensor([[2., 3., 4., 5.]], dtype=dtype, requires_grad=True)
    target = torch.tensor([[-1., -1., 1., 1.]], dtype=dtype)
    value = center_distance_loss(predicted, target, torch.ones(1, 1), torch.tensor([2.]), 4., 5.)
    assert value.dtype == torch.float32
    assert value.item() == pytest.approx(.31606027941427883, rel=2e-6, abs=2e-7)
    value.backward()
    # Each x endpoint moves center by1/2; dD/dx=3/5, dD/dy=4/5.
    expected = torch.tensor([[.01103638323514327, .014715177646857695,
                              .01103638323514327, .014715177646857695]])
    # Half/bfloat gradients are stored in their input dtype.
    tolerance = {torch.float32: 2e-7, torch.float16: 8e-6, torch.bfloat16: 8e-5}[dtype]
    torch.testing.assert_close(predicted.grad.float(), expected, rtol=0, atol=tolerance)


def test_equal_centers_have_exact_zero_and_finite_zero_gradient():
    predicted = torch.tensor([[-2., -2., 2., 2.]], requires_grad=True)
    target = torch.tensor([[-1., -1., 1., 1.]])
    loss = center_distance_loss(predicted, target, torch.tensor([[32.]]), torch.ones(1), 1., 4.)
    assert loss.item() == 0.
    loss.backward()
    assert torch.equal(predicted.grad, torch.zeros_like(predicted))


def test_fpn_grid_strides_represent_the_same_pixel_offset():
    strides = torch.tensor([[8.], [16.], [32.]])
    offset = 16. / strides
    predicted = torch.cat((offset, torch.zeros(3, 1), offset + 1., torch.ones(3, 1)), dim=1)
    targets = torch.tensor([[0., 0., 1., 1.]]).repeat(3, 1)
    loss = center_distance_loss(predicted, targets, strides, torch.ones(3), 3., 16.)
    assert loss.item() == pytest.approx(.6321205588285577, rel=2e-6)


def test_target_score_weights_and_unassigned_zero_rows_do_not_become_plain_mean():
    predicted = torch.tensor([[16., 0., 18., 2.], [0., 0., 2., 2.]])
    target = torch.tensor([[0., 0., 2., 2.]]).repeat(2, 1)
    loss = center_distance_loss(predicted, target, torch.ones(2, 1), torch.tensor([1., 3.]), 4., 16.)
    assert loss.item() == pytest.approx(.15803013970713942, rel=2e-6)


def test_small_distance_preserves_the_exponential_second_order_correction():
    step = 2. ** -20
    predicted = torch.tensor([[step, 0., step, 0.]])
    value = center_distance_loss(predicted, torch.zeros(1, 4), torch.ones(1, 1), torch.ones(1), 1., 1.)
    # For x>0, x-x^2/2 < 1-exp(-x) < x. Direct 1-exp rounds to x here.
    assert step - step * step / 2 <= value.item() < step


def test_empty_assigned_set_is_zero_and_remains_connected_to_autograd():
    predicted = torch.empty(0, 4, requires_grad=True)
    value = center_distance_loss(predicted, torch.empty(0, 4), torch.empty(0, 1), torch.empty(0), 1., 16.)
    assert value.item() == 0.
    value.backward()
    assert predicted.grad.shape == (0, 4)


@pytest.mark.parametrize("options", [
    None, [], True, {"enabled": "false"}, {"weight": True}, {"weight": -1},
    {"weight": float("nan")}, {"weight": float("inf")}, {"weight": 10**400},
    {"normalization": "batch"}, {"enabled": True, "typo": 1},
])
def test_invalid_project_options_have_explicit_rejection(options):
    with pytest.raises(ValueError, match="center_loss"):
        center_loss_config(options)


def test_config_defaults_zero_weight_and_recipe_identity():
    assert center_loss_config({}) == {"enabled": False}
    assert center_loss_config({"enabled": True, "weight": 0})["weight"] == 0.
    assert require_center_recipe({}, {}) == {"enabled": False}
    with pytest.raises(ValueError, match="configuration changed or missing"):
        require_center_recipe({}, {"enabled": True})
    with pytest.raises(ValueError, match="configuration changed"):
        require_center_recipe({"center_loss": recipe()}, {"enabled": False})
    with pytest.raises(ValueError, match="normalization changed"):
        require_center_recipe({"center_loss": recipe()}, {"enabled": True}, resolved=recipe(scale=8.))


@pytest.mark.parametrize("field,value", [
    ("version", True), ("version", 2), ("scale_pixels", 0), ("scale_pixels", float("nan")),
    ("scale_pixels", 10**400), ("imgsz", 0), ("ground_truth_count", 0),
    ("training_data_signature", ""), ("enabled", "true"),
])
def test_incomplete_or_invalid_resume_normalization_is_rejected(field, value):
    broken = {**recipe(), field: value}
    with pytest.raises(ValueError, match="Center loss"):
        validate_center_recipe(broken)


def test_train_mean_uses_box_area_and_exact_loader_resize_units():
    labels = [
        {"shape": (100, 200), "bboxes": [[.5, .5, .1, .2]]},  #20x20 pixels
        {"shape": (100, 100), "bboxes": [[.5, .5, .2, .2]]},  #40x40 pixels
    ]
    result = train_center_recipe(labels, 200, "train-only", {"enabled": True})
    assert result["scale_pixels"] == 30.
    assert result["ground_truth_count"] == 2
    assert result["training_data_signature"] == "train-only"
    # Installed loader uses ceil(3*3.2)=10, not continuous height9.6.
    result = train_center_recipe([{"shape": (3, 10), "bboxes": [[.5, .5, .5, .4]]}],
                                 32, "ceil-case", {"enabled": True})
    assert result["scale_pixels"] == 8.


@pytest.mark.parametrize("labels", [
    [], [{"shape": (10, 10), "bboxes": []}],
    [{"shape": (0, 10), "bboxes": [[.5, .5, .5, .5]]}],
    [{"shape": (10, 10), "bboxes": [[.5, .5, 0, .5]]}],
    [{"shape": (10, 10), "bboxes": [[.5, .5, float("nan"), .5]]}],
    [{"shape": (10, 10), "bbox_format": "xyxy", "bboxes": [[.1, .1, .2, .2]]}],
])
def test_training_normalization_refuses_missing_or_invalid_labels(labels):
    with pytest.raises(ValueError, match="Center loss"):
        train_center_recipe(labels, 32, "train", {"enabled": True})


def test_disabled_normalization_never_reads_training_labels():
    class Unreadable:
        def __iter__(self):
            pytest.fail("disabled option scanned labels")
    assert train_center_recipe(Unreadable(), None, None, {}) == {"enabled": False}


def test_wrapper_preserves_vendor_dfl_and_adds_only_the_known_box_term():
    from ultralytics.utils.loss import BboxLoss
    base = BboxLoss(reg_max=1)
    wrapper = CenterAssistedBboxLoss(base, recipe(weight=.25, scale=16.))
    wrapper.anchor_strides = torch.tensor([[8.]])
    predicted = torch.tensor([[[1., -1., 3., 1.]]], requires_grad=True)  #center2grid ->16pixels
    target = torch.tensor([[[-1., -1., 1., 1.]]])
    args = (torch.zeros(1, 1, 4), predicted, torch.zeros(1, 2), target,
            torch.ones(1, 1, 1), 1., torch.tensor([[True]]))
    reference_box, reference_dfl = base(*args)
    box, dfl = wrapper(*args)
    assert (box-reference_box).item() == pytest.approx(.15803013970713942, rel=3e-6)
    assert torch.equal(dfl, reference_dfl)
