"""Legacy joint-loss ABI and strict optional-recipe boundary regressions."""
from copy import deepcopy
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from car_smart_assist.config.detection import center_loss_config
from car_smart_assist.perception.joint_yolo import JointDetectionLoss, MaskedClassificationLoss
from test_yolo_center_loss_training import tiny_architecture


@pytest.fixture(autouse=True)
def preserve_random_state():
    state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    threads = torch.get_num_threads()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    torch.set_num_threads(2)
    try:
        yield
    finally:
        random.setstate(state[0]); np.random.set_state(state[1]); torch.set_rng_state(state[2])
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)
        torch.set_num_threads(threads)


def model_and_batch():
    from ultralytics.nn.tasks import DetectionModel
    model = DetectionModel(tiny_architecture(), nc=9, verbose=False)
    model.args = SimpleNamespace(box=7.5, cls=.5, dfl=1.5)
    pixels = torch.linspace(0, 1, 2*3*32*32).reshape(2, 3, 32, 32)
    predictions = model(pixels)
    batch = {"batch_idx":torch.tensor([0,1]), "cls":torch.tensor([[0.],[7.]]),
             "bboxes":torch.tensor([[.5,.5,.25,.25],[.5,.5,.25,.25]]),
             "class_supervision":torch.tensor([[True]*8+[False]]*2),
             "obstacle_roi":torch.ones(2,32,32,dtype=torch.bool),
             "source_loss_weight":torch.tensor([1.,.5])}
    return model, predictions, batch


def test_old_serialized_field_set_keeps_default_loss_values_and_gradients(tmp_path):
    from ultralytics.utils.loss import v8DetectionLoss
    model, predictions, batch = model_and_batch()
    # main@6d2c468 has exactly these legacy construction fields, with no
    # center_bbox. Build that ABI before serializing under the current class name;
    # pickle restoration must not depend on the new __init__ having run.
    legacy = JointDetectionLoss.__new__(JointDetectionLoss)
    legacy.base = v8DetectionLoss(model)
    legacy.base.bce = MaskedClassificationLoss()
    legacy.obstacle_class = 8
    path = tmp_path/"legacy_joint_loss.pt"
    torch.save(legacy,path)
    restored = torch.load(path,map_location="cpu",weights_only=False)
    assert not hasattr(restored,"center_bbox")
    fresh = JointDetectionLoss(model,8)
    expected, expected_items = fresh(predictions,batch)
    actual, actual_items = restored(predictions,batch)
    # Metamorphic ABI contract: identical old fields preserve value and complete
    # gradients. Center-term mathematics has separate closed-form test oracles.
    assert torch.equal(actual,expected) and torch.equal(actual_items,expected_items)
    expected_grad = torch.autograd.grad(expected.sum(),predictions,retain_graph=True)
    actual_grad = torch.autograd.grad(actual.sum(),predictions)
    assert all(torch.equal(a,b) for a,b in zip(actual_grad,expected_grad))
    assert restored.base.bce.valid is None and restored.base.bce.weights is None


@pytest.mark.parametrize("invalid",[False,0,"",[],{},()])
def test_falsey_invalid_resolved_recipe_is_rejected_before_vendor_initialization(invalid):
    # The model is deliberately unusable: a recipe error must be classified at
    # its own boundary before the vendor tries to inspect/allocate model state.
    with pytest.raises(ValueError,match="Center loss"):
        JointDetectionLoss(None,8,center_loss=invalid)


def test_none_and_explicit_disabled_normalized_config_remain_equivalent():
    model,predictions,batch = model_and_batch()
    omitted=JointDetectionLoss(model,8,center_loss=None)
    normalized=JointDetectionLoss(model,8,center_loss=center_loss_config({}))
    explicit=JointDetectionLoss(model,8,center_loss={"enabled":False})
    expected,items=omitted(predictions,batch)
    for criterion in (normalized,explicit):
        actual,actual_items=criterion(predictions,batch)
        assert torch.equal(actual,expected) and torch.equal(actual_items,items)

