"""Real detector assignment, default equivalence, train-only wiring and restore guards."""
from copy import deepcopy
from contextlib import nullcontext
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from _model_test_helpers import ROOT, load_module
from car_smart_assist.config.detection import center_loss_config, load_yolo_config
from car_smart_assist.perception.joint_yolo import JointDetectionLoss
from car_smart_assist.perception.yolo_center_loss import train_center_recipe
from test_yolo_center_loss import recipe


@pytest.fixture(autouse=True)
def preserve_state():
    import random
    state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        yield
    finally:
        random.setstate(state[0]); np.random.set_state(state[1]); torch.set_rng_state(state[2])
        torch.set_num_threads(threads)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


def tiny_architecture():
    return {"nc": 9, "backbone": [[-1, 1, "Conv", [8, 3, 2]],
                                 [-1, 1, "Conv", [16, 3, 2]],
                                 [-1, 1, "Conv", [32, 3, 2]]],
            "head": [[[0, 1, 2], 1, "Detect", [9]]]}


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_real_assignment_preserves_masked_classification_dfl_and_zero_weight_gradients(device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA hardware is unavailable")
    from ultralytics.nn.tasks import DetectionModel
    model = DetectionModel(tiny_architecture(), nc=9, verbose=False).to(device)
    model.args = SimpleNamespace(box=7.5, cls=.5, dfl=1.5)
    pixels = torch.linspace(0, 1, 2*3*32*32, device=device).reshape(2, 3, 32, 32)
    with torch.autocast('cuda', dtype=torch.float16) if device.startswith('cuda') else nullcontext():
        predictions = model(pixels)
        batch = {
            "batch_idx": torch.tensor([0, 1], device=device),
            "cls": torch.tensor([[0.], [8.]], device=device),
            "bboxes": torch.tensor([[.5, .5, .25, .25], [.5, .75, .25, .25]], device=device),
            "class_supervision": torch.tensor([[True]*8+[False], [False]*8+[True]], device=device),
            "obstacle_roi": torch.ones(2, 32, 32, dtype=torch.bool, device=device),
            "source_loss_weight": torch.tensor([1., .5], device=device),
        }
        batch["obstacle_roi"][1, :16] = False
        baseline = JointDetectionLoss(model, 8)
        zero = JointDetectionLoss(model, 8, center_loss=recipe(weight=0))
        candidate = JointDetectionLoss(model, 8, center_loss=recipe(scale=8.))
        base_loss, base_items = baseline(predictions, batch)
        zero_loss, zero_items = zero(predictions, batch)
        active_loss, active_items = candidate(predictions, batch)
        assert torch.equal(base_loss, zero_loss) and torch.equal(base_items, zero_items)
        assert torch.equal(active_items[1:], base_items[1:])  #classification and DFL
        assert active_items[0] > base_items[0]
        base_grad = torch.autograd.grad(base_loss.sum(), predictions, retain_graph=True)
        zero_grad = torch.autograd.grad(zero_loss.sum(), predictions, retain_graph=True)
        active_grad = torch.autograd.grad(active_loss.sum(), predictions)
        assert all(torch.equal(a, b) for a, b in zip(base_grad, zero_grad))
        assert all(torch.isfinite(g).all() for g in active_grad)
        assert any(not torch.equal(a, b) for a, b in zip(base_grad, active_grad))
        assert candidate.base.bce.valid is None and candidate.base.bce.weights is None
        assert candidate.center_bbox.anchor_strides is None


def test_trial_and_baseline_share_initialization_and_vendor_settings_but_isolate_outputs():
    trial = load_yolo_config(ROOT/"configs/model/yolo_center_loss_trial.yaml")
    baseline = load_yolo_config(ROOT/"configs/model/yolo_center_loss_baseline.yaml")
    deployed = load_yolo_config(ROOT/"configs/model/yolo_detection.yaml")
    assert trial["model"] == baseline["model"] == deployed["model"]
    assert trial["train"] == baseline["train"] == deployed["train"]
    assert trial["inference"] == baseline["inference"] == deployed["inference"]
    assert len({trial["name"], baseline["name"], deployed["name"]}) == 3
    assert len({trial["checkpoint"], baseline["checkpoint"], deployed["checkpoint"]}) == 3
    assert len({trial["calibration_file"], baseline["calibration_file"], deployed["calibration_file"]}) == 3
    assert trial["center_loss"]["enabled"] is True
    assert baseline["center_loss"] == {"enabled": False}
    assert "center_loss" not in trial["train"]
    assert "center_loss" not in deployed
    assert trial["enabled_for_inference"] is False and baseline["enabled_for_inference"] is False


def test_joint_builder_computes_normalization_only_from_its_actual_train_dataset(monkeypatch):
    from car_smart_assist.perception import joint_yolo_backend as backend
    from test_yolo_training_state import SELECTION_CONTRACT
    options = {"enabled": True}
    factory = backend.joint_trainer(None, selection_contract=SELECTION_CONTRACT, center_loss=options)
    trainer = factory.__new__(factory)
    from ultralytics.cfg import DEFAULT_CFG
    trainer.args = deepcopy(DEFAULT_CFG)
    for name in ("mosaic","mixup","copy_paste","cutmix","degrees","translate","scale",
                 "shear","perspective","fliplr","flipud","multi_scale"):
        setattr(trainer.args, name, 0)
    trainer.args.imgsz, trainer.args.batch = 64, 2
    trainer.data = {}
    trainer.model = SimpleNamespace(joint_obstacle_class=8, joint_foreground_exclusive=True)
    class Dataset:
        imgsz = 64
        manifest = {"signature": "actual-train"}
        labels = [{"shape": (64, 64), "bboxes": [[.5,.5,.5,.5]]}]
    monkeypatch.setattr(backend, "JointDataset", lambda **kwargs: Dataset())
    captured = []
    monkeypatch.setattr(backend, "JointDetectionLoss", lambda *args, **kwargs: captured.append(kwargs["center_loss"]) or object())
    trainer.build_dataset("train", mode="train")
    assert trainer._project_center_loss["scale_pixels"] == 32.
    assert trainer._project_center_loss["ground_truth_count"] == 1
    assert captured[0]["training_data_signature"] == "actual-train"
    trainer.validation_args = lambda: trainer.args
    class ForbiddenLabels:
        imgsz = 64
        manifest = {"signature": "validation-must-not-change-C"}
        @property
        def labels(self):
            pytest.fail("normalization read validation labels")
    monkeypatch.setattr(backend, "JointDataset", lambda **kwargs: ForbiddenLabels())
    trainer.build_dataset("val", mode="val")
    assert len(captured) == 1 and trainer._project_center_loss["scale_pixels"] == 32.


def test_callbacks_save_recipe_and_reject_changed_runtime_scale_before_restoring_weights(tmp_path):
    from test_yolo_training_state import trainer, SELECTION_CONTRACT, TRAINING_SEGMENT
    from car_smart_assist.perception.yolo_training import attach_training_state_callbacks
    target = trainer(tmp_path)
    target._project_center_loss = recipe()
    callbacks = {}
    model = SimpleNamespace(add_callback=lambda event, callback: callbacks.update({event: callback}))
    attach_training_state_callbacks(model, "synthetic", None, selection_contract=SELECTION_CONTRACT,
                                    training_segment=TRAINING_SEGMENT, center_loss={"enabled": True})
    callbacks["on_model_save"](target)
    state = torch.load(tmp_path/"last_training_state.pt", map_location="cpu", weights_only=True)
    assert state["center_loss"] == recipe()
    attach_training_state_callbacks(model, "synthetic", state, selection_contract=SELECTION_CONTRACT,
                                    training_segment=TRAINING_SEGMENT, center_loss={"enabled": True})
    before = target.model.weight.detach().clone()
    target._project_center_loss = recipe(scale=8.)
    with pytest.raises(ValueError, match="normalization changed"):
        callbacks["on_train_start"](target)
    assert torch.equal(before, target.model.weight)
    with pytest.raises(ValueError, match="configuration changed"):
        attach_training_state_callbacks(model, "synthetic", state, selection_contract=SELECTION_CONTRACT,
                                        training_segment=TRAINING_SEGMENT)


def test_enabled_loss_is_rejected_on_non_joint_training_plan(tmp_path):
    module = load_module("scripts/train_detection.py", "center_non_joint_cli")
    cfg = {"train": {}, "center_loss": {"enabled": True}}
    with pytest.raises(ValueError, match="requires the joint"):
        module.resolve_training_plan(tmp_path, cfg, {}, tmp_path/"run", fresh=True)
