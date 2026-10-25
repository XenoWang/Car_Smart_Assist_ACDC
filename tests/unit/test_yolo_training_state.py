"""中断检查点的精度和身份检查。"""

import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from car_smart_assist.perception.yolo_training import (
    attach_training_state_callbacks,
    load_training_state,
    require_model_selection_contract,
)


TRAINING_SEGMENT = {"start_epoch": 0, "target_epochs": 3, "resume_extra_epochs": None, "epochs_override": None, "early_stopping_policy": "preserve_history"}


SELECTION_CONTRACT = {
    "schema_version": 1,
    "postprocess_version": "best_class_threshold_v1",
    "ap_mode": "ap_sweep",
    "sweep_confidence": 0.001,
    "imgsz": 960,
    "nms_iou": 0.5,
    "max_det": 100,
    "max_nms": 30000,
    "max_wh": 7680,
    "agnostic_nms": False,
    "class_selection": "original_best_class_only",
    "threshold_comparison": ">=",
    "task": "detect",
    "fitness": "0.1*mAP50+0.9*mAP50_95",
    "joint_selection": None,
}


@pytest.mark.parametrize(
    "replacement",
    [
        {"postprocess_version": "other"},
        {"ap_mode": "locked"},
        {"sweep_confidence": 0.01},
        {"imgsz": 641},
        {"nms_iou": 0.7},
        {"max_det": 300},
        {"class_selection": "all_classes"},
        {
            "joint_selection": {
                "known_class_count": 8,
                "source_weights": {"acdc": 0.75, "laf": 0.25},
                "valid_roi_ids": [1],
                "foreground_exclusive": True,
            }
        },
    ],
)
def test_every_model_selection_identity_change_is_rejected(replacement):
    changed = {**SELECTION_CONTRACT, **replacement}
    with pytest.raises(ValueError, match="contract"):
        require_model_selection_contract(
            {"model_selection_contract": changed}, SELECTION_CONTRACT
        )


def trainer(tmp_path):
    module = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.AdamW(module.parameters(), lr=0.001)
    module.weight.grad = torch.full_like(module.weight, 1e-8)
    optimizer.step()
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    optimizer.param_groups[0]["lr"] = np.float64(0.001)
    last = tmp_path / "last.pt"
    last.write_bytes(b"paired checkpoint")
    best = tmp_path / "best.pt"
    best.write_bytes(b"paired best checkpoint")
    return SimpleNamespace(
        model=module,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=torch.amp.GradScaler("cpu", enabled=False),
        ema=SimpleNamespace(ema=copy.deepcopy(module), updates=1),
        epoch=0,
        epochs=TRAINING_SEGMENT["target_epochs"],
        stop=False,
        best_fitness=0.2,
        fitness=0.2,
        best=best,
        stopper=SimpleNamespace(best_fitness=0.2, best_epoch=0, patience=3, possible_stop=False),
        start_epoch=1,
        last=last,
        wdir=tmp_path,
        device=torch.device("cpu"),
    )


def test_adam_small_variance_and_model_restored_without_half_rounding(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    callbacks = {}
    model = SimpleNamespace(add_callback=lambda key, callback: callbacks.update({key: callback}))
    original = trainer(tmp_path)
    attach_training_state_callbacks(model, "dataset", None, selection_contract=SELECTION_CONTRACT, training_segment=TRAINING_SEGMENT)
    callbacks["on_model_save"](original)
    state = load_training_state(
        tmp_path / "last_training_state.pt", original.last, "dataset", SELECTION_CONTRACT
    )
    variance = next(iter(state["optimizer"]["state"].values()))["exp_avg_sq"]
    assert variance.dtype == torch.float32
    assert variance.item() > 0
    assert (
        variance.half().item() == 0
    )  # 这些状态会被依赖库的 FP16 转换丢掉。
    assert type(state["optimizer"]["param_groups"][0]["lr"]) is float
    restored = trainer(tmp_path)
    with torch.no_grad():
        restored.model.weight.fill_(100)
    attach_training_state_callbacks(model, "dataset", state, selection_contract=SELECTION_CONTRACT, training_segment=TRAINING_SEGMENT)
    callbacks["on_train_start"](restored)
    torch.testing.assert_close(restored.model.weight, original.model.weight)
    restored_variance = next(iter(restored.optimizer.state.values()))["exp_avg_sq"]
    torch.testing.assert_close(restored_variance, variance)
    restored.start_epoch = 2
    with pytest.raises(ValueError, match="epoch"):
        callbacks["on_train_start"](restored)


def test_missing_or_mismatched_training_state_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    saved = trainer(tmp_path)
    state_path = tmp_path / "last_training_state.pt"
    with pytest.raises(ValueError, match="missing"):
        load_training_state(state_path, saved.last, "dataset", SELECTION_CONTRACT)
    callbacks = {}
    model = SimpleNamespace(add_callback=lambda key, callback: callbacks.update({key: callback}))
    attach_training_state_callbacks(model, "dataset", None, selection_contract=SELECTION_CONTRACT, training_segment=TRAINING_SEGMENT)
    callbacks["on_model_save"](saved)
    with pytest.raises(ValueError, match="match"):
        load_training_state(state_path, saved.last, "different dataset", SELECTION_CONTRACT)
    saved.last.write_bytes(b"different checkpoint")
    with pytest.raises(ValueError, match="match"):
        load_training_state(state_path, saved.last, "dataset", SELECTION_CONTRACT)


def test_missing_or_changed_selection_contract_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    saved = trainer(tmp_path)
    callbacks = {}
    model = SimpleNamespace(add_callback=lambda key, callback: callbacks.update({key: callback}))
    attach_training_state_callbacks(model, "dataset", None, selection_contract=SELECTION_CONTRACT, training_segment=TRAINING_SEGMENT)
    callbacks["on_model_save"](saved)
    state_path = tmp_path / "last_training_state.pt"
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    state.pop("model_selection_contract")
    torch.save(state, state_path)
    with pytest.raises(ValueError, match="contract"):
        load_training_state(state_path, saved.last, "dataset", SELECTION_CONTRACT)
    state["model_selection_contract"] = {**SELECTION_CONTRACT, "imgsz": 641}
    torch.save(state, state_path)
    with pytest.raises(ValueError, match="contract"):
        load_training_state(state_path, saved.last, "dataset", SELECTION_CONTRACT)


def test_completed_checkpoint_rebound_to_final_file(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    saved = trainer(tmp_path)
    callbacks = {}
    model = SimpleNamespace(add_callback=lambda key, callback: callbacks.update({key: callback}))
    attach_training_state_callbacks(model, "dataset", None, selection_contract=SELECTION_CONTRACT, training_segment=TRAINING_SEGMENT)
    callbacks["on_model_save"](saved)
    saved.last.write_bytes(b"final stripped checkpoint")
    callbacks["on_train_end"](saved)
    state = load_training_state(
        tmp_path / "last_training_state.pt", saved.last, "dataset", SELECTION_CONTRACT
    )
    assert state["format_version"] == 4
    assert state["model_selection_contract"] == SELECTION_CONTRACT
    assert state["epoch"] == 0
    assert state["best_fitness"] == 0.2
    assert state["best_checkpoint_sha256"] == __import__("hashlib").sha256(saved.best.read_bytes()).hexdigest()
    assert state["training_segment"] == TRAINING_SEGMENT
def test_resume_restores_saved_early_stopper_history(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    original = trainer(tmp_path)
    callbacks = {}
    model = SimpleNamespace(add_callback=lambda key, callback: callbacks.update({key: callback}))
    attach_training_state_callbacks(model, "dataset", None,
                                    selection_contract=SELECTION_CONTRACT,
                                    training_segment=TRAINING_SEGMENT)
    callbacks["on_model_save"](original)
    state = load_training_state(tmp_path / "last_training_state.pt", original.last,
                                "dataset", SELECTION_CONTRACT)
    state["early_stopper"] = {
        "best_fitness": 0.17, "best_epoch": 4, "patience": 9, "possible_stop": True,
    }
    restored = trainer(tmp_path)
    restored.stopper = SimpleNamespace(best_fitness=None, best_epoch=0,
                                       patience=3, possible_stop=False)
    callbacks = {}
    model = SimpleNamespace(add_callback=lambda key, callback: callbacks.update({key: callback}))
    attach_training_state_callbacks(model, "dataset", state,
                                    selection_contract=SELECTION_CONTRACT,
                                    training_segment=TRAINING_SEGMENT)
    callbacks["on_train_start"](restored)
    assert vars(restored.stopper) == state["early_stopper"]


@pytest.mark.parametrize("best_mode", ["missing", "replaced"])
def test_resume_refuses_missing_or_replaced_paired_best(tmp_path, monkeypatch, best_mode):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    saved = trainer(tmp_path)
    callbacks = {}
    model = SimpleNamespace(add_callback=lambda key, callback: callbacks.update({key: callback}))
    attach_training_state_callbacks(model, "dataset", None,
                                    selection_contract=SELECTION_CONTRACT,
                                    training_segment=TRAINING_SEGMENT)
    callbacks["on_model_save"](saved)
    if best_mode == "missing":
        saved.best.unlink()
    else:
        saved.best.write_bytes(b"unpaired same-structure fixture")
    with pytest.raises(ValueError, match="best"):
        load_training_state(tmp_path / "last_training_state.pt", saved.last,
                            "dataset", SELECTION_CONTRACT)
