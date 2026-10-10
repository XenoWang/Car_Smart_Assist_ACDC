"""中断检查点的精度和身份检查。"""

import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from car_smart_assist.perception.yolo_training import (
    attach_training_state_callbacks,
    load_training_state,
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
    return SimpleNamespace(
        model=module,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=torch.amp.GradScaler("cpu", enabled=False),
        ema=SimpleNamespace(ema=copy.deepcopy(module), updates=1),
        epoch=0,
        best_fitness=0.2,
        start_epoch=1,
        last=last,
        wdir=tmp_path,
    )


def test_adam_small_variance_and_model_restored_without_half_rounding(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    callbacks = {}
    model = SimpleNamespace(add_callback=lambda key, callback: callbacks.update({key: callback}))
    original = trainer(tmp_path)
    attach_training_state_callbacks(model, "dataset", None)
    callbacks["on_model_save"](original)
    state = load_training_state(tmp_path / "last_training_state.pt", original.last, "dataset")
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
    attach_training_state_callbacks(model, "dataset", state)
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
        load_training_state(state_path, saved.last, "dataset")
    callbacks = {}
    model = SimpleNamespace(add_callback=lambda key, callback: callbacks.update({key: callback}))
    attach_training_state_callbacks(model, "dataset", None)
    callbacks["on_model_save"](saved)
    with pytest.raises(ValueError, match="match"):
        load_training_state(state_path, saved.last, "different dataset")
    saved.last.write_bytes(b"different checkpoint")
    with pytest.raises(ValueError, match="match"):
        load_training_state(state_path, saved.last, "dataset")


def test_completed_checkpoint_rebound_to_final_file(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    saved = trainer(tmp_path)
    callbacks = {}
    model = SimpleNamespace(add_callback=lambda key, callback: callbacks.update({key: callback}))
    attach_training_state_callbacks(model, "dataset", None)
    callbacks["on_model_save"](saved)
    saved.last.write_bytes(b"final stripped checkpoint")
    callbacks["on_train_end"](saved)
    state = load_training_state(tmp_path / "last_training_state.pt", saved.last, "dataset")
    assert state["format_version"] == 2
    assert state["epoch"] == 0
    assert state["best_fitness"] == 0.2
