"""Controlled two-device evidence; no claim of physical multi-GPU coverage."""
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from car_smart_assist.perception import weather_attention_training as training


@pytest.fixture
def two_devices(monkeypatch):
    current = {"index": 0}
    @contextmanager
    def device_scope(device):
        old = current["index"]
        current["index"] = torch.device(device).index
        try:
            yield
        finally:
            current["index"] = old
    monkeypatch.setattr(torch.cuda, "device", device_scope)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: current["index"])
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: current["index"] == 0)
    return current


def test_bf16_query_uses_selected_card_and_restores_current_card(two_devices):
    assert torch.cuda.current_device() == 0
    assert training._bf16_supported(torch.device("cuda:1")) is False
    assert torch.cuda.current_device() == 0
    assert training._bf16_supported(torch.device("cuda:0")) is True
    assert torch.cuda.current_device() == 0


def test_shared_preflight_helpers_use_selected_context_even_on_error(tmp_path, monkeypatch, two_devices):
    cfg = SimpleNamespace(device="cuda:1", report_dir="reports", report_file="reports/REPORT.md")
    monkeypatch.setattr(training, "select_device", lambda _: torch.device("cuda:1"))
    def inner(*args):
        assert torch.cuda.current_device() == 1
        raise ValueError("controlled preflight failure")
    monkeypatch.setattr(training, "_preflight_attention", inner)
    with pytest.raises(ValueError, match="controlled preflight failure"):
        training.preflight_attention(cfg, tmp_path, {}, {})
    assert torch.cuda.current_device() == 0


def test_preflight_transfer_backward_and_tracker_sync_the_selected_device(tmp_path, monkeypatch, two_devices):
    from test_training_stop_models import _attention_setup
    training, cfg, _, split, _ = _attention_setup(tmp_path, monkeypatch)
    cfg = replace(cfg, device="cuda:1", train={**cfg.train, "swad": {
        "enabled": True, "start_epoch": 1}})
    # The optimizer fixture stubs preflight; load its real code in isolation.
    from _model_test_helpers import load_module
    actual = load_module("src/car_smart_assist/perception/weather_attention_training.py",
                         "selected_device_preflight")
    monkeypatch.setattr(actual, "select_device", lambda _: torch.device("cuda:1"))
    monkeypatch.setattr(actual, "ensure_pretrained_weights", lambda *_: tmp_path / "pretrained.pth")
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Linear(1, 4)
        def forward(self, x, *_):
            return self.backbone(x.float().mean(dim=(1, 2, 3))[:, None])
        def set_training_epoch(self, _):
            pass
    class Dataset:
        images = np.zeros((2, 4, 4, 3), np.uint8)
        targets = np.zeros((2, 4), np.float32)
        mask = np.ones((2, 4), np.float32)
        samples = (object(), object())
        def __len__(self):
            return 2
        def __getitem__(self, index):
            return torch.from_numpy(self.images[index]).permute(2, 0, 1), torch.zeros(1), 0
    ds = Dataset()
    old_to = torch.Tensor.to
    def to_cpu(self, *args, **kwargs):
        if args and isinstance(args[0], torch.device) and args[0].type == "cuda":
            args = (torch.device("cpu"), *args[1:])
        if isinstance(kwargs.get("device"), torch.device) and kwargs["device"].type == "cuda":
            kwargs["device"] = torch.device("cpu")
        return old_to(self, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, "to", to_cpu)
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda self: self)
    monkeypatch.setattr(torch, "autocast", lambda *args, **kwargs: nullcontext())
    sync = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device=None: sync.append(
        torch.cuda.current_device() if device is None else torch.device(device).index))
    monkeypatch.setattr(actual, "WeatherAttention", lambda *_: Model())
    monkeypatch.setattr(actual, "batch_images", lambda d, ids, *_:
                        torch.from_numpy(d.images[ids]).permute(0, 3, 1, 2).float().div(255))
    monkeypatch.setattr(actual, "optimize_attention_inference", lambda model: model)
    contexts = []
    def benchmark(*args, **kwargs):
        contexts.append(torch.cuda.current_device())
        return {"cpu": 1., "cuda": 1.}
    monkeypatch.setattr(actual, "benchmark_model", benchmark)
    @dataclass
    class LegacyConfig:
        device: str = "cpu"
    monkeypatch.setattr(actual, "load_weather_config", lambda _: LegacyConfig())
    monkeypatch.setattr(actual, "load_weather_pyramid_config", lambda _: object())
    monkeypatch.setattr(actual.WeatherPredictor, "from_checkpoint", lambda *_: SimpleNamespace(model=Model()))
    monkeypatch.setattr(actual.WeatherPyramidPredictor, "from_checkpoint", lambda *_: SimpleNamespace(model=Model()))
    monkeypatch.setattr(actual, "WeatherDataset", lambda *_: ds)
    monkeypatch.setattr(actual, "masked_weather_loss", lambda logits, *_: (logits - 1).square().mean())
    monkeypatch.setattr(actual, "SWADTracker", lambda *_: SimpleNamespace(update=lambda _: None))
    monkeypatch.setattr(actual, "attention_model_config", lambda _: {"tiny": True})
    monkeypatch.setattr(actual, "model_report", lambda *_: None)
    result = actual.preflight_attention(cfg, tmp_path, {"train": ds}, split)
    assert result["finite_forward_backward"] is True
    assert result["device"] == "cuda:1"
    assert sync and set(sync) == {1}
    assert contexts and set(contexts) == {1}
    assert torch.cuda.current_device() == 0


def test_training_precision_is_chosen_from_selected_card(tmp_path, monkeypatch, two_devices):
    from test_training_stop_models import _attention_setup
    actual, cfg, datasets, split, _ = _attention_setup(tmp_path, monkeypatch)
    cfg = replace(cfg, device="cuda:1", train={**cfg.train, "amp": True})
    monkeypatch.setattr(actual, "select_device", lambda _: torch.device("cuda:1"))
    monkeypatch.setattr(torch.random, "fork_rng", lambda **_: nullcontext())
    old_to = torch.Tensor.to
    def to_cpu(self, *args, **kwargs):
        if args and isinstance(args[0], torch.device) and args[0].type == "cuda":
            args = (torch.device("cpu"), *args[1:])
        return old_to(self, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, "to", to_cpu)
    class ProbeComplete(Exception):
        pass
    observed = []
    def scaler(_device, *, enabled):
        observed.append(enabled)
        raise ProbeComplete
    monkeypatch.setattr(torch.amp, "GradScaler", scaler)
    with pytest.raises(ProbeComplete):
        actual.train_attention(cfg, tmp_path, datasets, split, fresh=True)
    # cuda:1 has no BF16: AMP must choose FP16 with scaling, despite cuda:0 BF16.
    assert observed == [True]
    assert torch.cuda.current_device() == 0


@pytest.fixture(autouse=True)
def isolated_random_state():
    import random
    import numpy as np
    import torch
    state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        yield
    finally:
        random.setstate(state[0])
        np.random.set_state(state[1])
        torch.set_rng_state(state[2])
        torch.set_num_threads(threads)
