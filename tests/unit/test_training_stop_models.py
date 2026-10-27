from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from car_smart_assist.utils.stop import StopController


@pytest.fixture(autouse=True)
def _small_cpu_thread_pool():
    previous = torch.get_num_threads()
    rng = torch.get_rng_state()
    torch.set_num_threads(2)
    try:
        yield
    finally:
        torch.set_num_threads(previous)
        torch.set_rng_state(rng)


class _TinyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(1, 1)
        self.head = nn.Parameter(torch.zeros(4))

    def set_training_epoch(self, _epoch):
        pass

    def forward(self, values):
        return self.backbone(values).expand(-1, 4) + self.head


class _TinyAttentionDataset:
    def __init__(self, signature):
        self.signature = signature
        self.targets = np.zeros((2, 4), dtype=np.float32)
        self.mask = np.ones_like(self.targets)
        self.samples = [SimpleNamespace(), SimpleNamespace()]

    def __len__(self):
        return len(self.samples)


def _nested_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        assert torch.equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _nested_equal(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            _nested_equal(left, right)
    elif isinstance(expected, np.ndarray):
        assert np.array_equal(actual, expected)
    else:
        assert actual == expected


def _attention_setup(root, monkeypatch, *, stop_controller=None, pause_on_first=False):
    import car_smart_assist.perception.weather_attention_training as training

    from car_smart_assist.config.weather_attention import load_weather_attention_config

    cfg = load_weather_attention_config(
        Path(__file__).resolve().parents[2] / "configs/model/weather_attention_retrain_300.yaml"
    )
    cfg = replace(
        cfg,
        train={**cfg.train, "epochs": 3, "schedule_epochs": 3, "batch_size": 2,
               "amp": False, "early_stopping": {"enabled": True, "patience": 1, "min_delta": 0.1},
               "swad": {"enabled": False}, "channels_last": False, "l2_sp": 0.0},
        evaluation={**cfg.evaluation, "old_checkpoint": "old.pt", "asl_checkpoint": "asl.pt",
                    "attention_references": {}},
        checkpoint_root="checkpoints", report_dir="reports", report_file="reports/REPORT.md",
    )
    root.mkdir(parents=True, exist_ok=True)
    pretrained = root / "pretrained.pth"
    pretrained.write_bytes(b"synthetic local initializer")
    (root / "old.pt").write_bytes(b"old model baseline")
    (root / "asl.pt").write_bytes(b"asl baseline")
    validation = _TinyAttentionDataset("validation")
    datasets = {name: _TinyAttentionDataset(name) for name in ("train", "calibration", "test")}
    datasets["validation"] = validation
    scores = iter([0.5, 0.4, 0.3])
    training_batches = []

    monkeypatch.setattr(training, "select_device", lambda _device: torch.device("cpu"))
    monkeypatch.setattr(training, "ensure_pretrained_weights", lambda *_args: pretrained)
    monkeypatch.setattr(training, "attention_model_config", lambda _cfg: {"tiny": True})
    monkeypatch.setattr(training, "WeatherAttention", lambda *_args, **_kwargs: _TinyAttention())
    monkeypatch.setattr(training, "attention_sampling_weights", lambda *_args: torch.ones(2))

    def batch_images(_dataset, indices, _device, _cfg):
        training_batches.append(1)
        return torch.ones((len(indices), 1))

    monkeypatch.setattr(training, "batch_images", batch_images)
    monkeypatch.setattr(training, "_augment", lambda values, _cfg: values)
    monkeypatch.setattr(training, "masked_weather_loss", lambda logits, *_args: logits.square().mean())
    monkeypatch.setattr(training, "attention_probabilities", lambda *_args: np.zeros((2, 4), np.float32))
    monkeypatch.setattr(training, "detailed_metrics", lambda *_args: {"synthetic": True})
    monkeypatch.setattr(training, "optimize_attention_inference", lambda model: model)
    monkeypatch.setattr(training, "model_report", lambda *_args: None)
    from _model_test_helpers import synthetic_attention_preflight
    monkeypatch.setattr(training, "preflight_attention", synthetic_attention_preflight)

    def controlled_selection(dataset, _values, _cfg):
        row = {name: {"source_balanced_ap": 0.5} for name in cfg.attributes}
        if dataset is validation:
            score = next(scores)
            if pause_on_first and len(training_batches) and not stop_controller.requested:
                stop_controller.request()
            return {}, {"score": score, "per_attribute": row}
        return dict.fromkeys(cfg.attributes, 0.5), {"score": 0.5, "per_attribute": row}

    monkeypatch.setattr(training, "balanced_thresholds", controlled_selection)
    return training, cfg, datasets, {"signature": "tiny-split"}, training_batches


def test_attention_optimizer_stop_continue_and_completed_noop(tmp_path, monkeypatch):
    from car_smart_assist.perception.weather_attention_training import train_attention

    training, cfg, ref_sets, split, _ = _attention_setup(tmp_path / "reference", monkeypatch)
    _, reference = train_attention(cfg, tmp_path / "reference", ref_sets, split, fresh=True)
    expected = torch.load(tmp_path / "reference/checkpoints/last.pt", map_location="cpu", weights_only=True)

    controller = StopController(tmp_path / "resumed/STOP", default_directory=tmp_path / "resumed")
    training, cfg, resumed_sets, split, batches = _attention_setup(
        tmp_path / "resumed", monkeypatch, stop_controller=controller, pause_on_first=True
    )
    _, stopped = train_attention(
        cfg, tmp_path / "resumed", resumed_sets, split, fresh=True, stop_controller=controller
    )
    first = torch.load(tmp_path / "resumed/checkpoints/last.pt", map_location="cpu", weights_only=True)
    first_best = torch.load(tmp_path / "resumed/checkpoints/best.pt", map_location="cpu", weights_only=True)
    assert stopped["manual_stop"] and stopped["termination_reason"] == "manual_stop"
    assert stopped["trained_through_epoch"] == 1 and not stopped["calibrated"]
    assert first["epoch"] == 1 and first["planned_epochs"] == 3
    assert first_best["epoch"] == 1

    controller = StopController(tmp_path / "resumed/STOP", default_directory=tmp_path / "resumed")
    _, continued = train_attention(
        cfg, tmp_path / "resumed", resumed_sets, split, stop_controller=controller, continue_only=True
    )
    actual = torch.load(tmp_path / "resumed/checkpoints/last.pt", map_location="cpu", weights_only=True)
    best = torch.load(tmp_path / "resumed/checkpoints/best.pt", map_location="cpu", weights_only=True)
    assert continued["trained_through_epoch"] == 2
    assert actual["epoch"] == 2 and actual["planned_epochs"] == 3
    assert actual["training_complete"] and actual["termination_reason"] == "early_stopping"
    assert best["epoch"] == 1  # Validation scores did not improve after the stopped epoch.
    for key in ("model_state", "optimizer_state", "generator_state", "torch_rng"):
        _nested_equal(actual[key], expected[key])

    before_batches = len(batches)
    controller = StopController(tmp_path / "resumed/STOP", default_directory=tmp_path / "resumed")
    train_attention(
        cfg, tmp_path / "resumed", resumed_sets, split, stop_controller=controller, continue_only=True
    )
    final = torch.load(tmp_path / "resumed/checkpoints/last.pt", map_location="cpu", weights_only=True)
    assert final["epoch"] == 2 and len(final["history"]) == 2
    assert len(batches) == before_batches  # Completed explicit continue performs no optimizer steps.
    _nested_equal(final["optimizer_state"], actual["optimizer_state"])


def test_pyramid_optimizer_stop_continue_and_completed_noop(tmp_path, monkeypatch):
    from test_weather_pyramid import build_sets, tiny_config

    import car_smart_assist.perception.weather_pyramid_training as training

    cfg = tiny_config()
    cfg = replace(cfg, train={**cfg.train, "epochs": 2, "min_epochs": 2, "force_epochs": True})
    datasets, split = build_sets(tmp_path / "data", cfg)
    reference_root = tmp_path / "reference"
    resumed_root = tmp_path / "resumed"
    stop_calls = {"enabled": False, "done": False, "calibration": 0}
    validation_metrics = {
        "macro_ap": 0.5,
        "per_attribute": {name: {"average_precision": 0.5} for name in cfg.attributes},
    }

    def probabilities(model, dataset, _cfg, _device):
        if dataset is datasets["validation"] and stop_calls["enabled"] and not stop_calls["done"]:
            stop_calls["done"] = True
            controller.request()
        if dataset is datasets["calibration"]:
            stop_calls["calibration"] += 1
        return np.full((len(dataset), len(cfg.attributes)), 0.5, dtype=np.float32)

    monkeypatch.setattr(training, "grouped_metrics", lambda *_args: validation_metrics)
    monkeypatch.setattr(training, "pyramid_probabilities", probabilities)
    monkeypatch.setattr(training, "calibrate_thresholds", lambda *_args: dict.fromkeys(cfg.attributes, 0.5))

    training.train_pyramid_variant(cfg, reference_root, datasets, split, "asl", fresh=True)
    expected = torch.load(reference_root / "checkpoints/asl/last.pt", map_location="cpu", weights_only=True)

    controller = StopController(resumed_root / "STOP", default_directory=resumed_root)
    stop_calls.update(enabled=True, done=False, calibration=0)
    _, stopped = training.train_pyramid_variant(
        cfg, resumed_root, datasets, split, "asl", fresh=True, stop_controller=controller
    )
    first = torch.load(resumed_root / "checkpoints/asl/last.pt", map_location="cpu", weights_only=True)
    first_best = torch.load(resumed_root / "checkpoints/asl/best.pt", map_location="cpu", weights_only=True)
    assert stopped["manual_stop"] and stopped["trained_through_epoch"] == 1
    assert not stopped["calibrated"] and stop_calls["calibration"] == 0
    assert first["epoch"] == 1 and first["planned_epochs"] == 2
    assert first_best["epoch"] == 1

    controller = StopController(resumed_root / "STOP", default_directory=resumed_root)
    stop_calls.update(enabled=False, done=False, calibration=0)
    _, continued = training.train_pyramid_variant(
        cfg, resumed_root, datasets, split, "asl", stop_controller=controller, continue_only=True
    )
    actual = torch.load(resumed_root / "checkpoints/asl/last.pt", map_location="cpu", weights_only=True)
    assert continued["trained_through_epoch"] == 2
    assert actual["epoch"] == actual["planned_epochs"] == 2
    for key in ("model_state", "optimizer_state", "generator_state", "torch_rng"):
        _nested_equal(actual[key], expected[key])

    stop_calls["validation_after_completion"] = 0
    original_probabilities = training.pyramid_probabilities

    def count_noop(model, dataset, options, device):
        if dataset is datasets["validation"]:
            stop_calls["validation_after_completion"] += 1
        return original_probabilities(model, dataset, options, device)

    monkeypatch.setattr(training, "pyramid_probabilities", count_noop)
    controller = StopController(resumed_root / "STOP", default_directory=resumed_root)
    training.train_pyramid_variant(
        cfg, resumed_root, datasets, split, "asl", stop_controller=controller, continue_only=True
    )
    final = torch.load(resumed_root / "checkpoints/asl/last.pt", map_location="cpu", weights_only=True)
    assert final["epoch"] == 2 and len(final["history"]) == 2
    assert stop_calls["validation_after_completion"] == 1  # Report validation only, no train epochs.
    _nested_equal(final["optimizer_state"], actual["optimizer_state"])


def _write_cnn_fixture(root, cfg):
    for attr_index, condition in enumerate(cfg.attributes):
        for split, count in (("train", 2), ("val", 1)):
            sequence = f"{condition}_{split}_sequence"
            directory = root / "acdc/rgb_anon" / condition / split / sequence
            directory.mkdir(parents=True)
            for index in range(count):
                image = np.full((40, 60, 3), 20 + attr_index * 50 + index, dtype=np.uint8)
                Image.fromarray(image).save(directory / f"{sequence}_frame_{index:06d}_rgb_anon.png")
    manifest = root / "data/processed/manifests/acdc.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"invalid": []}), encoding="utf-8")


def test_legacy_cnn_optimizer_stop_continue_and_completed_noop(tmp_path, monkeypatch):
    from test_weather_classifier import small_config

    import car_smart_assist.perception.weather_training as training

    cfg = small_config(epochs=3)
    cfg = replace(
        cfg,
        train={**cfg.train, "acdc_cleaning_manifest": "data/processed/manifests/acdc.json", "patience": 10},
    )
    reference_root, resumed_root = tmp_path / "reference", tmp_path / "resumed"
    _write_cnn_fixture(reference_root, cfg)
    _write_cnn_fixture(resumed_root, cfg)
    original_metrics = training.attribute_metrics

    def complete_reference(*args):
        return original_metrics(*args)

    monkeypatch.setattr(training, "attribute_metrics", complete_reference)
    torch.manual_seed(123)
    training.train_weather(cfg, reference_root, resume=False)
    expected = torch.load(reference_root / "checkpoints/weather_attributes/last.pt", map_location="cpu", weights_only=True)

    controller = StopController(resumed_root / "checkpoints/weather_attributes/STOP",
                                default_directory=resumed_root / "checkpoints/weather_attributes")
    metrics_called = {"value": False}

    def request_after_validation(*args):
        result = original_metrics(*args)
        if not metrics_called["value"]:
            metrics_called["value"] = True
            controller.request()
        return result

    monkeypatch.setattr(training, "attribute_metrics", request_after_validation)
    torch.manual_seed(123)
    stopped = training.train_weather(cfg, resumed_root, resume=False, stop_controller=controller)
    first = torch.load(resumed_root / "checkpoints/weather_attributes/last.pt", map_location="cpu", weights_only=True)
    assert stopped.manual_stop and stopped.termination_reason == "manual_stop"
    assert first["epoch"] == 1 and first["planned_epochs"] == 3

    monkeypatch.setattr(training, "attribute_metrics", complete_reference)
    controller = StopController(resumed_root / "checkpoints/weather_attributes/STOP",
                                default_directory=resumed_root / "checkpoints/weather_attributes")
    training.train_weather(cfg, resumed_root, resume=True, stop_controller=controller, continue_only=True)
    actual = torch.load(resumed_root / "checkpoints/weather_attributes/last.pt", map_location="cpu", weights_only=True)
    assert actual["epoch"] == actual["planned_epochs"] == 3
    _nested_equal(actual["model_state"], expected["model_state"])
    _nested_equal(actual["optimizer_state"], expected["optimizer_state"])
    for key in ("shuffle_state", "torch_rng"):
        _nested_equal(actual[key], expected[key])

    snapshot = actual["optimizer_state"]
    controller = StopController(resumed_root / "checkpoints/weather_attributes/STOP",
                                default_directory=resumed_root / "checkpoints/weather_attributes")
    training.train_weather(cfg, resumed_root, resume=True, stop_controller=controller, continue_only=True)
    final = torch.load(resumed_root / "checkpoints/weather_attributes/last.pt", map_location="cpu", weights_only=True)
    assert final["epoch"] == 3
    _nested_equal(final["optimizer_state"], snapshot)
