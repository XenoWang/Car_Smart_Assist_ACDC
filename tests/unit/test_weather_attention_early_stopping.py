from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from car_smart_assist.config.weather_attention import load_weather_attention_config
from car_smart_assist.perception.weather_attention_training import train_attention


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def cfg():
    return load_weather_attention_config(
        ROOT / "configs/model/weather_attention_retrain_300.yaml"
    )


class TinyWeatherModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(1, 1)
        self.head = nn.Parameter(torch.zeros(4))

    def set_training_epoch(self, _epoch):
        pass

    def forward(self, values):
        return self.backbone(values).expand(-1, 4) + self.head


class TinyWeatherDataset:
    def __init__(self, signature):
        self.signature = signature
        self.targets = np.zeros((2, 4), dtype=np.float32)
        self.mask = np.ones((2, 4), dtype=np.float32)
        self.samples = [SimpleNamespace() for _ in range(2)]

    def __len__(self):
        return len(self.samples)


@pytest.mark.parametrize(
    (
        "planned_epochs",
        "score_values",
        "expected_termination",
        "expected_best_epoch",
        "expected_wait",
        "expected_stopped",
    ),
    [
        (6, [0.5, 0.50005, 0.50011, 0.50015, 0.50010], "early_stopping", 4, 2, True),
        (5, [0.5, 0.50005, 0.50011, 0.50015, 0.50016], "budget_completed", 5, 2, False),
    ],
)
def test_weather_best_and_early_stop_use_distinct_improvement_thresholds(
    cfg,
    tmp_path,
    monkeypatch,
    planned_epochs,
    score_values,
    expected_termination,
    expected_best_epoch,
    expected_wait,
    expected_stopped,
):
    import car_smart_assist.perception.weather_attention_training as training

    cfg = replace(
        cfg,
        train={
            **cfg.train,
            "epochs": planned_epochs,
            "schedule_epochs": planned_epochs,
            "batch_size": 2,
            "amp": False,
            "early_stopping": {"enabled": True, "patience": 2, "min_delta": 0.0001},
        },
        evaluation={
            **cfg.evaluation,
            "old_checkpoint": "old.pt",
            "asl_checkpoint": "asl.pt",
            "attention_references": {},
        },
        checkpoint_root="checkpoints",
        report_dir="reports",
        report_file="reports/REPORT.md",
    )
    pretrained = tmp_path / "local_pretrained.pth"
    pretrained.write_bytes(b"fixture weights")
    (tmp_path / "old.pt").write_bytes(b"old model baseline")
    (tmp_path / "asl.pt").write_bytes(b"asl baseline")

    validation = TinyWeatherDataset("validation")
    datasets = {
        "train": TinyWeatherDataset("train"),
        "validation": validation,
        "calibration": TinyWeatherDataset("calibration"),
        "test": TinyWeatherDataset("test"),
    }
    scores = iter(score_values)

    monkeypatch.setattr(training, "select_device", lambda _device: torch.device("cpu"))
    monkeypatch.setattr(training, "ensure_pretrained_weights", lambda *_: pretrained)
    monkeypatch.setattr(training, "attention_model_config", lambda _cfg: {"tiny": True})
    monkeypatch.setattr(training, "WeatherAttention", lambda *_args, **_kwargs: TinyWeatherModel())
    monkeypatch.setattr(training, "attention_sampling_weights", lambda *_: torch.ones(2))
    monkeypatch.setattr(
        training,
        "batch_images",
        lambda _dataset, indices, _device, _cfg: torch.ones((len(indices), 1)),
    )
    monkeypatch.setattr(training, "_augment", lambda values, _cfg: values)
    monkeypatch.setattr(
        training,
        "masked_weather_loss",
        lambda logits, _labels, _mask, _loss, _cfg: logits.square().mean(),
    )
    monkeypatch.setattr(training, "attention_probabilities", lambda *_: np.zeros((2, 4), np.float32))
    monkeypatch.setattr(training, "detailed_metrics", lambda *_: {"synthetic": True})
    monkeypatch.setattr(training, "optimize_attention_inference", lambda model: model)

    def controlled_selection(dataset, _probabilities, _cfg):
        per_attribute = {
            name: {"source_balanced_ap": 0.5} for name in cfg.attributes
        }
        if dataset is validation:
            return {}, {"score": next(scores), "per_attribute": per_attribute}
        return {name: 0.5 for name in cfg.attributes}, {
            "score": 0.5,
            "per_attribute": per_attribute,
        }

    monkeypatch.setattr(training, "balanced_thresholds", controlled_selection)
    monkeypatch.setattr(training, "model_report", lambda *_: None)

    from _model_test_helpers import synthetic_attention_preflight
    monkeypatch.setattr(training, "preflight_attention", synthetic_attention_preflight)
    model, result = train_attention(cfg, tmp_path, datasets, {"signature": "fixture"}, fresh=True)
    best = torch.load(tmp_path / "checkpoints/best.pt", map_location="cpu", weights_only=False)
    last = torch.load(tmp_path / "checkpoints/last.pt", map_location="cpu", weights_only=False)

    assert result["best_epoch"] == expected_best_epoch  # Tiny gains still update best.
    assert result["trained_through_epoch"] == 5
    assert result["termination_reason"] == expected_termination
    assert result["early_stopping"] == {
        "best_score": score_values[2],
        "wait": expected_wait,
        "stopped": expected_stopped,
    }
    assert best["epoch"] == expected_best_epoch
    assert last["epoch"] == 5
    assert last["planned_epochs"] == planned_epochs
    assert last["termination_reason"] == expected_termination
    assert [row["best_epoch"] for row in last["history"]][-1] == expected_best_epoch
    assert all(
        torch.equal(model.state_dict()[name], best["model_state"][name])
        for name in best["model_state"]
    )

    if expected_stopped:
        _, resumed_result = train_attention(
            cfg, tmp_path, datasets, {"signature": "fixture"}, fresh=False
        )
        resumed_last = torch.load(
            tmp_path / "checkpoints/last.pt", map_location="cpu", weights_only=False
        )
        assert resumed_result["trained_through_epoch"] == 5
        assert resumed_result["termination_reason"] == "early_stopping"
        assert resumed_last["epoch"] == 5
        assert resumed_last["early_stopping"] == {
            "best_score": 0.50011,
            "wait": 2,
            "stopped": True,
        }

