"""密集平均、谷底边界、来源权重和保存恢复。"""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from car_smart_assist.config.weather_attention import (
    load_weather_attention_config,
    validate_weather_attention_config,
)
from car_smart_assist.perception.weather_attention_ablation import (
    experiment_configs,
    load_ablation_config,
)
from car_smart_assist.perception.weather_swad import (
    DenseAverage,
    LossValley,
    SWADTracker,
    source_balanced_bce,
)


def options(**changes):
    return {
        "enabled": True,
        "n_converge": 2,
        "n_tolerance": 3,
        "tolerance_ratio": 0.1,
        "start_epoch": 1,
        **changes,
    }


def segment(epoch, value, count=1):
    return {
        "epoch": epoch,
        "start_step": epoch,
        "end_step": epoch,
        "count": count,
        "vector": torch.tensor([float(value)]),
    }


def test_dense_average_includes_every_update_without_mutating_weights_or_rng():
    model = torch.nn.Linear(1, 1, bias=False)
    average = DenseAverage(model)
    rng = torch.get_rng_state().clone()
    with torch.no_grad():
        for value in (0.0, 4.0, 8.0):
            model.weight.fill_(value)
            average.update()
    result = average.take(1, 3)
    assert result["vector"].item() == 4.0
    assert result["count"] == 3 and result["start_step"] == 1 and result["end_step"] == 3
    assert model.weight.item() == 8.0
    assert torch.equal(rng, torch.get_rng_state())
    assert average.count == 0


def test_valley_excludes_overfit_tail_and_weights_successful_updates():
    valley = LossValley(options())
    for epoch, (value, count, loss) in enumerate(
        ((2, 1, 1), (6, 3, 1), (10, 2, 1), (30, 1, 2), (40, 1, 2), (50, 1, 2)), 1
    ):
        valley.add(segment(epoch, value, count), loss)
    assert valley.dead
    assert valley.finish().item() == pytest.approx((2 + 18 + 20) / 6)
    assert valley.count == 6
    assert valley.start_epoch == 1 and valley.end_epoch == 3
    assert valley.threshold == pytest.approx(1.1)


def test_finish_keeps_valid_pending_segments_but_excludes_bad_suffix():
    valley = LossValley(options(n_tolerance=6))
    for epoch, loss in enumerate((1, 1, 5), 1):
        valley.add(segment(epoch, epoch), loss)
    assert valley.finish().item() == pytest.approx(1.5)
    assert valley.end_epoch == 2
    assert valley.finish().item() == pytest.approx(1.5)


def test_no_stable_valley_is_reported_instead_of_mislabeling_last_weight():
    model = torch.nn.Linear(1, 1, bias=False)
    tracker = SWADTracker(model, options())
    for epoch, loss in enumerate((3, 2, 1), 1):
        tracker.update(epoch)
        tracker.end_epoch(epoch, loss)
    final, info = tracker.final_model(model)
    assert final is None and not info["converged"]
    assert info["fallback_reason"]


def test_tracker_resume_roundtrip_and_repeat_export_are_identical(tmp_path):
    model = torch.nn.Linear(1, 1, bias=False)
    model.cfg = SimpleNamespace(train={"channels_last": False})
    uninterrupted = SWADTracker(model, options(start_epoch=2))
    resumed = SWADTracker(model, options(start_epoch=2))
    for epoch in range(1, 7):
        with torch.no_grad():
            model.weight.fill_(epoch)
        for _ in range(2):
            uninterrupted.update(epoch)
            resumed.update(epoch)
        uninterrupted.end_epoch(epoch, 1.0)
        resumed.end_epoch(epoch, 1.0)
        if epoch == 3:
            path = tmp_path / "swad.pt"
            torch.save(resumed.state_dict(), path)
            resumed = SWADTracker(model, options(start_epoch=2))
            resumed.load_state_dict(torch.load(path, weights_only=True))
    expected, info = uninterrupted.final_model(model)
    actual, other_info = resumed.final_model(model)
    repeated, repeated_info = resumed.final_model(model)
    assert info == other_info == repeated_info
    assert info["averaged_updates"] == 10
    assert expected.weight.item() == pytest.approx(4.0)
    assert torch.equal(actual.weight, expected.weight) and torch.equal(
        actual.weight, repeated.weight
    )
    assert model.weight.item() == 6.0


def test_source_balanced_bce_is_invariant_to_repetition_inside_a_source():
    cfg = SimpleNamespace(
        attributes=("fog", "rain", "snow", "night"),
        evaluation={"selection_weights": {"fog": 2.0, "rain": 2.0, "snow": 1.0, "night": 1.0}},
    )

    def dataset(repeat):
        return SimpleNamespace(
            samples=[SimpleNamespace(path=Path("a"))] + [SimpleNamespace(path=None)] * repeat,
            targets=np.zeros((repeat + 1, 4), np.float32),
            mask=np.ones((repeat + 1, 4), np.float32),
        )

    scores = [
        source_balanced_bce(dataset(n), np.array([[0.9] * 4] + [[0.1] * 4] * n), cfg)
        for n in (1, 9)
    ]
    assert scores[0] == pytest.approx(scores[1])
    assert scores[0] == pytest.approx((-np.log(0.1) - np.log(0.9)) / 2)


@pytest.mark.parametrize(
    "change",
    [
        {"n_converge": 0},
        {"n_tolerance": 2},
        {"tolerance_ratio": float("nan")},
        {"enabled": 1},
        {"start_epoch": 5},
    ],
)
def test_invalid_swad_configuration_rejected(change):
    root = Path(__file__).resolve().parents[2]
    cfg = load_weather_attention_config(root / "configs/model/weather_attention_swad.yaml")
    with pytest.raises(ValueError):
        validate_weather_attention_config(
            replace(cfg, train={**cfg.train, "swad": {**cfg.train["swad"], **change}})
        )


def test_swad_recipe_keeps_best_overall_control_and_fixed_budget():
    root = Path(__file__).resolve().parents[2]
    cfg = load_weather_attention_config(root / "configs/model/weather_attention_swad.yaml")
    base = load_weather_attention_config(root / "configs/model/weather_attention.yaml")
    plan = load_ablation_config(root / "configs/model/weather_attention_ablation.yaml")
    control = experiment_configs(base, plan, root)["reduced_tuning"]
    assert cfg.backbone == control.backbone and cfg.head == control.head
    assert cfg.data == control.data and cfg.loss == control.loss
    assert {key: value for key, value in cfg.train.items() if key != "swad"} == control.train
    assert (
        cfg.train["epochs"] == 300
        and cfg.train["schedule_epochs"] == 500
        and cfg.data["seed"] == 42
    )
    assert cfg.checkpoint_root != control.checkpoint_root
