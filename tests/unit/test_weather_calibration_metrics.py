"""Calibration-error oracles from known labels, exact probabilities and log identities."""
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from car_smart_assist.perception.weather_attention_training import detailed_metrics
from car_smart_assist.perception.weather_training import WeatherSample

ATTRIBUTES = ("fog", "rain", "snow", "night")
CFG = SimpleNamespace(attributes=ATTRIBUTES)
THRESHOLDS = dict.fromkeys(ATTRIBUTES, .5)


def dataset(labels, mask=None):
    targets = np.asarray(labels, dtype=np.float64).reshape(-1, 4)
    return SimpleNamespace(
        targets=targets,
        mask=np.ones_like(targets) if mask is None else np.asarray(mask, dtype=np.float64),
        samples=[
            WeatherSample(tuple(int(value) for value in row), f"group-{index}", "synthetic",
                          path=Path(f"synthetic-{index}.png"))
            for index, row in enumerate(targets)
        ],
    )


@pytest.mark.parametrize("empty", [False, True])
def test_attributes_without_known_labels_have_null_errors_and_strict_json(empty):
    data = dataset([] if empty else [[0, 0, 0, 0], [1, 1, 1, 1]])
    data.mask[:, 0] = 0
    probabilities = np.full(data.targets.shape, .5)
    report = detailed_metrics(data, probabilities, CFG, THRESHOLDS)
    payload = json.loads(json.dumps(report, allow_nan=False))
    fog = payload["per_attribute"]["fog"]
    assert fog["brier"] is None and fog["nll"] is None
    assert fog["known_positive"] == fog["known_negative"] == 0
    assert fog["unknown"] == (0 if empty else 2)
    if not empty:
        # Other attributes still have one known positive and one known negative.
        for name in ATTRIBUTES[1:]:
            assert payload["per_attribute"][name]["brier"] == .25
            assert payload["per_attribute"][name]["nll"] == pytest.approx(math.log(2))
    else:
        assert all(row["brier"] is None and row["nll"] is None
                   for row in payload["per_attribute"].values())


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_perfect_endpoint_predictions_have_exact_zero_brier(dtype):
    data = dataset([[0, 0, 0, 0], [1, 1, 1, 1]])
    probabilities = np.array([[0, 0, 0, 0], [1, 1, 1, 1]], dtype=dtype)
    original = probabilities.copy()
    report = detailed_metrics(data, probabilities, CFG, THRESHOLDS)
    for row in report["per_attribute"].values():
        assert row["brier"] == 0.0
        # Only NLL is clipped to [1e-7,1-1e-7]; both labels yield -log(1-eps).
        assert row["nll"] == pytest.approx(-math.log1p(-1e-7), rel=0, abs=1e-14)
    np.testing.assert_array_equal(probabilities, original)
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("unknown_probability", [0., 1.])
def test_brier_and_log_loss_use_only_known_labels_with_hand_computed_values(unknown_probability):
    data = dataset([[0, 0, 0, 0], [1, 1, 1, 1], [0, 1, 0, 1]],
                   [[1, 1, 1, 1], [1, 1, 1, 1], [0, 0, 0, 0]])
    probabilities = np.array([[.25] * 4, [.75] * 4, [unknown_probability] * 4])
    report = detailed_metrics(data, probabilities, CFG, THRESHOLDS)
    # Both known errors are (1/4)^2; both log terms are -log(3/4).
    for row in report["per_attribute"].values():
        assert row["brier"] == 1 / 16
        assert row["nll"] == pytest.approx(math.log(4 / 3), rel=0, abs=1e-12)
        assert row["known_positive"] == row["known_negative"] == row["unknown"] == 1
    json.dumps(report, allow_nan=False)
