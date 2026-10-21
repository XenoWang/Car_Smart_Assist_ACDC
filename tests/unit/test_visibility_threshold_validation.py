from __future__ import annotations

import math

import pytest

from car_smart_assist.perception.visibility.gate import GateThresholds, VisibilityGate, VisibilityLevel
from car_smart_assist.perception.visibility.scorer import InformationFeatures, VisibilityScore


def _score(information):
    return VisibilityScore(
        path="synthetic", recon_mean=0.0, recon_p90_block=0.0, recon_z=0.0,
        features=InformationFeatures(contrast=1.0, entropy=1.0, edge_density=0.0, hf_ratio=0.0),
        information=information,
    )


@pytest.mark.parametrize("field", ["info_blind", "info_degraded", "z_degraded", "z_blind"])
@pytest.mark.parametrize("bad", [True, None, [], {}, float("nan"), float("inf"), float("-inf")])
def test_direct_threshold_construction_rejects_non_numeric_or_nonfinite(field, bad):
    with pytest.raises((TypeError, ValueError)):
        GateThresholds(**{field: bad})


@pytest.mark.parametrize("field", ["info_blind", "info_degraded", "z_degraded", "z_blind"])
@pytest.mark.parametrize("bad", [True, None, [], {}, float("nan"), float("inf"), float("-inf")])
def test_config_threshold_construction_rejects_invalid_values(field, bad):
    with pytest.raises((TypeError, ValueError)):
        GateThresholds.from_config({field: bad})


@pytest.mark.parametrize("overrides", [
    {"info_blind": -0.01}, {"info_blind": 1.01}, {"info_degraded": -0.01},
    {"info_degraded": 1.01}, {"info_blind": 0.6, "info_degraded": 0.5},
    {"z_degraded": 1.0, "z_blind": 0.5},
])
def test_direct_and_config_paths_reject_out_of_range_or_inverted_thresholds(overrides):
    with pytest.raises(ValueError):
        GateThresholds(**overrides)
    with pytest.raises(ValueError):
        GateThresholds.from_config(overrides)


def test_legal_zero_one_and_equal_boundaries_are_preserved():
    direct = GateThresholds(info_blind=0.0, info_degraded=1.0, z_degraded=-1.0, z_blind=-1.0)
    configured = GateThresholds.from_config({
        "info_blind": 0, "info_degraded": 1, "z_degraded": -1, "z_blind": -1,
    })
    assert direct == configured
    assert direct.info_blind == 0.0 and direct.info_degraded == 1.0


def test_valid_default_low_information_still_routes_to_blind():
    thresholds = GateThresholds()
    score = _score(math.nextafter(thresholds.info_blind, 0.0))
    assert VisibilityGate(thresholds).judge(score).level is VisibilityLevel.BLIND
