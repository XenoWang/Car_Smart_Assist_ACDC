from __future__ import annotations

import json
import math

import pytest

from car_smart_assist.perception.visibility.gate import VisibilityLevel, VisibilityVerdict


@pytest.mark.parametrize(
    ("information", "recon_z", "expected_information", "expected_recon_z"),
    [
        (0.25, -1.25, 0.25, -1.25),
        (float("nan"), -1.25, None, -1.25),
        (float("inf"), -1.25, None, -1.25),
        (float("-inf"), -1.25, None, -1.25),
        (0.25, float("nan"), 0.25, None),
        (0.25, float("inf"), 0.25, None),
        (0.25, float("-inf"), 0.25, None),
        (float("nan"), float("nan"), None, None),
    ],
)
def test_verdict_json_converts_nonfinite_fields_without_mutating_internal_values(
    information, recon_z, expected_information, expected_recon_z
):
    verdict = VisibilityVerdict(
        level=VisibilityLevel.BLIND,
        reason="info_low",
        information=information,
        recon_z=recon_z,
    )

    payload = verdict.to_dict()
    json.dumps(payload, ensure_ascii=False, allow_nan=False)

    assert payload["information"] == expected_information
    assert payload["recon_z"] == expected_recon_z
    assert (
        math.isnan(verdict.information)
        if math.isnan(information)
        else verdict.information == information
    )
    assert (
        math.isnan(verdict.recon_z)
        if math.isnan(recon_z)
        else verdict.recon_z == recon_z
    )
