"""The public weather candidate config rejects the withdrawn Group DRO option."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from car_smart_assist.config.weather_attention import load_weather_attention_config


@pytest.mark.parametrize("enabled", [True, False])
def test_weather_attention_config_rejects_withdrawn_group_dro_field(tmp_path, enabled):
    project_root = Path(__file__).resolve().parents[2]
    baseline_path = project_root / "configs/model/weather_attention_a.yaml"
    document = yaml.safe_load(baseline_path.read_text(encoding="utf-8"))

    # The current shipped config is a valid positive control for this schema path.
    load_weather_attention_config(baseline_path)
    assert "group_dro" not in document["weather_attention"]["train"]

    document["weather_attention"]["train"]["group_dro"] = {"enabled": enabled, "step_size": 0.01}
    candidate_path = tmp_path / f"weather_attention_group_dro_{enabled}.yaml"
    candidate_path.write_text(
        yaml.safe_dump(document, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as exc_info:
        load_weather_attention_config(candidate_path)
    assert "train 包含未知配置键：group_dro" in str(exc_info.value)
