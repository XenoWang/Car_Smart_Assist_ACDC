"""追加阶段只按验证选择；失败时恢复 A 组，测试分数不参与决定。"""

import copy
from dataclasses import replace
from pathlib import Path

import pytest

from car_smart_assist.config.weather_attention import (
    load_weather_attention_config,
    validate_weather_attention_config,
)
from car_smart_assist.perception.weather_extension import choose_extension


@pytest.mark.parametrize(
    "swad,success", [(None, False), (0.93, False), (0.95, False), (0.96, True)]
)
def test_extension_requires_beating_a_and_same_run_single(swad, success):
    result = {
        "models": {
            "reduced_tuning": {"selection": {"score": 0.94}},
            "single_best": {"selection": {"score": 0.95}},
        }
    }
    if swad is not None:
        result["models"]["swad"] = {"selection": {"score": swad}, "test": {"macro_f1": 0.1}}
    policy = {
        "candidate": "swad",
        "reference": "reduced_tuning",
        "require_beating_single_best": True,
        "min_validation_gain": 1e-12,
    }
    expected = choose_extension(result, policy)
    assert expected["swad_effective"] is success
    assert expected["needs_alternative_research"] is not success
    changed = copy.deepcopy(result)
    for model in changed["models"].values():
        model["test"] = {"macro_f1": 1.0, "rain_f1": 1.0}
    assert choose_extension(changed, policy) == expected


def test_invalid_validation_score_cannot_change_default_route():
    result = {
        "models": {
            name: {"selection": {"score": score}}
            for name, score in (
                ("reduced_tuning", 0.94),
                ("single_best", 0.95),
                ("swad", float("nan")),
            )
        }
    }
    with pytest.raises(ValueError, match="非有限"):
        choose_extension(
            result,
            {
                "candidate": "swad",
                "reference": "reduced_tuning",
                "require_beating_single_best": True,
                "min_validation_gain": 1e-12,
            },
        )


@pytest.mark.parametrize(
    "key,value",
    [
        ("additional_epochs", 599),
        ("checkpoint_sha256", "invalid"),
        ("source_epoch", -1),
        ("start_factor", float("nan")),
    ],
)
def test_invalid_continuation_recipe_is_rejected(key, value):
    root = Path(__file__).resolve().parents[2]
    cfg = load_weather_attention_config(
        root / "configs/model/weather_attention_swad_extend_600.yaml"
    )
    with pytest.raises(ValueError):
        validate_weather_attention_config(
            replace(
                cfg, train={**cfg.train, "continuation": {**cfg.train["continuation"], key: value}}
            )
        )
