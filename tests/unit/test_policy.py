"""可解释风险、无法判断时接管、配置和生成器接入的回归测试。"""

from __future__ import annotations

import copy
import itertools
import json
import subprocess
import sys
from dataclasses import replace

import pytest

from car_smart_assist.advisory.generator import AdvisoryGenerator
from car_smart_assist.advisory.policy import assess_risk, evaluate_handover
from car_smart_assist.advisory.prompt.templates import check_wording, render_policy
from car_smart_assist.advisory.schema import (
    PerceptionResult,
    RiskLevel,
    TargetDirection,
    TargetObject,
)
from car_smart_assist.config.policy import load_policy_config
from car_smart_assist.perception.visibility.gate import VisibilityLevel, VisibilityVerdict


@pytest.fixture(scope="module")
def config():
    return load_policy_config()


def target(distance=60.0, **kwargs):
    return replace(TargetObject("car", TargetDirection.LEADING, distance, confidence=1.0), **kwargs)


def scene(*objects, **kwargs):
    return replace(
        PerceptionResult(
            visibility_level="visible",
            road_condition="clear",
            road_condition_confidence=1.0,
            object_detection_available=True,
            objects=list(objects),
        ),
        **kwargs,
    )


@pytest.mark.parametrize("perception", [None, PerceptionResult()])
def test_missing_information_is_unknown_and_requests_takeover(config, perception):
    decision = evaluate_handover(perception, config)
    assert decision.risk.risk_level is RiskLevel.UNKNOWN
    assert decision.should_takeover and decision.unable_to_judge
    assert decision.reasons


def test_empty_detection_requires_explicit_success(config):
    good = evaluate_handover(scene(), config)
    assert good.risk.risk_level is RiskLevel.NONE
    assert not good.should_takeover
    bad = evaluate_handover(scene(object_detection_available=False), config)
    assert bad.risk.risk_level is RiskLevel.UNKNOWN
    assert bad.should_takeover
    assert "detection_unavailable" in bad.risk.triggered


@pytest.mark.parametrize(
    ("distance", "expected"),
    [
        (0, "critical"),
        (14.99, "critical"),
        (15, "warning"),
        (29.99, "warning"),
        (30, "notice"),
        (49.99, "notice"),
        (50, "none"),
        (100, "none"),
    ],
)
def test_strict_distance_boundaries(config, distance, expected):
    decision = evaluate_handover(scene(target(distance)), config)
    assert decision.risk.risk_level.value == expected
    assert decision.should_takeover == (expected == "critical")
    assert not decision.unable_to_judge
    assert f"{distance:.1f} 米" in " ".join(decision.risk.reasons)


@pytest.mark.parametrize("weather", ["fog", "rain", "snow", "night"])
def test_recognized_weather_has_baseline_notice(config, weather):
    decision = evaluate_handover(scene(road_condition=weather), config)
    assert decision.risk.risk_level is RiskLevel.NOTICE
    assert not decision.should_takeover
    assert weather in " ".join(decision.risk.reasons)


def test_weather_is_only_handover_support_when_visibility_is_degraded(config):
    visible = evaluate_handover(scene(weather_attributes=("fog",)), config)
    assert not visible.should_takeover
    assert not visible.unable_to_judge

    degraded = evaluate_handover(
        scene(weather_attributes=("fog",), visibility_level="degraded"), config
    )
    assert degraded.should_takeover
    assert not degraded.unable_to_judge
    assert degraded.action == "request_takeover"
    assert "weather_visibility_auxiliary_handover" in degraded.risk.triggered
    advisory = AdvisoryGenerator().from_perception(
        scene(weather_attributes=("fog",), visibility_level="degraded")
    )
    assert advisory.should_takeover
    assert "恶劣天气且能见度下降" in advisory.text
    assert not check_wording(advisory.text)

    night_only = evaluate_handover(
        scene(weather_attributes=("night",), visibility_level="degraded"), config
    )
    assert not night_only.should_takeover
    disabled = load_policy_config({"weather_handover_enabled": False})
    assert not evaluate_handover(
        scene(weather_attributes=("fog",), visibility_level="degraded"), disabled
    ).should_takeover


def test_weather_and_oncoming_use_configured_multipliers(config):
    assert assess_risk(scene(target(20)), config).risk_level is RiskLevel.WARNING
    assert (
        assess_risk(scene(target(20), road_condition="snow"), config).risk_level
        is RiskLevel.CRITICAL
    )
    assert assess_risk(scene(target(17)), config).risk_level is RiskLevel.WARNING
    assert (
        assess_risk(scene(target(17, direction=TargetDirection.ONCOMING)), config).risk_level
        is RiskLevel.CRITICAL
    )


def test_all_targets_evaluated_not_only_nearest(config):
    p = scene(target(80), target(16), target(17, direction=TargetDirection.ONCOMING))
    risk = assess_risk(p, config)
    assert risk.risk_level is RiskLevel.CRITICAL
    assert risk.primary_target_index == 2
    assert len([code for code in risk.triggered if code.startswith("target_distance")]) == 3
    assert assess_risk(scene(*reversed(p.objects)), config).primary_target_index == 0


@pytest.mark.parametrize("confidence", [None, -0.1, 1.1, float("nan"), float("inf"), True, 0.699])
@pytest.mark.parametrize("field", ["target", "head"])
def test_invalid_or_low_confidence_requests_takeover(config, confidence, field):
    p = scene(target())
    if field == "weather":
        p.road_condition_confidence = confidence
    elif field == "target":
        p.objects[0].confidence = confidence
    else:
        p.handover_level, p.handover_confidence = 0, confidence
    result = evaluate_handover(p, config)
    assert result.should_takeover and result.unable_to_judge
    json.dumps(result.to_dict(), allow_nan=False)


def test_degraded_uses_raw_confidence_for_target_head_and_legacy_weather(config):
    p = scene(
        target(60, confidence=0.7),
        road_condition="clear",
        road_condition_confidence=0.7,
        visibility_level="degraded",
        visibility_confidence_multiplier=0.6,
        handover_level=0,
        handover_confidence=0.7,
    )
    assert not evaluate_handover(p, config).should_takeover
    assert p.visibility_confidence_multiplier == 0.6

    p.objects[0].confidence = 0.699
    result = evaluate_handover(p, config)
    assert result.should_takeover and result.unable_to_judge
    assert "target_confidence_low" in result.risk.triggered

    p.objects[0].confidence = 1.0
    p.handover_confidence = 0.699
    result = evaluate_handover(p, config)
    assert result.should_takeover and result.unable_to_judge

    p.handover_confidence = 1.0
    p.road_condition = "rain"
    p.road_condition_confidence = 0.7
    assert assess_risk(p, config).risk_level is RiskLevel.NOTICE
    p.road_condition_confidence = 0.699
    assert assess_risk(p, config).risk_level is RiskLevel.NOTICE  # DEGRADED itself is NOTICE.
    assert "天气分类置信度不足" in " ".join(assess_risk(p, config).diagnostics)


def test_modern_weather_attributes_override_legacy_and_use_low_probability(config):
    p = scene(
        target(60),
        road_condition="clear",
        road_condition_confidence=1.0,
        weather_attributes=("rain",),
        weather_probabilities={"rain": 0.05},
    )
    risk = assess_risk(p, config)
    assert risk.risk_level is RiskLevel.NOTICE
    assert "weather_recognized" in risk.triggered
    assert not risk.diagnostics

    # Attribute-only callers remain supported, and modern empty predictions do not
    # fall back to a conflicting legacy rain label.
    p.weather_probabilities = {}
    assert assess_risk(p, config).risk_level is RiskLevel.NOTICE
    p.weather_attributes = ()
    p.weather_probabilities = {"rain": 0.95}
    p.road_condition = "rain"
    p.road_condition_confidence = 1.0
    risk = assess_risk(p, config)
    assert risk.risk_level is RiskLevel.NONE
    assert "weather_recognized" not in risk.triggered
    assert risk.diagnostics


def test_weather_attribute_subsets_choose_highest_level_and_max_multiplier(config):
    levels = {"clear": "none", "fog": "notice", "rain": "critical", "snow": "warning", "night": "notice"}
    multipliers = {"clear": 1.0, "fog": 1.3, "rain": 1.1, "snow": 1.8, "night": 1.2}
    custom = load_policy_config(
        {"weather_risk_levels": levels, "weather_risk_multiplier": multipliers}
    )
    rank = {RiskLevel.NONE: 0, RiskLevel.NOTICE: 1, RiskLevel.WARNING: 2, RiskLevel.CRITICAL: 3}
    attributes = ("fog", "rain", "snow", "night")
    subsets = [
        subset
        for size in range(1, len(attributes) + 1)
        for subset in itertools.combinations(attributes, size)
    ]
    for subset in subsets:
        p = scene(weather_attributes=subset, weather_probabilities={name: 0.01 for name in subset})
        risk = assess_risk(p, custom)
        expected = max((RiskLevel(levels[name]) for name in subset), key=rank.__getitem__)
        assert risk.risk_level is expected
        assert risk.triggered.count("weather_recognized") == len(subset)

    # The highest weather risk (rain) and largest distance multiplier (snow)
    # intentionally come from different attributes.
    p = scene(target(20), weather_attributes=("rain", "snow"))
    risk = assess_risk(p, custom)
    assert risk.risk_level is RiskLevel.CRITICAL
    target_reason = next(reason for reason in risk.reasons if reason.startswith("目标[0]"))
    assert "天气倍率 1.8" in target_reason


@pytest.mark.parametrize(
    ("attributes", "probabilities"),
    [
        ("rain", {}),
        (None, {"rain": 0.9}),
        (("rain", "unknown"), {"rain": 0.9}),
        (("rain", 1), {"rain": 0.9}),
        (("rain",), None),
        (("rain",), {"rain": float("nan")}),
        (("rain",), {"rain": float("inf")}),
        (("rain",), {"rain": float("-inf")}),
        (("rain",), {"rain": None}),
        (("rain",), {"rain": "0.9"}),
        (("rain",), {"rain": -0.1}),
        (("rain",), {"rain": 1.1}),
        (("rain",), {"unknown": 0.9}),
        (("rain",), {"rain": 0.9, "unknown": 0.1}),
        (("rain",), {None: 0.9}),
        (("rain",), {"rain": True}),
    ],
)
def test_invalid_modern_weather_is_diagnostic_without_legacy_fallback(
    config, attributes, probabilities
):
    p = scene(
        target(60),
        visibility_level="degraded",
        road_condition="rain",
        road_condition_confidence=1.0,
        weather_attributes=attributes,
        weather_probabilities=probabilities,
    )
    result = evaluate_handover(p, config)
    assert not result.should_takeover
    assert result.risk.risk_level is RiskLevel.NOTICE  # Only the visibility evidence remains.
    assert "weather_recognized" not in result.risk.triggered
    assert "weather_visibility_auxiliary_handover" not in result.risk.triggered
    assert result.risk.diagnostics


def test_modern_weather_order_duplicates_and_auxiliary_rule_are_consistent(config):
    first = scene(weather_attributes=("rain", "fog", "rain"))
    second = scene(weather_attributes=("fog", "rain"))
    assert assess_risk(first, config).risk_level is assess_risk(second, config).risk_level
    assert assess_risk(first, config).triggered.count("weather_recognized") == 2

    for weather in ("fog", "rain", "snow"):
        result = evaluate_handover(
            scene(weather_attributes=(weather,), visibility_level="degraded"), config
        )
        assert result.should_takeover
        assert "weather_visibility_auxiliary_handover" in result.risk.triggered
    night_only = evaluate_handover(
        scene(weather_attributes=("night",), visibility_level="degraded"), config
    )
    assert not night_only.should_takeover


def test_legacy_weather_confidence_uses_original_threshold(config):
    accepted = scene(target(60), road_condition="rain", road_condition_confidence=0.7)
    rejected = scene(target(60), road_condition="rain", road_condition_confidence=0.699)
    assert assess_risk(accepted, config).risk_level is RiskLevel.NOTICE
    assert assess_risk(rejected, config).risk_level is RiskLevel.NONE
    assert assess_risk(rejected, config).diagnostics


def test_weather_policy_does_not_mutate_input(config):
    p = scene(
        target(60),
        weather_attributes=("rain", "fog"),
        weather_probabilities={"rain": 0.2, "fog": 0.3},
    )
    original = copy.deepcopy(p)
    evaluate_handover(p, config)
    assert p == original


@pytest.mark.parametrize("distance", [None, -1, float("nan"), float("inf"), True, "10", 10**400])
def test_invalid_distance_is_not_a_close_object(config, distance):
    decision = evaluate_handover(scene(target(distance)), config)
    assert decision.risk.risk_level is RiskLevel.UNKNOWN
    assert decision.should_takeover
    assert "distance_unknown" in decision.risk.triggered


@pytest.mark.parametrize("direction", [TargetDirection.UNKNOWN, None, "sideways"])
def test_unknown_direction_never_reported_as_leading(config, direction):
    decision = evaluate_handover(scene(target(10, direction=direction)), config)
    assert decision.should_takeover and decision.unable_to_judge
    assert "direction_unknown" in decision.risk.triggered
    assert "前方同向目标" not in " ".join(decision.risk.reasons)


def test_distance_uncertainty_and_required_flag(config):
    p = scene(target(20, distance_uncertainty_m=6))
    assert assess_risk(p, config).risk_level is RiskLevel.CRITICAL
    p.objects[0].distance_uncertainty_m = 11
    assert assess_risk(p, config).risk_level is RiskLevel.UNKNOWN
    strict = load_policy_config({"require_distance_uncertainty": True})
    assert evaluate_handover(scene(target()), strict).unable_to_judge
    assert not evaluate_handover(scene(target()), config).unable_to_judge


def test_unknown_direction_does_not_invent_oncoming_multiplier(config):
    decision = evaluate_handover(scene(target(16, direction=TargetDirection.UNKNOWN)), config)
    assert decision.risk.risk_level is RiskLevel.WARNING
    assert decision.should_takeover and decision.unable_to_judge


@pytest.mark.parametrize("uncertainty", [-1, float("inf"), float("nan"), True])
def test_invalid_uncertainty_requests_takeover(config, uncertainty):
    assert evaluate_handover(
        scene(target(distance_uncertainty_m=uncertainty)), config
    ).unable_to_judge


def test_partial_failure_preserves_known_risk(config):
    decision = evaluate_handover(scene(target(10), target(None)), config)
    assert decision.risk.risk_level is RiskLevel.CRITICAL
    assert decision.should_takeover and decision.unable_to_judge
    assert decision.risk.primary_target_index == 0


@pytest.mark.parametrize(
    ("level", "action", "takeover"),
    [
        (0, "no_action", False),
        (1, "light_notice", False),
        (2, "request_takeover", True),
    ],
)
def test_head_action_independent_from_scene_risk(config, level, action, takeover):
    decision = evaluate_handover(scene(handover_level=level, handover_confidence=1.0), config)
    assert decision.risk.risk_level is RiskLevel.NONE
    assert decision.action == action and decision.should_takeover == takeover
    assert not decision.unable_to_judge


@pytest.mark.parametrize("level", [-1, 3, True, "0", 1.5])
def test_invalid_head_level_requests_takeover(config, level):
    assert evaluate_handover(
        scene(handover_level=level, handover_confidence=1), config
    ).unable_to_judge


def test_head_clear_cannot_override_high_risk(config):
    p = scene(target(5), handover_level=0, handover_confidence=1)
    assert evaluate_handover(p, config).should_takeover
    strict = load_policy_config({"require_handover_head": True})
    assert evaluate_handover(scene(), strict).unable_to_judge


@pytest.mark.parametrize(
    "kwargs",
    [
        {"objects": None},
        {"objects": [{}]},
        {"visibility_level": "blind"},
        {"visibility_confidence_multiplier": 0},
        {"visibility_confidence_multiplier": float("nan")},
    ],
)
def test_invalid_scene_requests_takeover(config, kwargs):
    assert evaluate_handover(scene(**kwargs), config).unable_to_judge


@pytest.mark.parametrize(
    ("weather", "confidence"), [(None, 0.0), ("rain", 0.2), ("unrecognized", 1.0)]
)
def test_weather_uncertainty_is_diagnostic_not_takeover(config, weather, confidence):
    decision = evaluate_handover(
        scene(target(60), road_condition=weather, road_condition_confidence=confidence), config
    )
    assert decision.risk.reliable
    assert decision.risk.risk_level is RiskLevel.NONE
    assert not decision.should_takeover and not decision.unable_to_judge
    assert decision.risk.diagnostics
    assert not decision.risk.unavailable_reasons


def test_input_not_mutated_and_outputs_are_serializable(config):
    p = scene(target(10))
    original = copy.deepcopy(p)
    result = evaluate_handover(p, config)
    json.dumps(result.to_dict(), ensure_ascii=False, allow_nan=False)
    assert p == original
    assert p.to_dict()["object_detection_available"] is True


def test_config_override_does_not_mutate_defaults(config):
    changed = load_policy_config({"distance_thresholds": {"critical": 10}})
    assert assess_risk(scene(target(12)), changed).risk_level is RiskLevel.WARNING
    assert assess_risk(scene(target(12)), config).risk_level is RiskLevel.CRITICAL
    with pytest.raises(TypeError):
        config.distance_thresholds["critical"] = 8


@pytest.mark.parametrize(
    "overrides",
    [
        {"typo": 1},
        {"distance_thresholds": {"warning": 14}},
        {"distance_thresholds": {"warnign": 20}},
        {"distance_thresholds": {"notice": float("nan")}},
        {"min_confidence_for_clear": 0},
        {"min_confidence_for_clear": 1.1},
        {"min_confidence_for_clear": True},
        {"uncertainty_sigma": 10**400},
        {"oncoming_stricter_factor": 0.5},
        {"weather_risk_levels": {"fog": "unknown"}},
        {"weather_risk_multiplier": {"extra": 1}},
        {"require_handover_head": "false"},
        {"handover_level_to_action": {2: "no_action"}},
        {"handover_level_to_action": {1.5: "no_action"}},
        {"handover_level_to_action": {"1": "no_action"}},
        {"weather_risk_multiplier": []},
        {"weather_handover_enabled": "false"},
        {"weather_handover_conditions": ["night"]},
        {"weather_handover_enabled": "false"},
        {"weather_handover_conditions": ["night"]},
        [],
        False,
    ],
)
def test_invalid_config_rejected(overrides):
    with pytest.raises(ValueError):
        load_policy_config(overrides)


@pytest.mark.parametrize("text", ["", "[", "advisory: {}", "advisory: false"])
def test_invalid_yaml_is_clear_error(tmp_path, text):
    path = tmp_path / "policy.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="YAML"):
        load_policy_config(path=path)


def test_config_can_be_imported_before_advisory():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from car_smart_assist.config.policy import load_policy_config; load_policy_config()",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ("distance", "expected_risk"),
    [(0, "critical"), (20, "warning"), (40, "notice"), (60, "none"), (None, "unknown")],
)
def test_generator_outputs_rules_and_valid_wording(distance, expected_risk):
    advisory = AdvisoryGenerator().generate(perception=scene(target(distance)))
    assert advisory.source == "policy"
    assert advisory.risk_level.value == expected_risk
    assert advisory.policy_details["risk"]["risk_level"] == expected_risk
    assert advisory.evidence and not check_wording(advisory.text)
    json.dumps(advisory.to_dict(), allow_nan=False)


def test_gate_result_overrides_perception_without_mutation():
    p = scene(target())
    verdict = VisibilityVerdict(level=VisibilityLevel.DEGRADED, reason="test", information=0.4)
    a = AdvisoryGenerator().generate(visibility=verdict, perception=p)
    assert not a.should_takeover
    assert not a.policy_details["unable_to_judge"]
    assert a.risk_level is RiskLevel.NOTICE
    assert p.visibility_level == "visible" and p.visibility_confidence_multiplier == 1


def test_invalid_config_falls_back_but_never_blocks_blind():
    generator = AdvisoryGenerator({"policy": {"min_confidence_for_clear": -1}})
    a = generator.generate(perception=scene())
    assert a.should_takeover and a.source == "fallback" and a.risk_level is RiskLevel.UNKNOWN
    blind = VisibilityVerdict(level=VisibilityLevel.BLIND, reason="test", information=0)
    b = generator.generate(visibility=blind, perception=scene())
    assert b.should_takeover and b.source == "visibility_gate"


def test_visibility_alone_cannot_clear_takeover():
    visible = VisibilityVerdict(level=VisibilityLevel.VISIBLE, reason="test", information=1)
    result = AdvisoryGenerator().generate(visibility=visible)
    assert result.should_takeover and result.risk_level is RiskLevel.UNKNOWN


def test_policy_config_loaded_once(monkeypatch):
    generator = AdvisoryGenerator()
    generator.from_perception(scene())

    def unexpected_load(*args):
        raise AssertionError("每帧不应重新读配置")

    monkeypatch.setattr("car_smart_assist.advisory.generator.load_policy_config", unexpected_load)
    assert not generator.from_perception(scene()).should_takeover


def test_all_head_templates_pass_wording_check(config):
    for level in (0, 1, 2):
        decision = evaluate_handover(scene(handover_level=level, handover_confidence=1), config)
        assert not check_wording(render_policy(decision))
