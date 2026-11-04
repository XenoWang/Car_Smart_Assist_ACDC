"""Independent weather JSON, failure isolation and per-frame ownership contracts."""
import json
import math
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from car_smart_assist.advisory.schema import PerceptionResult
from car_smart_assist.inference.pipeline import InferencePipeline
from car_smart_assist.perception.weather import WeatherPrediction
from test_inference_pipeline import StubScorer


def prediction(**changes):
    base = WeatherPrediction(
        ("fog",), {"fog": True, "rain": False}, {"fog": 0.9, "rain": 0.1},
        {"low_visibility_proxy": 0.25}, "controlled weather",
    )
    return replace(base, **changes)


def pipeline(weather, information=0.9):
    return InferencePipeline(
        scorer=StubScorer(information=information),
        predictor=SimpleNamespace(predict=lambda _: PerceptionResult(
            object_detection_available=True, road_obstacle_detection_available=True,
        )),
        weather_predictor=SimpleNamespace(predict=weather),
    )


def strict_payload(result):
    return json.loads(json.dumps(result.to_dict(), ensure_ascii=False, allow_nan=False))


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_weather_keeps_raw_policy_evidence_but_serializes_null(bad):
    original = prediction(probabilities={"fog": bad, "rain": 0.1},
                          cues={"low_visibility_proxy": bad, "reflection_proxy": 0.25})
    result = pipeline(lambda _: original, information=0.4).run(np.zeros((8, 8, 3), np.uint8))
    payload = strict_payload(result)
    assert payload["weather"]["probabilities"] == {"fog": None, "rain": 0.1}
    assert payload["weather"]["cues"] == {"low_visibility_proxy": None, "reflection_proxy": 0.25}
    assert payload["perception"]["weather_probabilities"] == {"fog": None, "rain": 0.1}
    assert not math.isfinite(original.probabilities["fog"])
    assert not math.isfinite(result.weather.probabilities["fog"])
    assert not math.isfinite(result.perception.weather_probabilities["fog"])
    assert payload["weather"]["decisions"] == {"fog": True, "rain": False}
    assert payload["blocked"] is False
    assert "weather" not in payload["skipped"]
    # Invalid probabilities must not become confirmed weather + DEGRADED takeover.
    assert payload["advisory"]["should_takeover"] is False
    assert "weather_visibility_auxiliary_handover" not in (
        result.advisory.policy_details["risk"]["triggered"]
    )


@pytest.mark.parametrize("attributes,decisions,probabilities,cues,warning", [
    ((), {}, {}, {}, None),
    (("fog",), {"fog": True}, {"fog": 0.9}, {"low_visibility_proxy": 0.25},
     "检测到雾，视线可能受影响，请减速"),
])
def test_valid_weather_preserves_literal_public_fields(attributes, decisions, probabilities, cues, warning):
    original = WeatherPrediction(attributes, decisions, probabilities, cues, "controlled")
    result = pipeline(lambda _: original).run(np.zeros((8, 8, 3), np.uint8))
    assert strict_payload(result)["weather"] == {
        "attributes": list(attributes), "decisions": decisions, "probabilities": probabilities,
        "cues": cues, "warning": warning, "reason": "controlled",
    }
    assert result.weather is not original
    assert result.weather_warning == warning
    assert result.advisory.should_takeover is False


@pytest.mark.parametrize("field,bad", [
    ("attributes", None), ("attributes", "fog"), ("attributes", ([],)),
    ("decisions", None), ("decisions", {"fog": "false"}),
    ("probabilities", None), ("probabilities", {"fog": {"nested": 0.9}}),
    ("probabilities", {"fog": True}), ("probabilities", {"fog": 10 ** 400}), ("probabilities", {1: 0.9}),
    ("cues", []), ("cues", {"proxy": object()}), ("reason", None),
])
def test_malformed_weather_is_rejected_atomically_and_next_frame_continues(field, bad):
    invalid = prediction(**{field: bad})
    valid = prediction()
    readings = iter([invalid, valid])
    results = pipeline(lambda _: next(readings)).run_batch(
        [np.zeros((8, 8, 3), np.uint8), np.ones((8, 8, 3), np.uint8)]
    )
    first, second = map(strict_payload, results)
    assert first["weather"] is None and first["weather_warning"] is None
    assert field in first["skipped"]["weather"]
    assert first["perception"]["weather_attributes"] == []
    assert first["perception"]["weather_probabilities"] == {}
    assert first["perception"]["object_detection_available"] is True
    assert first["blocked"] is False and first["advisory"]["should_takeover"] is False
    assert second["weather"]["probabilities"] == {"fog": 0.9, "rain": 0.1}
    assert second["weather"]["attributes"] == ["fog"]
    assert "weather" not in second["skipped"]
    assert getattr(invalid, field) is bad


def test_shared_weather_dictionaries_do_not_rewrite_previous_frames_or_caller():
    original = prediction()
    calls = 0

    def predict(_):
        nonlocal calls
        calls += 1
        if calls == 2:
            original.probabilities["fog"] = 0.8
            original.cues["low_visibility_proxy"] = 0.5
            original.decisions["rain"] = True
        return original

    first, second = pipeline(predict).run_batch(
        [np.zeros((8, 8, 3), np.uint8), np.ones((8, 8, 3), np.uint8)]
    )
    assert first.weather.probabilities == {"fog": 0.9, "rain": 0.1}
    assert first.perception.weather_probabilities == {"fog": 0.9, "rain": 0.1}
    assert first.weather.decisions == {"fog": True, "rain": False}
    assert first.weather.cues == {"low_visibility_proxy": 0.25}
    assert second.weather.probabilities == {"fog": 0.8, "rain": 0.1}
    assert second.perception.weather_probabilities == {"fog": 0.8, "rain": 0.1}
    assert second.weather.cues == {"low_visibility_proxy": 0.5}
    assert original.probabilities == {"fog": 0.8, "rain": 0.1}
    first_payload = strict_payload(first)  # Only a mutation-invariance baseline; values checked above.
    original.probabilities["fog"] = 0.1
    original.decisions.clear()
    second.weather.probabilities["fog"] = 0.2
    second.weather.cues.clear()
    second.perception.weather_probabilities["rain"] = 0.7
    assert strict_payload(first) == first_payload
    assert second.perception.weather_probabilities["fog"] == 0.8
    assert original.cues == {"low_visibility_proxy": 0.5}


def test_mutable_attribute_list_and_exported_dictionary_are_detached():
    attributes = ["fog"]
    original = prediction(attributes=attributes)
    result = pipeline(lambda _: original).run(np.zeros((8, 8, 3), np.uint8))
    attributes[:] = ["rain"]
    exported = result.weather.to_dict()
    exported["probabilities"]["fog"] = 0.2
    exported["attributes"].append("snow")
    assert result.weather.attributes == ("fog",)
    assert result.perception.weather_attributes == ("fog",)
    assert result.weather.probabilities == {"fog": 0.9, "rain": 0.1}
    assert original.attributes == ["rain"]
    assert result.weather_warning == "检测到雾，视线可能受影响，请减速"
