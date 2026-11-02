"""Strict frame output and caller-owned nested perception isolation."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from car_smart_assist.advisory.schema import PerceptionResult, TargetDirection, TargetObject
from car_smart_assist.inference.pipeline import InferencePipeline
from car_smart_assist.perception.weather import WeatherPrediction
from test_inference_pipeline import StubScorer


def pipeline(perception, weather=None):
    return InferencePipeline(scorer=StubScorer(), predictor=SimpleNamespace(predict=lambda _: perception),
                             weather_predictor=weather)


@pytest.mark.parametrize("direction,expected", [
    (TargetDirection.LEADING, "leading"), ("oncoming", "oncoming"), ("invalid", "unknown"), (None, "unknown")])
def test_direction_serializes_without_losing_frame_advisory(direction, expected):
    original = PerceptionResult(
        objects=[TargetObject("car", direction=direction, confidence=.9, distance_m=3,
                              distance_uncertainty_m=.1)],
        object_detection_available=True, road_obstacle_detection_available=True)
    result = pipeline(original).run(np.zeros((8, 8, 3), np.uint8))
    payload = json.loads(json.dumps(result.to_dict(), allow_nan=False))
    assert payload["perception"]["objects"][0]["direction"] == expected
    assert payload["advisory"]["text"]
    assert payload["blocked"] is False
    assert original.objects[0].direction == direction


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_detection_has_conservative_advisory_and_strict_json(bad):
    original = PerceptionResult(
        objects=[TargetObject("car", confidence=bad, distance_m=bad, bbox=(0., bad, 3., 4.))],
        object_detection_available=True, road_obstacle_detection_available=True)
    result = pipeline(original).run(np.zeros((8, 8, 3), np.uint8))
    payload = json.loads(json.dumps(result.to_dict(), allow_nan=False))
    target = payload["perception"]["objects"][0]
    assert target["confidence"] is None and target["distance_m"] is None
    assert target["bbox"] == [0., None, 3., 4.]
    assert payload["advisory"]["should_takeover"] is True
    assert payload["blocked"] is False
    assert not np.isfinite(original.objects[0].confidence)


@pytest.mark.parametrize("objects", [[object()], None])
def test_invalid_target_structure_remains_serializable_and_conservative(objects):
    result = pipeline(PerceptionResult(objects=objects, object_detection_available=True)).run(
        np.zeros((8, 8, 3), np.uint8))
    payload = json.loads(json.dumps(result.to_dict(), allow_nan=False))
    assert payload["perception"]["objects"] == ([None] if isinstance(objects, list) else None)
    assert payload["advisory"]["should_takeover"] is True


def test_shared_predictor_result_keeps_each_frame_and_caller_independent():
    original = PerceptionResult(
        objects=[TargetObject("car", confidence=.9, distance_m=4)],
        object_detection_available=True, road_obstacle_detection_available=True,
        weather_attributes=("night",), weather_probabilities={"night": .8})
    readings = iter([("fog",), ("rain",)])
    def weather(_):
        attributes = next(readings)
        return WeatherPrediction(attributes, {}, {attributes[0]: .9}, {}, "synthetic weather")
    results = pipeline(original, SimpleNamespace(predict=weather)).run_batch(
        [np.zeros((8, 8, 3), np.uint8), np.ones((8, 8, 3), np.uint8)])
    first = results[0].to_dict()
    assert first["perception"]["weather_attributes"] == ["fog"]
    assert results[1].to_dict()["perception"]["weather_attributes"] == ["rain"]
    assert original.weather_attributes == ("night",)
    assert original.visibility_level == "unknown"
    results[1].perception.objects[0].distance_m = 99
    results[1].perception.visibility_reasons.append("second frame only")
    assert results[0].to_dict() == first
    assert original.objects[0].distance_m == 4
    assert original.visibility_reasons == []


@pytest.fixture(autouse=True)
def isolated_random_state():
    import random
    import numpy as np
    import torch
    state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        yield
    finally:
        random.setstate(state[0])
        np.random.set_state(state[1])
        torch.set_rng_state(state[2])
        torch.set_num_threads(threads)
