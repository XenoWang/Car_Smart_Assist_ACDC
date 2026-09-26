"""四类天气小模型的数据口径、训练续训和管线接入测试。"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from car_smart_assist.advisory.schema import (
    PerceptionResult,
    RiskLevel,
    TargetDirection,
    TargetObject,
)
from car_smart_assist.config.weather import load_weather_config
from car_smart_assist.inference.pipeline import InferencePipeline
from car_smart_assist.perception.visibility.gate import GateThresholds, VisibilityGate
from car_smart_assist.perception.visibility.scorer import InformationFeatures, VisibilityScore
from car_smart_assist.perception.weather import (
    FEATURE_NAMES,
    WeatherPrediction,
    WeatherPredictor,
    prepare_image,
    select_device,
    visual_cues,
)
from car_smart_assist.perception.weather_training import list_condition_images, train_weather


def small_config(root: Path, *, epochs: int = 1):
    cfg = load_weather_config()
    train = {
        **cfg.train,
        "acdc_root": "acdc",
        "cache_dir": "cache",
        "epochs": epochs,
        "batch_size": 2,
        "num_workers": 0,
        "patience": 3,
    }
    return replace(
        cfg,
        image_size=(32, 48),
        channels=(4, 8, 8),
        dropout=0,
        device="cpu",
        min_confidence=1.0,
        train=train,
        checkpoint="checkpoints/weather/best.pt",
    )


def write_dataset(root: Path, cfg) -> None:
    for class_index, condition in enumerate(cfg.classes):
        for split, count in (("train", 2), ("val", 1)):
            sequence = f"{condition}_{split}_sequence"
            directory = root / "acdc" / "rgb_anon" / condition / split / sequence
            directory.mkdir(parents=True)
            for index in range(count):
                image = np.full((40, 60, 3), 20 + class_index * 50 + index, dtype=np.uint8)
                Image.fromarray(image).save(
                    directory / f"{sequence}_frame_{index:06d}_rgb_anon.png"
                )


def test_acdc_class_source_and_reference_exclusion(tmp_path):
    cfg = small_config(tmp_path)
    write_dataset(tmp_path, cfg)
    refs = tmp_path / "acdc" / "rgb_anon" / "fog" / "train_ref" / "reference"
    refs.mkdir(parents=True)
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(refs / "ref_rgb_ref_anon.png")
    records = list_condition_images(tmp_path / "acdc", "train", cfg.classes)
    assert len(records) == 8
    assert {label for _, label in records} == {0, 1, 2, 3}
    assert all("_ref" not in p.parts for p, _ in records)
    with pytest.raises(ValueError):
        list_condition_images(tmp_path / "acdc", "test", cfg.classes)


def test_visual_cues_are_distinct_proxies(tmp_path):
    cfg = small_config(tmp_path)
    white = np.full((32, 48, 3), 235, dtype=np.uint8)
    white_cues = visual_cues(white, cfg)
    assert white_cues["snow_coverage_proxy"] > 0.9
    assert white_cues["reflection_proxy"] < 0.1
    stripes = white.copy()
    stripes[20:, ::2] = 5
    stripe_cues = visual_cues(stripes, cfg)
    assert stripe_cues["reflection_proxy"] > white_cues["reflection_proxy"]
    assert set(stripe_cues) == set(FEATURE_NAMES)
    assert all(0 <= value <= 1 for value in stripe_cues.values())
    assert prepare_image(np.zeros((40, 60, 3), dtype=np.uint8), cfg.image_size).shape == (32, 48, 3)


def test_training_checkpoint_predict_and_resume(tmp_path):
    cfg = small_config(tmp_path)
    write_dataset(tmp_path, cfg)
    first = train_weather(cfg, tmp_path)
    assert first.best_epoch == 1
    assert first.train_count == 8 and first.val_count == 4
    assert first.checkpoint.is_file()
    assert (tmp_path / "checkpoints/weather/last.pt").is_file()
    predictor = WeatherPredictor.from_checkpoint(first.checkpoint, cfg)
    result = predictor.predict(np.zeros((40, 60, 3), dtype=np.uint8))
    assert result.condition is None and not result.accepted  # 人工小样本不能当作可信天气分类器
    assert len(result.probabilities) == 4
    assert sum(result.probabilities.values()) == pytest.approx(1.0)
    json.dumps(result.to_dict(), allow_nan=False)
    resumed = train_weather(replace(cfg, train={**cfg.train, "epochs": 2}), tmp_path)
    assert resumed.best_epoch in (1, 2)
    assert resumed.checkpoint.is_file()


def test_changed_data_rejects_resume(tmp_path):
    cfg = small_config(tmp_path)
    write_dataset(tmp_path, cfg)
    train_weather(cfg, tmp_path)
    new_image = tmp_path / "acdc/rgb_anon/fog/train/fog_train_sequence/new_rgb_anon.png"
    Image.fromarray(np.zeros((40, 60, 3), dtype=np.uint8)).save(new_image)
    with pytest.raises(ValueError, match="数据变化"):
        train_weather(replace(cfg, train={**cfg.train, "epochs": 2}), tmp_path)


def test_missing_data_cannot_train(tmp_path):
    with pytest.raises(FileNotFoundError, match="ACDC"):
        train_weather(small_config(tmp_path), tmp_path)


def test_auto_device_respects_cuda_availability(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert select_device("auto").type == "cpu"
    with pytest.raises(ValueError, match="不可用"):
        select_device("cuda")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert select_device("auto").type == "cuda"


def weather_pipeline(stub, information=0.95):
    class Scorer:
        def score_arrays(self, images, paths=None):
            return [
                VisibilityScore(
                    path="<weather-test>",
                    recon_mean=0.01,
                    recon_p90_block=0.01,
                    recon_z=0.0,
                    features=InformationFeatures(1, 1, 1, 1),
                    information=information,
                )
                for _ in images
            ]

    return InferencePipeline(
        scorer=Scorer(),
        gate=VisibilityGate(GateThresholds(), require_calibration=False),
        weather_predictor=stub,
    )


def test_weather_alone_reaches_risk_but_requests_takeover():
    class WeatherStub:
        def predict(self, image):
            return WeatherPrediction(
                "snow",
                0.9,
                {"fog": 0.03, "night": 0.03, "rain": 0.04, "snow": 0.9},
                dict.fromkeys(FEATURE_NAMES, 0.3),
                True,
                "synthetic",
            )

    pipeline = weather_pipeline(WeatherStub())
    result = pipeline.run(np.zeros((32, 48, 3), dtype=np.uint8))
    assert result.weather.condition == "snow"
    assert result.perception.road_condition == "snow"
    assert result.perception.object_detection_available is False
    assert result.advisory.risk_level is RiskLevel.NOTICE
    assert result.advisory.should_takeover
    assert result.advisory.policy_details["unable_to_judge"]
    assert "perception" in result.skipped


def test_uncertain_weather_alone_is_unknown():
    class WeatherStub:
        def predict(self, image):
            return WeatherPrediction(
                None,
                0.4,
                dict.fromkeys(("fog", "night", "rain", "snow"), 0.25),
                dict.fromkeys(FEATURE_NAMES, 0.0),
                False,
                "uncertain",
            )

    result = weather_pipeline(WeatherStub()).run(np.zeros((32, 48, 3), dtype=np.uint8))
    assert result.advisory.risk_level is RiskLevel.UNKNOWN
    assert result.advisory.should_takeover
    assert not any("天气分类置信度不足" in reason for reason in result.advisory.evidence)


def test_uncertain_weather_does_not_trigger_takeover_when_other_inputs_are_complete():
    class WeatherStub:
        def predict(self, image):
            return WeatherPrediction(None, 0.4, dict.fromkeys(("fog", "night", "rain", "snow"), 0.25),
                                     dict.fromkeys(FEATURE_NAMES, 0.0), False, "uncertain")

    class PerceptionStub:
        def predict(self, image):
            return PerceptionResult(
                road_condition=None, road_condition_confidence=0.0,
                object_detection_available=True,
                objects=[TargetObject("car", TargetDirection.LEADING, 60, confidence=1.0)],
            )

    result = weather_pipeline(WeatherStub())
    result.predictor = PerceptionStub()
    advisory = result.run(np.zeros((32, 48, 3), dtype=np.uint8)).advisory
    assert advisory.risk_level is RiskLevel.NONE
    assert advisory.should_takeover is False
    assert any("天气类别未能确认" in reason for reason in advisory.evidence)


def test_blind_gate_skips_weather_model():
    class WeatherStub:
        def predict(self, image):
            raise AssertionError("BLIND frame must not reach weather classifier")

    result = weather_pipeline(WeatherStub(), information=0.01).run(
        np.zeros((32, 48, 3), dtype=np.uint8)
    )
    assert result.blocked and result.weather is None
    assert result.advisory.should_takeover is True
    assert result.advisory.source == "visibility_gate"
