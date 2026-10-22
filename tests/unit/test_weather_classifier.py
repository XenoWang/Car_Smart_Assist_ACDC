"""天气属性标签、训练、推理，以及 pipeline 里的警告行为。"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest
import torch
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
    WeatherClassifier,
    WeatherPrediction,
    WeatherPredictor,
    prepare_image,
    select_device,
    visual_cues,
)
from car_smart_assist.perception.weather_training import (
    list_condition_images,
    list_pixel_accurate_images,
    train_weather,
)


def small_config(*, epochs: int = 1):
    cfg = load_weather_config()
    train = {
        **cfg.train,
        "acdc_root": "acdc",
        "cache_dir": "cache",
        "epochs": epochs,
        "batch_size": 2,
        "num_workers": 0,
        "patience": 3,
        "use_pixel_accurate": False,
    }
    return replace(
        cfg,
        image_size=(32, 48),
        channels=(4, 8, 8),
        dropout=0,
        device="cpu",
        train=train,
        checkpoint="checkpoints/weather_attributes/best.pt",
    )


@pytest.mark.parametrize("retired", ["rain_adapter_channels", "enhanced"])
def test_retired_rain_adapter_configuration_is_rejected(tmp_path, retired):
    import yaml

    original = Path(__file__).resolve().parents[2] / "configs/model/weather_classifier.yaml"
    values = yaml.safe_load(original.read_text(encoding="utf-8"))
    if retired == "rain_adapter_channels":
        values["weather_classifier"][retired] = 8
    else:
        values["weather_classifier"]["train"][retired] = {}
    path = tmp_path / "retired.yaml"
    path.write_text(yaml.safe_dump(values), encoding="utf-8")
    with pytest.raises(ValueError, match="已废弃"):
        load_weather_config(path)


def write_dataset(root: Path, cfg) -> None:
    for attr_index, condition in enumerate(cfg.attributes):
        for split, count in (("train", 2), ("val", 1)):
            sequence = f"{condition}_{split}_sequence"
            directory = root / "acdc" / "rgb_anon" / condition / split / sequence
            directory.mkdir(parents=True)
            for index in range(count):
                image = np.full((40, 60, 3), 20 + attr_index * 50 + index, dtype=np.uint8)
                Image.fromarray(image).save(
                    directory / f"{sequence}_frame_{index:06d}_rgb_anon.png"
                )


def test_acdc_attribute_source_and_reference_exclusion(tmp_path):
    cfg = small_config()
    write_dataset(tmp_path, cfg)
    refs = tmp_path / "acdc" / "rgb_anon" / "fog" / "train_ref" / "reference"
    refs.mkdir(parents=True)
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(refs / "ref_rgb_ref_anon.png")
    records = list_condition_images(tmp_path / "acdc", "train", cfg.attributes)
    assert len(records) == 8
    assert {sample.stratum for sample in records} == set(cfg.attributes)
    assert all(sample.path is not None and "_ref" not in sample.path.parts for sample in records)
    for sample in records:
        assert sample.labels[cfg.attributes.index(sample.stratum)] == 1
        assert sum(sample.labels) == 1
    with pytest.raises(ValueError):
        list_condition_images(tmp_path / "acdc", "test", cfg.attributes)


def test_pixel_accurate_filename_labels_allow_cooccurrence(tmp_path):
    archive_path = tmp_path / "rgb_left_8bit.zip"
    with ZipFile(archive_path, "w") as archive:
        for name in (
            "rgb_left_8bit/scene1_night_fog20_0.png",
            "rgb_left_8bit/scene1_night_rain55_0.png",
            "rgb_left_8bit/scene2_day_clear_0.png",
        ):
            archive.writestr(name, b"metadata-only")
    cfg = small_config()
    train, val = list_pixel_accurate_images(archive_path, cfg.attributes, validation_scene=2)
    assert len(train) == 2 and len(val) == 1
    fog = next(sample for sample in train if sample.stratum.endswith("fog20"))
    rain = next(sample for sample in train if sample.stratum.endswith("rain55"))
    assert fog.labels[cfg.attributes.index("fog")] == 1
    assert fog.labels[cfg.attributes.index("night")] == 1
    assert rain.labels[cfg.attributes.index("rain")] == 1
    assert rain.labels[cfg.attributes.index("night")] == 1
    assert not any(val[0].labels)


def test_visual_cues_are_distinct_proxies():
    cfg = small_config()
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


def test_classifier_can_predict_fog_and_night_together():
    cfg = small_config()
    model = WeatherClassifier(cfg)
    with torch.no_grad():
        model.head[-1].weight.zero_()
        model.head[-1].bias.fill_(-8)
        model.head[-1].bias[cfg.attributes.index("fog")] = 8
        model.head[-1].bias[cfg.attributes.index("night")] = 8
    prediction = WeatherPredictor(model, cfg, torch.device("cpu")).predict(
        np.zeros((32, 48, 3), dtype=np.uint8)
    )
    assert prediction.attributes == ("fog", "night")
    assert prediction.decisions["fog"] and prediction.decisions["night"]


def test_training_checkpoint_predict_and_resume(tmp_path):
    cfg = small_config()
    write_dataset(tmp_path, cfg)
    first = train_weather(cfg, tmp_path)
    assert first.best_epoch == 1
    assert first.train_count == 8 and first.val_count == 4
    assert first.pixel_train_count == 0 and first.pixel_val_count == 0
    assert first.checkpoint.is_file()
    assert (tmp_path / "checkpoints/weather_attributes/last.pt").is_file()
    predictor = WeatherPredictor.from_checkpoint(first.checkpoint, cfg)
    result = predictor.predict(np.zeros((40, 60, 3), dtype=np.uint8))
    assert set(result.decisions) == set(cfg.attributes)
    assert set(result.attributes) <= set(cfg.attributes)
    assert set(result.probabilities) == set(cfg.attributes)
    assert all(0 <= probability <= 1 for probability in result.probabilities.values())
    json.dumps(result.to_dict(), allow_nan=False)
    resumed = train_weather(replace(cfg, train={**cfg.train, "epochs": 2}), tmp_path)
    assert resumed.best_epoch in (1, 2)
    assert resumed.checkpoint.is_file()


def test_changed_data_rejects_resume(tmp_path):
    cfg = small_config()
    write_dataset(tmp_path, cfg)
    train_weather(cfg, tmp_path)
    new_image = tmp_path / "acdc/rgb_anon/fog/train/fog_train_sequence/new_rgb_anon.png"
    Image.fromarray(np.zeros((40, 60, 3), dtype=np.uint8)).save(new_image)
    with pytest.raises(ValueError, match="数据变化"):
        train_weather(replace(cfg, train={**cfg.train, "epochs": 2}), tmp_path)


def test_missing_data_cannot_train(tmp_path):
    with pytest.raises(FileNotFoundError, match="ACDC"):
        train_weather(small_config(), tmp_path)


def test_auto_device_respects_cuda_availability(monkeypatch):
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


def test_weather_warning_is_separate_from_handover_decision():
    class WeatherStub:
        def predict(self, image):
            return WeatherPrediction(
                ("fog", "night"),
                {"fog": True, "rain": False, "snow": False, "night": True},
                {"fog": 0.9, "rain": 0.1, "snow": 0.02, "night": 0.88},
                dict.fromkeys(FEATURE_NAMES, 0.2),
                "synthetic co-occurrence",
            )

    class PerceptionStub:
        def predict(self, image):
            return PerceptionResult(object_detection_available=True)

    pipeline = weather_pipeline(WeatherStub())
    pipeline.predictor = PerceptionStub()
    result = pipeline.run(np.zeros((32, 48, 3), dtype=np.uint8))
    assert result.weather.attributes == ("fog", "night")
    assert result.weather_warning == "夜间有雾，视线可能受影响，请减速"
    assert result.perception.weather_attributes == ("fog", "night")
    assert result.advisory.should_takeover is False
    assert result.advisory.risk_level is RiskLevel.NOTICE


def test_pipeline_uses_weather_only_as_auxiliary_with_degraded_visibility():
    class WeatherStub:
        def predict(self, image):
            return WeatherPrediction(
                ("rain",),
                {"fog": False, "rain": True, "snow": False, "night": False},
                {"fog": 0.08, "rain": 0.9, "snow": 0.03, "night": 0.12},
                dict.fromkeys(FEATURE_NAMES, 0.3),
                "synthetic rain attribute",
            )

    class PerceptionStub:
        def predict(self, image):
            return PerceptionResult(object_detection_available=True)

    pipeline = weather_pipeline(WeatherStub(), information=0.4)
    pipeline.predictor = PerceptionStub()
    result = pipeline.run(np.zeros((32, 48, 3), dtype=np.uint8))
    assert result.visibility.level.value == "degraded"
    assert result.perception.weather_attributes == ("rain",)
    assert result.weather_warning is not None
    assert result.advisory.should_takeover is True
    assert result.advisory.policy_details["unable_to_judge"] is False
    assert (
        "weather_visibility_auxiliary_handover"
        in result.advisory.policy_details["risk"]["triggered"]
    )


def test_adverse_weather_and_degraded_visibility_jointly_request_handover():
    class WeatherStub:
        def predict(self, image):
            return WeatherPrediction(
                ("rain",),
                {"fog": False, "rain": True, "snow": False, "night": False},
                {"fog": 0.08, "rain": 0.9, "snow": 0.03, "night": 0.12},
                dict.fromkeys(FEATURE_NAMES, 0.3),
                "synthetic rain attribute",
            )

    class PerceptionStub:
        def predict(self, image):
            return PerceptionResult(object_detection_available=True)

    pipeline = weather_pipeline(WeatherStub(), information=0.4)
    pipeline.predictor = PerceptionStub()
    result = pipeline.run(np.zeros((32, 48, 3), dtype=np.uint8))
    assert result.visibility.level.value == "degraded"
    assert result.weather_warning is not None
    assert result.advisory.should_takeover is True
    assert result.advisory.policy_details["unable_to_judge"] is False
    assert (
        "weather_visibility_auxiliary_handover"
        in result.advisory.policy_details["risk"]["triggered"]
    )


def test_missing_perception_still_requests_takeover_independently_of_weather():
    class WeatherStub:
        def predict(self, image):
            return WeatherPrediction(
                ("snow",),
                {"fog": False, "rain": False, "snow": True, "night": False},
                {"fog": 0.03, "rain": 0.04, "snow": 0.9, "night": 0.03},
                dict.fromkeys(FEATURE_NAMES, 0.3),
                "synthetic",
            )

    result = weather_pipeline(WeatherStub()).run(np.zeros((32, 48, 3), dtype=np.uint8))
    assert result.weather_warning == "检测到降雪或积雪，请减速并留足制动距离"
    assert result.advisory.should_takeover
    assert result.advisory.risk_level is RiskLevel.UNKNOWN
    assert "perception" in result.skipped


def test_weather_uncertainty_does_not_trigger_takeover_with_complete_perception():
    class WeatherStub:
        def predict(self, image):
            return WeatherPrediction(
                (),
                dict.fromkeys(("fog", "rain", "snow", "night"), False),
                dict.fromkeys(("fog", "rain", "snow", "night"), 0.25),
                dict.fromkeys(FEATURE_NAMES, 0.0),
                "no weather attribute exceeded its threshold",
            )

    class PerceptionStub:
        def predict(self, image):
            return PerceptionResult(
                object_detection_available=True,
                objects=[TargetObject("car", TargetDirection.LEADING, 60, confidence=1.0)],
            )

    pipeline = weather_pipeline(WeatherStub())
    pipeline.predictor = PerceptionStub()
    result = pipeline.run(np.zeros((32, 48, 3), dtype=np.uint8))
    assert result.weather_warning is None
    assert result.advisory.risk_level is RiskLevel.NONE
    assert result.advisory.should_takeover is False


def test_blind_gate_skips_weather_model():
    class WeatherStub:
        def predict(self, image):
            raise AssertionError("BLIND frame must not reach weather classifier")

    result = weather_pipeline(WeatherStub(), information=0.01).run(
        np.zeros((32, 48, 3), dtype=np.uint8)
    )
    assert result.blocked and result.weather is None
    assert result.weather_warning is None
    assert result.advisory.should_takeover is True
    assert result.advisory.source == "visibility_gate"
def test_attention_training_output_guard_rejects_default_checkpoint_directory(tmp_path):
    import yaml
    from car_smart_assist.perception.weather_training import require_independent_weather_output

    config = tmp_path / "configs/model/weather_classifier.yaml"
    config.parent.mkdir(parents=True)
    values = yaml.safe_load((Path(__file__).resolve().parents[2]
                             / "configs/model/weather_classifier.yaml").read_text(encoding="utf-8"))
    values["weather_classifier"]["checkpoint"] = "artifacts/checkpoints/weather/default/best.pt"
    config.write_text(yaml.safe_dump(values), encoding="utf-8")
    protected = (tmp_path / values["weather_classifier"]["checkpoint"]).parent
    protected.mkdir(parents=True)
    best, last = protected / "best.pt", protected / "last.pt"
    best.write_bytes(b"default best")
    last.write_bytes(b"default last")
    before = (best.read_bytes(), last.read_bytes())
    with pytest.raises(ValueError, match="保护"):
        require_independent_weather_output(tmp_path, protected)
    assert (best.read_bytes(), last.read_bytes()) == before


def test_weather_fresh_archive_preserves_best_and_last_bytes(tmp_path):
    from car_smart_assist.perception.weather_training import _archive_weather_checkpoints

    best, last = tmp_path / "best.pt", tmp_path / "last.pt"
    best.write_bytes(b"old-best")
    last.write_bytes(b"old-last")
    _archive_weather_checkpoints((best, last))
    assert not best.exists() and not last.exists()
    snapshots = list((tmp_path / "history").glob("*/best.pt"))
    assert len(snapshots) == 1
    archive = snapshots[0].parent
    assert snapshots[0].read_bytes() == b"old-best"
    assert (archive / "last.pt").read_bytes() == b"old-last"


def test_weather_fresh_archive_failure_keeps_original_checkpoints(tmp_path, monkeypatch):
    from car_smart_assist.perception import weather_training

    best, last = tmp_path / "best.pt", tmp_path / "last.pt"
    best.write_bytes(b"old-best")
    last.write_bytes(b"old-last")
    monkeypatch.setattr(
        weather_training.os, "fsync",
        lambda _fd: (_ for _ in ()).throw(OSError("disk")),
    )
    with pytest.raises(OSError, match="disk"):
        weather_training._archive_weather_checkpoints((best, last))
    assert best.read_bytes() == b"old-best"
    assert last.read_bytes() == b"old-last"


def test_weather_last_rejects_missing_or_replaced_best_identity(tmp_path):
    import hashlib
    from car_smart_assist.perception.weather_training import _load_verified_weather_best

    best = tmp_path / "best.pt"
    best.write_bytes(b"best payload")
    reference = {
        "best_sha256": hashlib.sha256(best.read_bytes()).hexdigest(),
        "format_version": 2, "model_config": {"width": 4},
        "dataset_signatures": {"train": "a", "val": "b"},
        "best_epoch": 2, "best_val_loss": 0.25, "best_val_metrics": {"f1": 0.7},
    }
    with pytest.raises(ValueError, match="完整恢复"):
        _load_verified_weather_best(tmp_path / "missing.pt", reference)
    best.write_bytes(b"unrelated best payload")
    with pytest.raises(ValueError, match="摘要不配对"):
        _load_verified_weather_best(best, reference)
