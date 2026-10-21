from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


BRANCHES = ("attention", "pyramid", "legacy")
DEVICES = ("cpu", "auto", "cuda", "cuda:2")


def _write(path: Path, text: str = "") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _build_mocked_pipeline(monkeypatch, tmp_path, branch, explicit_device):
    from car_smart_assist.inference import pipeline as pipeline_module
    from car_smart_assist.config import detection as detection_config
    from car_smart_assist.config import weather_attention as attention_config
    from car_smart_assist.config import weather_pyramid as pyramid_config
    from car_smart_assist.perception import weather_attention, weather_pyramid
    from car_smart_assist.perception.visibility import trainer as visibility_trainer
    from car_smart_assist.perception.weather import WeatherPredictor

    root = tmp_path
    visibility_checkpoint = _write(root / "artifacts/checkpoints/visibility/best.pt")
    yolo_checkpoint = _write(root / "artifacts/runs/detection/yolo/weights/best.pt")
    config_dir = root / "configs/model"
    yolo_yaml = _write(config_dir / "yolo_detection.yaml", "test: true\\n")
    original_yolo_yaml = yolo_yaml.read_bytes()
    _write(config_dir / "yolo_acdc_baseline.yaml", "test: true\n")
    _write(config_dir / "weather_classifier.yaml", "weather_classifier: {}\\n")

    original_yolo_cfg = {
        "device": "auto",
        "checkpoint": "artifacts/runs/detection/yolo/weights/best.pt",
        "enabled_for_inference": True,
    }
    monkeypatch.setattr(detection_config, "load_yolo_config", lambda _path: original_yolo_cfg)

    selected = "cpu" if explicit_device == "auto" else ("cuda:2" if explicit_device == "cuda" else explicit_device)
    expected_visibility = "cuda:1" if explicit_device is None else selected
    expected_yolo = "auto" if explicit_device is None else selected
    monkeypatch.setattr(pipeline_module, "select_device", lambda request: torch.device(
        "cpu" if request == "auto" else request
    ))
    monkeypatch.setattr(visibility_trainer, "resolve_device", lambda request: request)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 2)

    calls = {"visibility": [], "weather": [], "yolo": []}
    fake_scorer = SimpleNamespace(input_size=(16, 16), model=object())
    monkeypatch.setattr(
        pipeline_module.VisibilityScorer,
        "from_checkpoint",
        staticmethod(lambda path, **kwargs: calls["visibility"].append((path, kwargs)) or fake_scorer),
    )

    if branch == "attention":
        weather_config_path = _write(config_dir / "weather_attention_a.yaml", "weather_attention: {}\n")
        weather_cfg = SimpleNamespace(checkpoint_root="artifacts/checkpoints/weather_attention_ablation/reduced_tuning", device="cuda:1")
        monkeypatch.setattr(pipeline_module, "load_weather_config", lambda _path: pytest.fail("wrong weather backend"))
        monkeypatch.setattr(
            attention_config, "load_weather_attention_config", lambda _path: weather_cfg
        )
        monkeypatch.setattr(
            weather_attention.WeatherAttentionPredictor,
            "from_checkpoint",
            classmethod(lambda cls, path, cfg, device=None: calls["weather"].append((path, cfg, device)) or object()),
        )
        weather_checkpoint = _write(root / "artifacts/checkpoints/weather_attention_ablation/reduced_tuning/best.pt")
    elif branch == "pyramid":
        weather_config_path = _write(config_dir / "weather_pyramid.yaml", "weather_pyramid: {}\n")
        weather_cfg = SimpleNamespace(checkpoint_root="artifacts/checkpoints/weather_pyramid", device="cuda:1")
        monkeypatch.setattr(pipeline_module, "load_weather_config", lambda _path: pytest.fail("wrong weather backend"))
        monkeypatch.setattr(
            pyramid_config, "load_weather_pyramid_config", lambda _path: weather_cfg
        )
        monkeypatch.setattr(
            weather_pyramid.WeatherPyramidPredictor,
            "from_checkpoint",
            classmethod(lambda cls, path, cfg, device=None: calls["weather"].append((path, cfg, device)) or object()),
        )
        weather_checkpoint = _write(root / "artifacts/checkpoints/weather_pyramid/focal/best.pt")
    else:
        weather_config_path = None
        weather_checkpoint = _write(root / "artifacts/checkpoints/weather_attributes/best.pt")
        weather_cfg = SimpleNamespace(checkpoint=str(weather_checkpoint.relative_to(root)), device="cuda:1")
        monkeypatch.setattr(pipeline_module, "load_weather_config", lambda _path: weather_cfg)
        monkeypatch.setattr(
            WeatherPredictor,
            "from_checkpoint",
            classmethod(lambda cls, path, cfg, device=None: calls["weather"].append((path, cfg, device)) or object()),
        )

    from car_smart_assist.perception.detection_yolo import YoloDetectionPredictor
    monkeypatch.setattr(
        YoloDetectionPredictor,
        "from_config",
        classmethod(lambda cls, cfg, project_root: calls["yolo"].append((cfg, project_root)) or object()),
    )

    cfg = {
        "device": "cuda:1",
        "model": {"input_size": [16, 16]},
        "train": {"checkpoint_dir": "artifacts/checkpoints/visibility"},
        "scoring": {},
        "thresholds": {},
    }
    original_cfg = {key: value.copy() if isinstance(value, dict) else value for key, value in cfg.items()}
    kwargs = {
        "visibility_cfg": cfg,
        "project_root": root,
        "device": explicit_device,
        "require_visibility": True,
    }
    if weather_config_path is not None:
        kwargs["weather_config"] = weather_config_path
        kwargs["weather_checkpoint"] = weather_checkpoint
    pipe = pipeline_module.InferencePipeline.from_config(**kwargs)

    assert cfg == original_cfg
    assert yolo_yaml.read_bytes() == original_yolo_yaml
    assert visibility_checkpoint.is_file()
    assert yolo_checkpoint.is_file()
    assert calls["visibility"][0][1]["device"] == expected_visibility
    assert calls["yolo"]
    if explicit_device is None:
        assert calls["yolo"][0][0] is original_yolo_cfg
    else:
        assert calls["yolo"][0][0] is not original_yolo_cfg
    assert original_yolo_cfg["device"] == "auto"
    assert calls["yolo"][0][0]["device"] == expected_yolo
    assert calls["weather"]
    return pipe, calls, weather_cfg, original_yolo_cfg


@pytest.mark.parametrize("branch", BRANCHES)
@pytest.mark.parametrize("device", DEVICES)
def test_explicit_device_reaches_all_pipeline_model_builders_without_config_mutation(
    monkeypatch, tmp_path, branch, device
):
    pipe, calls, weather_cfg, original_yolo_cfg = _build_mocked_pipeline(
        monkeypatch, tmp_path, branch, device
    )
    expected = "cpu" if device == "auto" else ("cuda:2" if device == "cuda" else device)
    assert pipe.scorer is not None
    assert calls["weather"][0][2] == expected
    assert calls["yolo"][0][0]["device"] == expected
    assert original_yolo_cfg["device"] == "auto"


@pytest.mark.parametrize("branch", BRANCHES)
def test_none_keeps_each_builders_configured_device(
    monkeypatch, tmp_path, branch
):
    pipe, calls, weather_cfg, original_yolo_cfg = _build_mocked_pipeline(
        monkeypatch, tmp_path, branch, None
    )
    assert calls["visibility"][0][1]["device"] == "cuda:1"
    assert calls["weather"][0][2] is None
    assert weather_cfg.device == "cuda:1"
    assert calls["yolo"][0][0] is original_yolo_cfg
    assert original_yolo_cfg["device"] == "auto"


def test_invalid_explicit_device_fails_before_any_model_builder(monkeypatch, tmp_path):
    from car_smart_assist.inference.pipeline import InferencePipeline, VisibilityScorer

    def forbidden(*_args, **_kwargs):
        raise AssertionError("model construction must not start for an invalid device")

    monkeypatch.setattr(VisibilityScorer, "from_checkpoint", staticmethod(forbidden))
    cfg = {"device": "auto", "model": {}, "train": {"checkpoint_dir": "missing"}}
    for device in ("mps", "cuda:99"):
        with pytest.raises(ValueError):
            InferencePipeline.from_config(cfg, project_root=tmp_path, device=device)


def test_legacy_weather_checkpoint_factory_accepts_original_two_argument_call():
    from car_smart_assist.perception.weather import WeatherPredictor

    inspect.signature(WeatherPredictor.from_checkpoint).bind(Path("checkpoint.pt"), object())


def test_injected_predictor_is_retained_by_identity_without_device_migration(monkeypatch, tmp_path):
    class CallerOwnedPredictor:
        def to(self, *_args, **_kwargs):
            raise AssertionError("pipeline must not migrate an injected predictor")

    from car_smart_assist.inference.pipeline import InferencePipeline, VisibilityScorer
    from car_smart_assist.perception.visibility import trainer as visibility_trainer

    monkeypatch.setattr(visibility_trainer, "resolve_device", lambda request: request)
    scorer = SimpleNamespace(input_size=(16, 16), model=object())
    monkeypatch.setattr(VisibilityScorer, "from_checkpoint", staticmethod(lambda *_a, **_k: scorer))
    checkpoint = _write(tmp_path / "artifacts/checkpoints/visibility/best.pt")
    injected = CallerOwnedPredictor()
    cfg = {"device": "auto", "model": {}, "train": {"checkpoint_dir": str(checkpoint.parent)}}
    pipe = InferencePipeline.from_config(cfg, project_root=tmp_path, device="cpu", predictor=injected)
    assert pipe.predictor is injected