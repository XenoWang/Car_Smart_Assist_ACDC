from __future__ import annotations

from pathlib import Path

import yaml

from car_smart_assist.config.detection import load_yolo_config
from car_smart_assist.config.weather_attention import load_weather_attention_config


ROOT = Path(__file__).resolve().parents[2]


def test_weather_retrain_config_uses_a85_recipe_with_fresh_isolated_outputs():
    cfg = load_weather_attention_config(ROOT / "configs/model/weather_attention_retrain_300.yaml")
    stopping = cfg.train["early_stopping"]

    assert cfg.data["seed"] == 42
    assert cfg.device == "auto"
    assert cfg.train["epochs"] == 300
    assert cfg.train["schedule_epochs"] == 500
    assert cfg.backbone["fine_tune_from"] == 13
    assert "asl" in cfg.loss
    assert stopping == {"enabled": True, "patience": 40, "min_delta": 0.0001}
    assert cfg.checkpoint_root == f"artifacts/checkpoints/retrain_300_20261017/weather"
    assert cfg.report_dir == f"artifacts/reports/retrain_300_20261017/weather"
    assert cfg.report_file.startswith(f"artifacts/reports/retrain_300_20261017/weather/")
    assert cfg.train.get("continuation") is None


def test_legacy_weather_config_keeps_early_stopping_disabled_by_default():
    cfg = load_weather_attention_config(ROOT / "configs/model/weather_attention_a.yaml")
    stopping = cfg.train.get("early_stopping", {"enabled": False})
    assert stopping.get("enabled", False) is False


def test_visibility_retrain_config_has_fixed_four_way_split_and_isolated_report():
    raw = yaml.safe_load(
        (ROOT / "configs/model/visibility_retrain_300.yaml").read_text(encoding="utf-8")
    )["visibility"]
    data, train = raw["data"], raw["train"]

    assert raw["device"] == "auto"
    assert data["split_mode"] == "fixed"
    assert data["split_seed"] == 42
    assert data["split_by_sequence"] is True
    assert sum(
        data[key]
        for key in ("train_ratio", "val_ratio", "calib_ratio", "test_ratio")
    ) == 1.0
    assert train["epochs"] == 300
    assert train["seed"] == 42
    assert train["early_stopping"] == {"enabled": True, "patience": 8, "min_delta": 1e-5}
    assert train["checkpoint_dir"] == f"artifacts/checkpoints/retrain_300_20261017/visibility"
    assert raw["report"]["output_dir"] == f"artifacts/reports/retrain_300_20261017/visibility"
    assert data["cache_path"].startswith("artifacts/cache/retrain_300_20261017/visibility/")


def test_yolo_retrain_config_is_fresh_seeded_and_separate_from_default():
    cfg = load_yolo_config(ROOT / "configs/model/yolo_retrain_300.yaml")

    assert cfg["train"]["epochs"] == 300
    assert cfg["train"]["seed"] == 42
    assert cfg["device"] == "auto"
    assert cfg["train"]["patience"] == 30
    assert cfg["train"]["resume_extra_epochs"] is None
    assert cfg["model"] == "artifacts/checkpoints/yolo_pretrained/yolo11n.pt"
    assert cfg["project"] == "artifacts/runs/detection"
    assert cfg["name"] == "retrain_300_20261017"
    assert cfg["enabled_for_inference"] is True
    assert cfg["calibration_file"].startswith("artifacts/reports/retrain_300_20261017/detection/")

    default = load_yolo_config(ROOT / "configs/model/yolo_detection.yaml")
    assert default["checkpoint"] == "artifacts/runs/detection/yolo11n_acdc_laf_v2/weights/best.pt"
    assert default["checkpoint"] != cfg["checkpoint"]

