"""天气小模型配置。模型和阈值从 configs/model/weather_classifier.yaml 加载。"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class WeatherConfig:
    attributes: tuple[str, ...]
    decision_thresholds: dict[str, float]
    image_size: tuple[int, int]
    channels: tuple[int, int, int]
    dropout: float
    checkpoint: str
    device: str
    features: dict[str, float]
    train: dict[str, Any]
    rain_adapter_channels: int = 0


def load_weather_config(path: str | Path | None = None) -> WeatherConfig:
    source = (
        Path(path)
        if path
        else Path(__file__).resolve().parents[3] / "configs/model/weather_classifier.yaml"
    )
    try:
        cfg = yaml.safe_load(source.read_text(encoding="utf-8"))["weather_classifier"]
        attributes = tuple(cfg["attributes"])
        decision_thresholds = {name: float(value) for name, value in cfg["decision_thresholds"].items()}
        size = tuple(cfg["image_size"])
        channels = tuple(cfg["channels"])
        features = dict(cfg["features"])
        train = dict(cfg["train"])
        dropout = float(cfg["dropout"])
        rain_adapter_channels = cfg.get("rain_adapter_channels", 0)
    except (KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
        raise ValueError(f"天气模型配置无效：{exc}") from exc
    if attributes != ("fog", "rain", "snow", "night"):
        raise ValueError("天气属性顺序必须是 fog/rain/snow/night")
    if set(decision_thresholds) != set(attributes) or any(
        not isfinite(value) or not 0 < value < 1 for value in decision_thresholds.values()
    ):
        raise ValueError("每个天气属性都需要 0 和 1 之间的有限判断阈值")
    if len(size) != 2 or any(type(x) is not int or x <= 0 for x in size):
        raise ValueError("image_size 必须是两个正整数")
    if len(channels) != 3 or any(type(x) is not int or x <= 0 for x in channels):
        raise ValueError("channels 必须是三个正整数")
    if not isfinite(dropout) or not 0 <= dropout < 1:
        raise ValueError("dropout 必须是 [0, 1) 内的有限数")
    expected = {
        "road_top_fraction",
        "road_side_fraction",
        "bright_threshold",
        "dark_threshold",
        "saturation_max",
        "smooth_gradient_max",
    }
    if set(features) != expected or any(
        type(v) not in (float, int) or not isfinite(v) or not 0 <= v <= 1 for v in features.values()
    ):
        raise ValueError("视觉线索阈值配置无效")
    if features["road_top_fraction"] >= 1 or features["road_side_fraction"] >= 0.5:
        raise ValueError("道路近似区域必须非空")
    if not isinstance(cfg["checkpoint"], str) or not isinstance(cfg["device"], str):
        raise ValueError("checkpoint/device 必须是字符串")
    if type(rain_adapter_channels) is not int or rain_adapter_channels < 0:
        raise ValueError("rain_adapter_channels 必须是非负整数")
    return WeatherConfig(
        attributes,
        decision_thresholds,
        size,
        channels,
        dropout,
        cfg["checkpoint"],
        cfg["device"],
        features,
        train,
        rain_adapter_channels,
    )
