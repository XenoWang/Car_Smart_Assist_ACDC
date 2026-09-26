"""天气小模型配置。模型和阈值从 configs/model/weather_classifier.yaml 加载。"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class WeatherConfig:
    classes: tuple[str, ...]
    image_size: tuple[int, int]
    channels: tuple[int, int, int]
    dropout: float
    checkpoint: str
    device: str
    min_confidence: float
    min_margin: float
    features: dict[str, float]
    train: dict[str, Any]


def load_weather_config(path: str | Path | None = None) -> WeatherConfig:
    source = (
        Path(path)
        if path
        else Path(__file__).resolve().parents[3] / "configs/model/weather_classifier.yaml"
    )
    try:
        cfg = yaml.safe_load(source.read_text(encoding="utf-8"))["weather_classifier"]
        classes = tuple(cfg["classes"])
        size = tuple(cfg["image_size"])
        channels = tuple(cfg["channels"])
        features = dict(cfg["features"])
        train = dict(cfg["train"])
        dropout = float(cfg["dropout"])
        min_confidence = float(cfg["min_confidence"])
        min_margin = float(cfg["min_margin"])
    except (KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
        raise ValueError(f"天气模型配置无效：{exc}") from exc
    if classes != ("fog", "night", "rain", "snow"):
        raise ValueError("天气类别顺序必须与 ACDC 四个子集一致")
    if len(size) != 2 or any(type(x) is not int or x <= 0 for x in size):
        raise ValueError("image_size 必须是两个正整数")
    if len(channels) != 3 or any(type(x) is not int or x <= 0 for x in channels):
        raise ValueError("channels 必须是三个正整数")
    if not all(isfinite(x) for x in (dropout, min_confidence, min_margin)):
        raise ValueError("概率和 dropout 必须为有限数")
    if not (0 <= dropout < 1 and 0 < min_confidence <= 1 and 0 <= min_margin < 1):
        raise ValueError("概率或 dropout 超出有效范围")
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
    return WeatherConfig(
        classes,
        size,
        channels,
        dropout,
        cfg["checkpoint"],
        cfg["device"],
        min_confidence,
        min_margin,
        features,
        train,
    )
