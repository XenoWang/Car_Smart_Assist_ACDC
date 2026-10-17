"""新天气实验的模型、损失和数据参数统一从 YAML 读取。"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class WeatherPyramidConfig:
    attributes: tuple[str, ...]
    image_size: tuple[int, int]
    channels: tuple[int, ...]
    fpn_channels: int
    pool_grid: int
    dropout: float
    device: str
    data: dict[str, Any]
    train: dict[str, Any]
    loss: dict[str, Any]
    evaluation: dict[str, Any]
    checkpoint_root: str
    report_dir: str


def load_weather_pyramid_config(path: str | Path) -> WeatherPyramidConfig:
    try:
        values = yaml.safe_load(Path(path).read_text(encoding="utf-8"))["weather_pyramid"]
        cfg = WeatherPyramidConfig(
            tuple(values["attributes"]),
            tuple(values["image_size"]),
            tuple(values["channels"]),
            values["fpn_channels"],
            values["pool_grid"],
            float(values["dropout"]),
            values["device"],
            dict(values["data"]),
            dict(values["train"]),
            dict(values["loss"]),
            dict(values["evaluation"]),
            values["checkpoint_root"],
            values["report_dir"],
        )
        if cfg.attributes != ("fog", "rain", "snow", "night"):
            raise ValueError("天气属性必须是 fog/rain/snow/night，分别判断、可以共存")
        if len(cfg.image_size) != 2 or any(
            type(v) is not int or v < 32 or v % 32 for v in cfg.image_size
        ):
            raise ValueError("image_size 的高和宽必须是至少 32 的 32 倍数")
        if len(cfg.channels) != 5 or any(type(v) is not int or v <= 0 for v in cfg.channels):
            raise ValueError("新模型需要五级正整数 channels")
        for value in (cfg.fpn_channels, cfg.pool_grid):
            if type(value) is not int or value <= 0:
                raise ValueError("fpn_channels/pool_grid 必须为正整数")
        if not isfinite(cfg.dropout) or not 0 <= cfg.dropout < 1:
            raise ValueError("dropout 必须在 [0, 1) 中")
        ratios = cfg.data["split_ratios"]
        if (
            set(ratios) != {"train", "validation", "calibration", "test"}
            or any(not isfinite(v) or not 0 < v < 1 for v in ratios.values())
            or abs(sum(ratios.values()) - 1) > 1e-8
        ):
            raise ValueError("四分比例必须为正数且合计为 1")
        if type(cfg.data["seed"]) is not int:
            raise ValueError("seed 必须为整数")
        if not cfg.data["acdc_splits"] or set(cfg.data["acdc_splits"]) - {"train", "val", "test"}:
            raise ValueError("acdc_splits 只能包含 train/val/test")
        if len(set(cfg.data["acdc_splits"])) != len(cfg.data["acdc_splits"]):
            raise ValueError("acdc_splits 不能重复")
        if type(cfg.data["include_normal_references"]) is not bool:
            raise ValueError("include_normal_references 必须是布尔值")
        for options, names in (
            (cfg.train, ("epochs", "resume_extra_epochs", "min_epochs", "patience", "batch_size")),
            (
                cfg.evaluation,
                (
                    "batch_size",
                    "threshold_steps",
                    "benchmark_iterations",
                    "benchmark_warmup",
                    "cpu_threads",
                ),
            ),
        ):
            if any(type(options[k]) is not int or options[k] <= 0 for k in names):
                raise ValueError("训练轮数、批量大小和评估次数必须是正整数")
        if cfg.train["num_workers"] != 0:
            raise ValueError("当前可恢复实验使用 num_workers=0")
        if type(cfg.train["force_epochs"]) is not bool:
            raise ValueError("force_epochs 必须为布尔值")
        if cfg.train["amp_dtype"] not in ("auto", "bfloat16", "float16"):
            raise ValueError("amp_dtype 只能是 auto/bfloat16/float16")
        if cfg.evaluation["threshold_tie_break"] not in ("median", "highest"):
            raise ValueError("threshold_tie_break 只能是 median 或 highest")
        for key in ("learning_rate", "gradient_clip", "rain_selection_weight"):
            if not isfinite(cfg.train[key]) or cfg.train[key] <= 0:
                raise ValueError(f"{key} 必须是正数")
        if not isfinite(cfg.train["weight_decay"]) or cfg.train["weight_decay"] < 0:
            raise ValueError("weight_decay 必须非负")
        if not 0 <= cfg.train["horizontal_flip_probability"] <= 1:
            raise ValueError("水平翻转概率无效")
        for key in ("brightness_range", "contrast_range"):
            bounds = cfg.train[key]
            if (
                len(bounds) != 2
                or not all(isfinite(v) for v in bounds)
                or not 0 < bounds[0] <= bounds[1]
            ):
                raise ValueError(f"{key} 无效")
        weights = cfg.loss["attribute_weights"]
        if set(weights) != set(cfg.attributes) or any(
            not isfinite(v) or v <= 0 for v in weights.values()
        ):
            raise ValueError("四个属性的损失权重必须为正数")
        focal, asl = cfg.loss["focal"], cfg.loss["asl"]
        if not 0 < focal["alpha"] < 1 or not isfinite(focal["gamma"]) or focal["gamma"] < 0:
            raise ValueError("Focal 参数无效")
        if (
            any(not isfinite(asl[k]) or asl[k] < 0 for k in ("gamma_positive", "gamma_negative"))
            or not 0 <= asl["clip"] < 1
        ):
            raise ValueError("ASL 参数无效")
        if any(
            not isinstance(v, str) or not v
            for v in (cfg.device, cfg.checkpoint_root, cfg.report_dir)
        ):
            raise ValueError("设备和输出目录不能为空")
    except (KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
        raise ValueError(f"新天气配置无效：{exc}") from exc
    return cfg


def pyramid_model_config(cfg: WeatherPyramidConfig) -> dict[str, Any]:
    return {
        "attributes": list(cfg.attributes),
        "image_size": list(cfg.image_size),
        "channels": list(cfg.channels),
        "fpn_channels": cfg.fpn_channels,
        "pool_grid": cfg.pool_grid,
        "dropout": cfg.dropout,
        "preprocessing": "rgb-letterbox-bilinear-v1",
    }
