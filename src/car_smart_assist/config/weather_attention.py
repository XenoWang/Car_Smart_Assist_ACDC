"""属性注意力天气候选的参数集中在独立 YAML 中。"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class WeatherAttentionConfig:
    attributes: tuple[str, ...]
    image_size: tuple[int, int]
    device: str
    backbone: dict[str, Any]
    head: dict[str, Any]
    data: dict[str, Any]
    train: dict[str, Any]
    loss: dict[str, Any]
    evaluation: dict[str, Any]
    checkpoint_root: str
    report_dir: str
    report_file: str


def load_weather_attention_config(path: str | Path) -> WeatherAttentionConfig:
    try:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))["weather_attention"]
        cfg = WeatherAttentionConfig(
            tuple(raw["attributes"]),
            tuple(raw["image_size"]),
            raw["device"],
            *[dict(raw[k]) for k in ("backbone", "head", "data", "train", "loss", "evaluation")],
            raw["checkpoint_root"],
            raw["report_dir"],
            raw["report_file"],
        )
        validate_weather_attention_config(cfg)
    except (KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
        raise ValueError(f"天气注意力配置无效：{exc}") from exc
    return cfg


def validate_weather_attention_config(cfg: WeatherAttentionConfig):
    if cfg.attributes != ("fog", "rain", "snow", "night"):
        raise ValueError("保持四个独立天气属性 fog/rain/snow/night")
    if len(cfg.image_size) != 2 or any(
        type(v) is not int or v < 32 or v % 32 for v in cfg.image_size
    ):
        raise ValueError("输入高宽须为不小于 32 的 32 倍数")
    b, h, t, e = cfg.backbone, cfg.head, cfg.train, cfg.evaluation
    if b["name"] != "mobilenet_v3_large" or b["weights"] != "IMAGENET1K_V2":
        raise ValueError("当前候选使用 MobileNetV3 Large 的 ImageNet V2 权重")
    if (
        b["feature_indices"] != [6, 12, 15]
        or b["feature_channels"] != [40, 112, 160]
        or b["feature_strides"] != [8, 16, 32]
    ):
        raise ValueError("预训练特征固定取 stride 8/16/32 的三层")
    if type(b["fine_tune_from"]) is not int or not 0 <= b["fine_tune_from"] <= 15:
        raise ValueError("fine_tune_from 必须在 [0,15]")
    if type(b["freeze_epochs"]) is not int or b["freeze_epochs"] < 0:
        raise ValueError("冻结轮数必须非负")
    for key in ("mean", "std"):
        if len(b[key]) != 3 or any(not isfinite(v) for v in b[key]):
            raise ValueError("RGB 标准化参数无效")
    if any(v <= 0 for v in b["std"]):
        raise ValueError("标准差必须为正数")
    for opts, keys in (
        (h, ("dimension", "heads", "expansion")),
        (t, ("epochs", "batch_size")),
        (
            e,
            (
                "batch_size",
                "threshold_steps",
                "benchmark_iterations",
                "benchmark_warmup",
                "cpu_threads",
            ),
        ),
    ):
        if any(type(opts[k]) is not int or opts[k] <= 0 for k in keys):
            raise ValueError("尺寸、轮数和批量参数必须为正整数")
    if h["dimension"] % h["heads"] or not isfinite(h["dropout"]) or not 0 <= h["dropout"] < 1:
        raise ValueError("注意力维度须整除 heads，dropout 须在 [0,1)")
    for key in ("learning_rate", "backbone_learning_rate", "gradient_clip"):
        if not isfinite(t[key]) or t[key] <= 0:
            raise ValueError(f"{key} 必须为正数")
    if not isfinite(t["weight_decay"]) or t["weight_decay"] < 0:
        raise ValueError("weight_decay 必须非负")
    if type(t["warmup_epochs"]) is not int or not 0 <= t["warmup_epochs"] < t["epochs"]:
        raise ValueError("warmup_epochs 须小于累计训练目标")
    schedule = t.get("schedule_epochs", t["epochs"])
    if type(schedule) is not int or schedule < t["epochs"]:
        raise ValueError("schedule_epochs 必须是至少覆盖训练目标的整数")
    coefficient = t.get("l2_sp", 0.0)
    if type(coefficient) not in (int, float) or not isfinite(coefficient) or coefficient < 0:
        raise ValueError("l2_sp 系数必须是有限非负数")
    continuation = t.get("continuation")
    if "continuation" in t:
        required = {
            "checkpoint",
            "checkpoint_sha256",
            "single_best",
            "single_best_sha256",
            "source_epoch",
            "additional_epochs",
            "start_factor",
        }
        if not isinstance(continuation, dict) or set(continuation) != required:
            raise ValueError("追加阶段须明确源 last、best、哈希、轮数和起始学习率比例")
        for key in ("source_epoch", "additional_epochs"):
            if type(continuation[key]) is not int or continuation[key] < 1:
                raise ValueError("追加阶段轮数须为正整数")
        if t["epochs"] != continuation["source_epoch"] + continuation["additional_epochs"]:
            raise ValueError("累计目标须等于源轮数加追加轮数")
        for key in ("checkpoint", "single_best"):
            if not isinstance(continuation[key], str) or not continuation[key]:
                raise ValueError("追加阶段源路径不能为空")
        for key in ("checkpoint_sha256", "single_best_sha256"):
            value = continuation[key]
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(c not in "0123456789abcdef" for c in value)
            ):
                raise ValueError("源检查点须提供完整 SHA256")
        factor = continuation["start_factor"]
        if (
            type(factor) not in (int, float)
            or not isfinite(factor)
            or not t["minimum_lr_ratio"] <= factor <= 1
        ):
            raise ValueError("追加起始学习率比例必须在下界和1之间")
        if (
            not isinstance(t.get("swad"), dict)
            or not t["swad"].get("enabled")
            or t["swad"].get("start_epoch") != continuation["source_epoch"] + 1
        ):
            raise ValueError("追加阶段须在首轮重新开始 SWAD 收集")
    swad = t.get("swad")
    if "swad" in t:
        if not isinstance(swad, dict) or set(swad) != {
            "enabled",
            "n_converge",
            "n_tolerance",
            "tolerance_ratio",
            "start_epoch",
        }:
            raise ValueError("SWAD 配置须明确启用状态、两个窗口、容忍比例和开始轮次")
        if type(swad["enabled"]) is not bool:
            raise ValueError("SWAD enabled 必须是布尔值")
        if any(
            type(swad[k]) is not int or swad[k] < 1
            for k in ("n_converge", "n_tolerance", "start_epoch")
        ):
            raise ValueError("SWAD 窗口和开始轮次必须为正整数")
        if swad["n_tolerance"] < swad["n_converge"]:
            raise ValueError("SWAD 容忍窗口须至少覆盖收敛窗口")
        if not max(b["freeze_epochs"], t["warmup_epochs"]) < swad["start_epoch"] <= t["epochs"]:
            raise ValueError("SWAD 开始轮次须位于冻结和预热之后、预算以内")
        ratio = swad["tolerance_ratio"]
        if type(ratio) not in (int, float) or not isfinite(ratio) or ratio < 0:
            raise ValueError("SWAD 容忍比例必须有限且非负")
    if not 0 < t["minimum_lr_ratio"] <= 1 or not 0 <= t["horizontal_flip_probability"] <= 1:
        raise ValueError("学习率下界或翻转概率无效")
    for key in ("amp", "channels_last"):
        if type(t[key]) is not bool:
            raise ValueError(f"{key} 必须为布尔值")
    for key in ("brightness_range", "contrast_range"):
        v = t[key]
        if len(v) != 2 or any(not isfinite(x) for x in v) or not 0 < v[0] <= v[1]:
            raise ValueError("颜色增广范围无效")
    for weights, keys in (
        (t["source_sampling_weights"], {"acdc", "pixel_accurate"}),
        (cfg.loss["attribute_weights"], set(cfg.attributes)),
        (e["selection_weights"], set(cfg.attributes)),
    ):
        if set(weights) != keys or any(not isfinite(v) or v <= 0 for v in weights.values()):
            raise ValueError("来源／属性权重无效")
    asl = cfg.loss["asl"]
    if (
        any(not isfinite(asl[k]) or asl[k] < 0 for k in ("gamma_positive", "gamma_negative"))
        or not 0 <= asl["clip"] < 1
    ):
        raise ValueError("ASL 参数无效")
    if not 0 <= e["ap_selection_fraction"] <= 1:
        raise ValueError("AP 评分比例无效")
    ratios = cfg.data["split_ratios"]
    if (
        set(ratios) != {"train", "validation", "calibration", "test"}
        or any(not isfinite(v) or v <= 0 for v in ratios.values())
        or abs(sum(ratios.values()) - 1) > 1e-8
    ):
        raise ValueError("四分比例必须为正数且合计 1")
    if type(cfg.data["seed"]) is not int or type(cfg.data["include_normal_references"]) is not bool:
        raise ValueError("种子或参考图设置无效")
    splits = cfg.data["acdc_splits"]
    if not splits or len(set(splits)) != len(splits) or set(splits) - {"train", "val", "test"}:
        raise ValueError("ACDC 集合无效")
    for path in (cfg.checkpoint_root, cfg.report_dir, cfg.report_file, b["weights_file"]):
        if not isinstance(path, str) or not path:
            raise ValueError("输出与权重路径不能为空")
    references = e.get("attention_references", {})
    if not isinstance(references, dict) or set(references) & {
        "attention_asl",
        "single_best",
        "swad",
        "original",
        "pyramid_asl",
    }:
        raise ValueError("注意力对照名称无效")
    for reference in references.values():
        if (
            not isinstance(reference, dict)
            or set(reference) != {"config", "checkpoint"}
            or any(not isinstance(p, str) or not p for p in reference.values())
        ):
            raise ValueError("注意力对照须提供配置和权重路径")
    if "summary_file" in e and (
        not isinstance(e["summary_file"], str)
        or not e["summary_file"]
        or e["summary_file"] == cfg.report_file
    ):
        raise ValueError("汇总报告须使用独立文件")
    if "summary_tag" in e and (
        not isinstance(e["summary_tag"], str)
        or not e["summary_tag"]
        or any(c not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ_0123456789" for c in e["summary_tag"])
    ):
        raise ValueError("报告标记只使用大写字母、数字和下划线")


def attention_model_config(cfg: WeatherAttentionConfig):
    return {
        "attributes": list(cfg.attributes),
        "image_size": list(cfg.image_size),
        "backbone": {
            k: v
            for k, v in cfg.backbone.items()
            if k not in ("weights_file", "fine_tune_from", "freeze_epochs")
        },
        "head": cfg.head,
        "preprocessing": "rgb-letterbox-bilinear-v1",
        "architecture": "mobilenet-three-scale-attribute-query-v1",
        "inference": "conv-bn-fold-v1",
    }
