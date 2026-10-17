"""五级轻量编码器融合四层特征，独立判断雾、雨、雪和夜间。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F

from car_smart_assist.config.weather_pyramid import WeatherPyramidConfig, pyramid_model_config
from car_smart_assist.perception.weather import WeatherPrediction, select_device


def prepare_pyramid_image(image: np.ndarray | Image.Image, size: tuple[int, int]) -> np.ndarray:
    """保留画幅比例，用黑边补齐；训练和推理共用。"""
    array = np.asarray(image.convert("RGB") if isinstance(image, Image.Image) else image)
    if array.ndim != 3 or array.shape[2] != 3 or array.dtype != np.uint8 or 0 in array.shape[:2]:
        raise ValueError("新天气模型需要非空 RGB uint8 图像")
    height, width = size
    scale = min(height / array.shape[0], width / array.shape[1])
    resized_width = min(width, max(1, round(array.shape[1] * scale)))
    resized_height = min(height, max(1, round(array.shape[0] * scale)))
    resized = np.asarray(
        Image.fromarray(array).resize((resized_width, resized_height), Image.Resampling.BILINEAR)
    )
    result = np.zeros((height, width, 3), dtype=np.uint8)
    top, left = (height - resized_height) // 2, (width - resized_width) // 2
    result[top : top + resized_height, left : left + resized_width] = resized
    return result


def _stage(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels, 3, stride=2, padding=1, groups=in_channels, bias=False),
        nn.BatchNorm2d(in_channels),
        nn.SiLU(inplace=True),
        nn.Conv2d(in_channels, out_channels, 1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.SiLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1, groups=out_channels, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.SiLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.SiLU(inplace=True),
    )


class WeatherPyramid(nn.Module):
    def __init__(self, cfg: WeatherPyramidConfig):
        super().__init__()
        self.stages = nn.ModuleList()
        previous = 3
        for channels in cfg.channels:
            self.stages.append(_stage(previous, channels))
            previous = channels
        self.lateral = nn.ModuleList(nn.Conv2d(c, cfg.fpn_channels, 1) for c in cfg.channels[1:])
        self.refine = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(
                    cfg.fpn_channels,
                    cfg.fpn_channels,
                    3,
                    padding=1,
                    groups=cfg.fpn_channels,
                    bias=False,
                ),
                nn.BatchNorm2d(cfg.fpn_channels),
                nn.SiLU(inplace=True),
            )
            for _ in self.lateral
        )
        self.pool = nn.AdaptiveAvgPool2d((cfg.pool_grid, cfg.pool_grid))
        self.head = nn.Sequential(
            nn.Dropout(cfg.dropout),
            nn.Linear(4 * cfg.fpn_channels * cfg.pool_grid**2, len(cfg.attributes)),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = []
        value = images.mul(2).sub(1)
        for stage in self.stages:
            value = stage(value)
            features.append(value)
        fused = []
        upper = None
        for index in range(len(self.lateral) - 1, -1, -1):
            current = self.lateral[index](features[index + 1])
            if upper is not None:
                current = current + F.interpolate(upper, size=current.shape[-2:], mode="nearest")
            upper = current
            fused.append(self.pool(self.refine[index](current)).flatten(1))
        return self.head(torch.cat(fused, dim=1))


def masked_weather_loss(logits, labels, mask, name, cfg: WeatherPyramidConfig):
    """只计算已确认属性；未知标签既不是正样本，也不是负样本。"""
    logits, labels, mask = logits.float(), labels.float(), mask.float()
    probability = logits.sigmoid()
    if name == "focal":
        options = cfg.loss["focal"]
        correct_probability = labels * probability + (1 - labels) * (1 - probability)
        alpha = labels * options["alpha"] + (1 - labels) * (1 - options["alpha"])
        terms = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
        terms = terms * alpha * (1 - correct_probability).pow(options["gamma"])
    elif name == "asl":
        options = cfg.loss["asl"]
        negative_probability = (1 - probability + options["clip"]).clamp(max=1)
        terms = (
            -labels * F.logsigmoid(logits)
            - (1 - labels) * negative_probability.clamp_min(1e-8).log()
        )
        correct_probability = labels * probability + (1 - labels) * negative_probability
        gamma = labels * options["gamma_positive"] + (1 - labels) * options["gamma_negative"]
        # ASL 的聚焦权重作为权重使用；detach 不改变全局梯度开关。
        terms = terms * (1 - correct_probability).pow(gamma).detach()
    else:
        raise ValueError("天气损失只能是 focal 或 asl")
    weights = logits.new_tensor([cfg.loss["attribute_weights"][a] for a in cfg.attributes])
    known_weights = mask * weights
    return (terms * known_weights).sum() / known_weights.sum().clamp_min(1)


class WeatherPyramidPredictor:
    def __init__(self, model, cfg, thresholds, device):
        self.model, self.cfg, self.thresholds, self.device = model, cfg, thresholds, device

    @classmethod
    def from_checkpoint(
        cls, path: str | Path, cfg: WeatherPyramidConfig, device: str | None = None
    ):
        selected = select_device(device or cfg.device)
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("format_version") != 3 or payload.get(
            "model_config"
        ) != pyramid_model_config(cfg):
            raise ValueError("新天气权重与模型配置不一致")
        if not payload.get("calibration_signature"):
            raise ValueError("新天气候选尚未完成独立校准，请先完成本轮训练与校准")
        thresholds = payload.get("decision_thresholds", {name: 0.5 for name in cfg.attributes})
        if set(thresholds) != set(cfg.attributes) or any(
            not np.isfinite(t) or not 0 < t < 1 for t in thresholds.values()
        ):
            raise ValueError("天气检查点阈值无效")
        model = WeatherPyramid(cfg).to(selected)
        model.load_state_dict(payload["model_state"])
        model.eval()
        return cls(model, cfg, thresholds, selected)

    def predict(self, image: np.ndarray | Image.Image) -> WeatherPrediction:
        array = prepare_pyramid_image(image, self.cfg.image_size)
        tensor = (
            torch.from_numpy(array)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(self.device, dtype=torch.float32)
            .div_(255)
        )
        with torch.inference_mode():
            values = self.model(tensor).float().sigmoid()[0].cpu().numpy()
        if not np.isfinite(values).all():
            raise ValueError("天气模型输出包含非有限值")
        probabilities = dict(zip(self.cfg.attributes, map(float, values), strict=True))
        decisions = {name: value >= self.thresholds[name] for name, value in probabilities.items()}
        attributes = tuple(name for name in self.cfg.attributes if decisions[name])
        return WeatherPrediction(
            attributes,
            decisions,
            probabilities,
            {},
            "多尺度图像特征分别判断四属性；没有输出属性表示未达到对应阈值",
        )
