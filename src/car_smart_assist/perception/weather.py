"""独立的天气现象/光照多标签小模型和可解释视觉线索。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import nn

from car_smart_assist.config.weather import WeatherConfig

FEATURE_NAMES = (
    "reflection_proxy",
    "wet_surface_proxy",
    "snow_coverage_proxy",
    "low_visibility_proxy",
)


def prepare_image(image: np.ndarray | Image.Image, size: tuple[int, int]) -> np.ndarray:
    """和训练使用同一套 RGB 缩放口径，返回 uint8 HWC。"""
    array = np.asarray(image.convert("RGB") if isinstance(image, Image.Image) else image)
    if array.ndim != 3 or array.shape[2] != 3 or array.dtype != np.uint8:
        raise ValueError("天气模型输入必须为 RGB uint8 图像")
    h, w = size
    if array.shape[:2] == size:
        return np.ascontiguousarray(array)
    return np.asarray(Image.fromarray(array).resize((w, h), Image.BILINEAR), dtype=np.uint8)


def visual_cues(image: np.ndarray, cfg: WeatherConfig) -> dict[str, float]:
    """输出视觉代理比例；近似下部区域不等于道路分割，也不等于物理积水深度。"""
    if image.shape[:2] != cfg.image_size or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("视觉线索输入尺寸与模型配置不一致")
    pixels = image.astype(np.float32) / 255.0
    h, w = cfg.image_size
    top = int(h * cfg.features["road_top_fraction"])
    side = int(w * cfg.features["road_side_fraction"])
    road = pixels[top:, side : w - side]
    gray = road @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    saturation = road.max(axis=2) - road.min(axis=2)
    bright = gray >= cfg.features["bright_threshold"]
    low_saturation = saturation <= cfg.features["saturation_max"]
    # 单帧里的亮斑、暗且平滑的区域都可能来自非雨水原因，只作为分类器的输入线索。
    gradient = np.zeros_like(gray)
    gradient[:, 1:] += np.abs(gray[:, 1:] - gray[:, :-1])
    gradient[1:, :] += np.abs(gray[1:, :] - gray[:-1, :])
    # 高亮边缘可能提示反光；大块亮且平滑可能提示积雪。两者都可能被其他物体混淆。
    reflections = bright & (gradient > cfg.features["smooth_gradient_max"])
    wet_proxy = (gray <= cfg.features["dark_threshold"]) & (
        gradient <= cfg.features["smooth_gradient_max"]
    )
    upper = pixels[: max(1, top), side : w - side]
    upper_gray = upper @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    contrast = float(np.std(upper_gray))
    return {
        "reflection_proxy": float(reflections.mean()),
        "wet_surface_proxy": float(wet_proxy.mean()),
        "snow_coverage_proxy": float(
            (bright & low_saturation & (gradient <= cfg.features["smooth_gradient_max"])).mean()
        ),
        "low_visibility_proxy": float(np.clip(1.0 - 4.0 * contrast, 0.0, 1.0)),
    }


def image_tensors(
    image: np.ndarray, cfg: WeatherConfig, cues: dict[str, float] | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    if cues is None:
        cues = visual_cues(image, cfg)
    pixels = torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1).float().div_(255.0)
    feature_vector = torch.tensor([cues[name] for name in FEATURE_NAMES], dtype=torch.float32)
    return pixels, feature_vector


class WeatherClassifier(nn.Module):
    """小型 CNN，融合全图特征和四个视觉代理比例。"""

    def __init__(self, cfg: WeatherConfig) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_channels = 3
        for channels in cfg.channels:
            layers.extend(
                (
                    nn.Conv2d(
                        in_channels, channels, kernel_size=3, stride=2, padding=1, bias=False
                    ),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                )
            )
            in_channels = channels
        self.encoder = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.cue_encoder = nn.Sequential(nn.Linear(len(FEATURE_NAMES), 16), nn.ReLU(inplace=True))
        self.head = nn.Sequential(
            nn.Dropout(cfg.dropout), nn.Linear(in_channels + 16, len(cfg.attributes))
        )
        self.rain_adapter = None
        if cfg.rain_adapter_channels:
            self.rain_adapter = nn.Sequential(
                nn.Linear(in_channels + 16, cfg.rain_adapter_channels), nn.ReLU(inplace=True),
                nn.Linear(cfg.rain_adapter_channels, 1),
            )
            # 零残差初始化保持原多标签模型的全部输出。
            nn.init.zeros_(self.rain_adapter[-1].weight)
            nn.init.zeros_(self.rain_adapter[-1].bias)

    def forward(self, images: torch.Tensor, cues: torch.Tensor) -> torch.Tensor:
        features = torch.cat((self.encoder(images), self.cue_encoder(cues)), dim=1)
        logits = self.head(features)
        if self.rain_adapter is not None:
            rain = logits[:, 1:2] + self.rain_adapter(features)
            logits = torch.cat((logits[:, :1], rain, logits[:, 2:]), dim=1)
        return logits


@dataclass(frozen=True)
class WeatherPrediction:
    attributes: tuple[str, ...]
    decisions: dict[str, bool]
    probabilities: dict[str, float]
    cues: dict[str, float]
    reason: str

    @property
    def warning(self) -> str | None:
        return render_weather_warning(self.attributes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "attributes": list(self.attributes),
            "decisions": dict(self.decisions),
            "probabilities": dict(self.probabilities),
            "cues": dict(self.cues),
            "warning": self.warning,
            "reason": self.reason,
        }


def render_weather_warning(attributes: tuple[str, ...]) -> str | None:
    """生成提示信息；这段文字本身不会请求接管。"""
    detected = set(attributes)
    night = "night" in detected
    fog = "fog" in detected
    rain = "rain" in detected
    snow = "snow" in detected
    if fog and rain:
        return "雾雨天气，视线可能受影响，请减速"
    if fog:
        return "夜间有雾，视线可能受影响，请减速" if night else "检测到雾，视线可能受影响，请减速"
    if rain:
        return "夜间降雨，路面可能湿滑，请减速" if night else "检测到降雨，路面可能湿滑，请减速"
    if snow:
        return "检测到降雪，请减速并留足制动距离"
    if night:
        return "夜间行车，请注意观察路况"
    return None


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and (
        not torch.cuda.is_available()
        or (device.index is not None and device.index >= torch.cuda.device_count())
    ):
        raise ValueError(f"天气模型请求的 CUDA 设备不可用：{requested}")
    if device.type not in ("cuda", "cpu"):
        raise ValueError("天气模型只支持 CPU 或 CUDA")
    return device


class WeatherPredictor:
    """只从训练检查点创建；随机初始权重不能用来判断真实图片。"""

    def __init__(self, model: WeatherClassifier, cfg: WeatherConfig, device: torch.device) -> None:
        self.model = model.to(device).eval()
        self.cfg = cfg
        self.device = device

    @classmethod
    def from_checkpoint(cls, path: str | Path, cfg: WeatherConfig) -> WeatherPredictor:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint.get("format_version") != 2:
            raise ValueError("天气检查点格式不兼容")
        if checkpoint["model_config"].get("rain_adapter_channels", 0) != cfg.rain_adapter_channels:
            raise ValueError("天气检查点的 rain_adapter_channels 与当前配置不一致")
        for key in ("attributes", "image_size", "channels", "features"):
            expected = (
                list(getattr(cfg, key))
                if key in ("attributes", "image_size", "channels")
                else getattr(cfg, key)
            )
            if checkpoint["model_config"][key] != expected:
                raise ValueError(f"天气检查点的 {key} 与当前配置不一致")
        model = WeatherClassifier(cfg)
        model.load_state_dict(checkpoint["model_state"])
        return cls(model, cfg, select_device(cfg.device))

    @torch.inference_mode()
    def predict(self, image: np.ndarray | Image.Image) -> WeatherPrediction:
        resized = prepare_image(image, self.cfg.image_size)
        cues = visual_cues(resized, self.cfg)
        pixels, vector = image_tensors(resized, self.cfg, cues)
        logits = self.model(
            pixels.unsqueeze(0).to(self.device), vector.unsqueeze(0).to(self.device)
        )
        probabilities = torch.sigmoid(logits.float()).cpu().numpy()[0]
        if not np.isfinite(probabilities).all():
            raise ValueError("天气模型概率包含非有限值")
        probability_map = {
            name: float(probabilities[i]) for i, name in enumerate(self.cfg.attributes)
        }
        decisions = {
            name: probability_map[name] >= self.cfg.decision_thresholds[name]
            for name in self.cfg.attributes
        }
        attributes = tuple(name for name in self.cfg.attributes if decisions[name])
        return WeatherPrediction(
            attributes=attributes,
            decisions=decisions,
            probabilities=probability_map,
            cues=cues,
            reason="天气属性由独立阈值分别判断；属性不确定不单独触发接管",
        )
