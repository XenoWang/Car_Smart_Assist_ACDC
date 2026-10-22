"""预训练三层视觉特征，由四个属性各自选择空间证据。"""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large

from car_smart_assist.config.weather_attention import WeatherAttentionConfig, attention_model_config
from car_smart_assist.perception.weather import WeatherPrediction, select_device
from car_smart_assist.perception.weather_pyramid import prepare_pyramid_image


def ensure_pretrained_weights(cfg: WeatherAttentionConfig, root: Path) -> Path:
    path = root / cfg.backbone["weights_file"]
    prefix = Path(MobileNet_V3_Large_Weights.IMAGENET1K_V2.url).stem.rsplit("-", 1)[1]
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".download")
        torch.hub.download_url_to_file(
            MobileNet_V3_Large_Weights.IMAGENET1K_V2.url,
            str(temporary),
            hash_prefix=prefix,
        )
        temporary.replace(path)
    if not hashlib.sha256(path.read_bytes()).hexdigest().startswith(prefix):
        raise ValueError("预训练权重哈希与 Torchvision 官方版本不符")
    return path


class WeatherAttention(nn.Module):
    def __init__(self, cfg: WeatherAttentionConfig, pretrained_path: Path | None = None):
        super().__init__()
        network = mobilenet_v3_large(weights=None)
        if pretrained_path is not None:
            network.load_state_dict(
                torch.load(pretrained_path, map_location="cpu", weights_only=True)
            )
        self.backbone = network.features[:16]
        self.cfg = cfg
        self.indices = tuple(cfg.backbone["feature_indices"])
        dim = cfg.head["dimension"]
        self.projections = nn.ModuleList(
            nn.Conv2d(c, dim, 1) for c in cfg.backbone["feature_channels"]
        )
        self.position = nn.Linear(2, dim)
        self.scale_embeddings = nn.Parameter(torch.zeros(len(self.indices), dim))
        self.queries = nn.Parameter(torch.randn(len(cfg.attributes), dim) * 0.02)
        self.attention = nn.MultiheadAttention(dim, cfg.head["heads"], batch_first=True)
        self.context = nn.Linear(2 * len(self.indices) * dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.refine = nn.Sequential(
            nn.Linear(dim, dim * cfg.head["expansion"]),
            nn.GELU(),
            nn.Dropout(cfg.head["dropout"]),
            nn.Linear(dim * cfg.head["expansion"], dim),
        )
        self.dropout = nn.Dropout(cfg.head["dropout"])
        self.class_weights = nn.Parameter(torch.randn(len(cfg.attributes), dim) * 0.02)
        self.class_bias = nn.Parameter(torch.zeros(len(cfg.attributes)))
        self.register_buffer("mean", torch.tensor(cfg.backbone["mean"]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(cfg.backbone["std"]).view(1, 3, 1, 1))
        for i, stride in enumerate(cfg.backbone["feature_strides"]):
            y, x = torch.meshgrid(
                torch.linspace(-1, 1, cfg.image_size[0] // stride),
                torch.linspace(-1, 1, cfg.image_size[1] // stride),
                indexing="ij",
            )
            self.register_buffer(
                f"coordinates_{i}", torch.stack((x, y), -1).reshape(1, -1, 2), persistent=False
            )
        self.set_training_epoch(1)

    def set_training_epoch(self, epoch: int):
        for index, block in enumerate(self.backbone):
            enabled = (
                epoch > self.cfg.backbone["freeze_epochs"]
                and index >= self.cfg.backbone["fine_tune_from"]
            )
            block.requires_grad_(enabled)

    def train(self, mode: bool = True):
        super().train(mode)
        # 保留预训练 BN 统计；冻结的层和小批量微调使用同一归一化。
        self.backbone.eval()
        return self

    def forward(self, images: torch.Tensor):
        value = (images - self.mean) / self.std
        tokens, contexts = [], []
        feature_index = 0
        for index, block in enumerate(self.backbone):
            value = block(value)
            if index in self.indices:
                feature = self.projections[feature_index](value).flatten(2).transpose(1, 2)
                contexts.extend((feature.mean(1), feature.amax(1)))
                position = self.position(getattr(self, f"coordinates_{feature_index}"))
                tokens.append(
                    feature + position + self.scale_embeddings[feature_index].to(feature.dtype)
                )
                feature_index += 1
        spatial = torch.cat(tokens, dim=1)
        query = self.queries.unsqueeze(0).expand(len(images), -1, -1)
        evidence, _ = self.attention(query, spatial, spatial, need_weights=False)
        hidden = self.norm(query + evidence + self.context(torch.cat(contexts, dim=1)).unsqueeze(1))
        hidden = hidden + self.refine(hidden)
        return (self.dropout(hidden) * self.class_weights).sum(-1) + self.class_bias


def _optimize_attention_inference_inplace(model: WeatherAttention) -> WeatherAttention:
    """在独占的推理模型中合并卷积和 BN。"""
    model.eval()
    for module in model.backbone.modules():
        if isinstance(module, nn.Sequential):
            children = list(module.named_children())
            for (conv_name, conv), (bn_name, bn) in zip(children[:-1], children[1:], strict=True):
                if isinstance(conv, nn.Conv2d) and isinstance(bn, nn.BatchNorm2d):
                    module._modules[conv_name] = torch.nn.utils.fusion.fuse_conv_bn_eval(conv, bn)
                    module._modules[bn_name] = nn.Identity()
    if model.cfg.train["channels_last"]:
        model.to(memory_format=torch.channels_last)
    return model


def optimize_attention_inference(model: WeatherAttention) -> WeatherAttention:
    """在独立推理副本中合并卷积和 BN；保留训练模型及检查点结构。"""
    return _optimize_attention_inference_inplace(copy.deepcopy(model))


class WeatherAttentionPredictor:
    def __init__(self, model, cfg, thresholds, device):
        self.model, self.cfg, self.thresholds, self.device = model, cfg, thresholds, device

    @classmethod
    def from_checkpoint(
        cls, path: str | Path, cfg: WeatherAttentionConfig, device: str | None = None
    ):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("format_version") != 1 or payload.get(
            "model_config"
        ) != attention_model_config(cfg):
            raise ValueError("天气注意力检查点与结构配置不符")
        if not payload.get("calibration_signature"):
            raise ValueError("天气注意力候选还没有完成独立校准")
        thresholds = payload["decision_thresholds"]
        if set(thresholds) != set(cfg.attributes) or any(
            not np.isfinite(v) or not 0 < v < 1 for v in thresholds.values()
        ):
            raise ValueError("天气属性阈值无效")
        selected = select_device(device or cfg.device)
        # 旧copy路径也隔离配置；只复制元数据，不复制整份设备参数。
        model = WeatherAttention(copy.deepcopy(cfg)).to(selected)
        model.load_state_dict(payload["model_state"])
        if cfg.train["channels_last"]:
            model.to(memory_format=torch.channels_last)
        return cls(_optimize_attention_inference_inplace(model), cfg, thresholds, selected)

    def predict(self, image: np.ndarray | Image.Image):
        pixels = prepare_pyramid_image(image, self.cfg.image_size)
        tensor = (
            torch.from_numpy(pixels)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(self.device, dtype=torch.float32)
            .div_(255)
        )
        with torch.inference_mode():
            values = self.model(tensor).float().sigmoid()[0].cpu().numpy()
        if not np.isfinite(values).all():
            raise ValueError("天气注意力输出非有限")
        probabilities = dict(zip(self.cfg.attributes, map(float, values), strict=True))
        decisions = {name: value >= self.thresholds[name] for name, value in probabilities.items()}
        return WeatherPrediction(
            tuple(name for name in self.cfg.attributes if decisions[name]),
            decisions,
            probabilities,
            {},
            "预训练图像特征和属性注意力分别判断四属性；仍是视觉估计",
        )
