"""独立的 YOLO 模型/训练配置；参数放在 configs/model。"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
import yaml


def load_yolo_config(path: Path) -> dict[str, Any]:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))["yolo_detection"]
    train = cfg["train"]
    for name in ("epochs", "seed", "imgsz", "batch", "workers", "patience"):
        if type(train[name]) is not int or train[name] < 0:
            raise ValueError(f"YOLO {name} must be a non-negative integer")
    if min(train["epochs"], train["imgsz"], train["batch"]) == 0:
        raise ValueError("YOLO epochs/imgsz/batch must be positive")
    if not isinstance(train["deterministic"], bool):
        raise ValueError("YOLO deterministic must be boolean")
    for name in ("enabled_for_inference", "joint_foreground_exclusive", "evaluation_matches_prediction"):
        if name in cfg and not isinstance(cfg[name], bool):
            raise ValueError(f"YOLO {name} must be boolean")
    extra = train.get("resume_extra_epochs")
    if extra is not None and (type(extra) is not int or extra <= 0):
        raise ValueError("YOLO resume_extra_epochs must be null or a positive integer")
    inference = cfg["inference"]
    for name in ("imgsz", "max_det"):
        if type(inference[name]) is not int or inference[name] <= 0:
            raise ValueError(f"YOLO inference {name} must be a positive integer")
    for name in ("confidence", "iou"):
        value = inference[name]
        if isinstance(value, bool) or not math.isfinite(value) or not 0 < value <= 1:
            raise ValueError(f"YOLO inference {name} must be in (0, 1]")
    return cfg


def resolve_yolo_device(requested: str) -> str:
    if requested == "auto":
        return "0" if torch.cuda.is_available() else "cpu"
    if requested == "cpu":
        return "cpu"
    device = torch.device(requested)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError(f"YOLO CUDA device is unavailable: {requested}")
    index = device.index or 0
    if index >= torch.cuda.device_count():
        raise ValueError(f"YOLO CUDA device is unavailable: {requested}")
    return str(index)
