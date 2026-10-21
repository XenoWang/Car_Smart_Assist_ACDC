"""独立的 YOLO 模型/训练配置；参数放在 configs/model。"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
import yaml


# 固定的标准 detect 后处理合同；不是新增的调参入口。
YOLO_POSTPROCESS_VERSION = "best_class_threshold_v1"
YOLO_CALIBRATION_SCHEMA_VERSION = 2
YOLO_LOCKED_MODE = "locked"
YOLO_PROPOSAL_MODE = "proposal_sweep"
YOLO_AP_MODE = "ap_sweep"
YOLO_SWEEP_CONFIDENCE = 0.001
YOLO_MAX_NMS = 30000
YOLO_MAX_WH = 7680
YOLO_AGNOSTIC_NMS = False


def yolo_model_selection_contract(cfg, manifest):
    """选best的AP口径；部署操作点阈值不参与扫分。"""
    inference = cfg["inference"]
    joint = bool(manifest.get("joint_supervision"))
    return {
        "schema_version": 1,
        "postprocess_version": YOLO_POSTPROCESS_VERSION,
        "ap_mode": YOLO_AP_MODE,
        "sweep_confidence": YOLO_SWEEP_CONFIDENCE,
        "imgsz": inference["imgsz"],
        "nms_iou": inference["iou"],
        "max_det": inference["max_det"],
        "max_nms": YOLO_MAX_NMS,
        "max_wh": YOLO_MAX_WH,
        "agnostic_nms": YOLO_AGNOSTIC_NMS,
        "class_selection": "original_best_class_only",
        "threshold_comparison": ">=",
        "task": "detect",
        "fitness": "0.1*mAP50+0.9*mAP50_95",
        "joint_selection": {
            "known_class_count": manifest["known_class_count"],
            "source_weights": dict(manifest["source_weights"]),
            "valid_roi_ids": list(manifest["valid_roi_ids"]),
            "foreground_exclusive": cfg.get("joint_foreground_exclusive", True),
        } if joint else None,
    }


def yolo_selection_validator_args(contract):
    return {
        "imgsz": contract["imgsz"],
        "iou": contract["nms_iou"],
        "max_det": contract["max_det"],
        "conf": contract["sweep_confidence"],
        "agnostic_nms": contract["agnostic_nms"],
        "single_cls": False,
        "task": contract["task"],
    }


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
