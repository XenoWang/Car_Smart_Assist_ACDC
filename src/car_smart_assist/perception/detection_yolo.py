"""YOLO 检测器适配层：框和类别是模型真实预测，距离和方向仍是未知。"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from car_smart_assist.advisory.schema import PerceptionResult, TargetObject
from car_smart_assist.config.detection import resolve_yolo_device


def configure_yolo_environment(project_root: Path) -> None:
    settings_dir = project_root / "artifacts/ultralytics_settings"
    settings_dir.mkdir(parents=True, exist_ok=True)
    # 改动只影响当前这个 Python 进程和项目的产物配置文件。
    os.environ["YOLO_CONFIG_DIR"] = str(settings_dir)


def load_yolo_model(project_root: Path, checkpoint: Path):
    configure_yolo_environment(project_root)
    try:
        from ultralytics import YOLO, settings
    except ImportError as exc:
        raise ImportError(
            "Install requirements/requirements-detection.txt in project .venv"
        ) from exc
    settings.update(
        {
            "datasets_dir": str(project_root / "data/external"),
            "weights_dir": str(project_root / "artifacts/checkpoints/yolo_pretrained"),
            "runs_dir": str(project_root / "artifacts/runs/detection"),
            "sync": False,
            "hub": False,
            "clearml": False,
            "comet": False,
            "dvc": False,
            "mlflow": False,
            "neptune": False,
            "raytune": False,
            "wandb": False,
        }
    )
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    return YOLO(str(checkpoint))


class YoloDetectionPredictor:
    def __init__(self, model, cfg: dict[str, Any], project_root: Path) -> None:
        self.model = model
        self.cfg = cfg
        self.device = resolve_yolo_device(cfg["device"])
        self.confidence = float(cfg["inference"]["confidence"])
        self.class_thresholds = {}
        calibration = project_root / cfg["calibration_file"]
        if calibration.is_file():
            result = json.loads(calibration.read_text(encoding="utf-8"))
            checkpoint = project_root / cfg["checkpoint"]
            digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            if result["checkpoint_sha256"] != digest:
                raise ValueError(
                    "Detection calibration refers to different weights; recalibrate first"
                )
            for arg, report_key in (("iou", "nms_iou"), ("imgsz", "imgsz"), ("max_det", "max_det")):
                if result[report_key] != cfg["inference"][arg]:
                    raise ValueError("Detection inference settings changed; recalibrate first")
            self.confidence = float(result["confidence_threshold"])
            self.class_thresholds = result.get("class_confidence_thresholds", {})
            if self.class_thresholds:
                if any(
                    not math.isfinite(float(value)) or not 0 < float(value) <= 1
                    for value in self.class_thresholds.values()
                ):
                    raise ValueError("Detection class calibration thresholds must be in (0, 1]")
                self.confidence = min(float(value) for value in self.class_thresholds.values())
            if not math.isfinite(self.confidence) or not 0 < self.confidence <= 1:
                raise ValueError("Detection calibration confidence must be in (0, 1]")

    @classmethod
    def from_config(cls, cfg: dict[str, Any], project_root: Path):
        checkpoint = project_root / cfg["checkpoint"]
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Trained YOLO checkpoint not found: {checkpoint}")
        return cls(load_yolo_model(project_root, checkpoint), cfg, project_root)

    def predict(self, image: np.ndarray) -> PerceptionResult:
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError("YOLO pipeline input must be RGB uint8")
        args = self.cfg["inference"]
        # PIL 输入是 RGB。Ultralytics 把 numpy 输入当 BGR 处理，所以不要把 RGB 数组直接传进去。
        result = self.model.predict(
            source=Image.fromarray(image),
            device=self.device,
            imgsz=args["imgsz"],
            conf=self.confidence,
            iou=args["iou"],
            max_det=args["max_det"],
            verbose=False,
        )[0]
        objects = []
        for box, score, label in zip(
            result.boxes.xyxy.cpu().tolist(),
            result.boxes.conf.cpu().tolist(),
            result.boxes.cls.cpu().tolist(),
            strict=True,
        ):
            category = result.names[int(label)]
            if float(score) < float(self.class_thresholds.get(category, self.confidence)):
                continue
            objects.append(
                TargetObject(
                    category=category,
                    bbox=tuple(box),
                    confidence=float(score),
                )
            )
        return PerceptionResult(
            objects=objects,
            object_detection_available=True,
            object_detection_classes=tuple(result.names[i] for i in sorted(result.names)),
            road_obstacle_detection_available=(
                self.cfg["inference"].get("obstacle_class", "road_obstacle")
                in result.names.values()
            ),
        )
