"""YOLO 检测器适配层：框和类别是模型真实预测，距离和方向仍是未知。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from PIL import Image

from car_smart_assist.advisory.schema import PerceptionResult, TargetObject
from car_smart_assist.config.detection import YOLO_AGNOSTIC_NMS, resolve_yolo_device
from car_smart_assist.data.image_input import ImageInput, to_rgb_uint8
from car_smart_assist.perception.yolo_postprocess import (
    calibration_path, detection_predictor_class, load_calibration,
)


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
        result = load_calibration(project_root, cfg, model.names)
        self.confidence = float(result["confidence_threshold"])
        self.class_thresholds = result["class_confidence_thresholds"]
        self.predictor_class = detection_predictor_class(self.confidence, self.class_thresholds)
        # model.names可能已初始化默认predictor；predict(predictor=...)仅在首次创建时生效。
        self.model.predictor = None

    @classmethod
    def from_config(cls, cfg: dict[str, Any], project_root: Path):
        calibration = calibration_path(project_root, cfg)
        if not calibration.is_file():
            raise FileNotFoundError(f"Current detection calibration missing; recalibrate first: {calibration}")
        checkpoint = project_root / cfg["checkpoint"]
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Trained YOLO checkpoint not found: {checkpoint}")
        return cls(load_yolo_model(project_root, checkpoint), cfg, project_root)

    def predict(self, image: ImageInput) -> PerceptionResult:
        image = to_rgb_uint8(image, copy=False)
        args = self.cfg["inference"]
        if not isinstance(self.model.predictor, self.predictor_class):
            self.model.predictor = None
        # PIL 输入是 RGB。Ultralytics 把 numpy 输入当 BGR 处理，所以不要把 RGB 数组直接传进去。
        result = self.model.predict(
            source=Image.fromarray(image),
            device=self.device,
            imgsz=args["imgsz"],
            conf=self.confidence,
            iou=args["iou"],
            max_det=args["max_det"],
            agnostic_nms=YOLO_AGNOSTIC_NMS,
            predictor=self.predictor_class,
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
