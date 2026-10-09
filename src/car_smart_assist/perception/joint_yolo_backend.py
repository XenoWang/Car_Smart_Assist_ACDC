"""Ultralytics dataset/trainer integration for partially annotated joint detection."""

from __future__ import annotations

import json
import os
from copy import copy
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from car_smart_assist.perception.detection_yolo import configure_yolo_environment
from car_smart_assist.perception.joint_yolo import (
    JointDetectionLoss,
    preserve_classification_outputs,
)
from car_smart_assist.perception.yolo_training import detection_trainer_for_resume

# Importing Ultralytics initializes settings; keep that initialization in the project.
if not os.environ.get("YOLO_CONFIG_DIR"):
    configure_yolo_environment(Path(__file__).resolve().parents[3])
from ultralytics.data.dataset import YOLODataset  # noqa: E402
from ultralytics.models.yolo.detect.val import DetectionValidator  # noqa: E402
from ultralytics.utils import ops  # noqa: E402
from ultralytics.utils.metrics import DetMetrics  # noqa: E402


def load_joint_index(data):
    path = Path(data["joint_manifest"])
    manifest = json.loads(path.read_text(encoding="utf-8"))
    index = {str((path.parent / row["image"]).resolve()): row for row in manifest["images"]}
    return manifest, index


def native_roi(row, manifest):
    with Image.open(row["roi_mask"]) as image:
        mask = np.isin(np.asarray(image), manifest["valid_roi_ids"])
    height, width = mask.shape
    # Sparse semantic annotations may leave holes inside a positive instance box.
    for _, xc, yc, w, h in row["boxes"]:
        x1, x2 = int((xc - w / 2) * width), int(np.ceil((xc + w / 2) * width))
        y1, y2 = int((yc - h / 2) * height), int(np.ceil((yc + h / 2) * height))
        mask[max(0, y1) : min(height, y2), max(0, x1) : min(width, x2)] = True
    return mask


class JointDataset(YOLODataset):
    def __init__(self, *args, **kwargs):
        self.manifest, self.index = load_joint_index(kwargs["data"])
        super().__init__(*args, **kwargs)

    def __getitem__(self, item):
        sample = super().__getitem__(item)
        row = self.index[str(Path(sample["im_file"]).resolve())]
        height, width = sample["img"].shape[-2:]
        roi = np.ones((height, width), dtype=np.uint8)
        if row["roi_mask"]:
            raw = native_roi(row, self.manifest)
            original_height, original_width = raw.shape
            ratio = min(height / original_height, width / original_width)
            resized_height, resized_width = (
                round(original_height * ratio),
                round(original_width * ratio),
            )
            resized = np.asarray(
                Image.fromarray(raw.astype(np.uint8)).resize(
                    (resized_width, resized_height), Image.Resampling.NEAREST
                )
            )
            top, left = (
                round((height - resized_height) / 2 - 0.1),
                round((width - resized_width) / 2 - 0.1),
            )
            roi.fill(0)
            roi[top : top + resized_height, left : left + resized_width] = resized
        coverage = torch.zeros(len(self.manifest["classes"]), dtype=torch.bool)
        coverage[row["supervised_classes"]] = True
        sample.update(
            class_supervision=coverage,
            obstacle_roi=torch.from_numpy(roi),
            source_loss_weight=torch.tensor(
                self.manifest["classification_loss_weights"][row["dataset"]]
            ),
        )
        return sample

    @staticmethod
    def collate_fn(batch):
        extra = [
            (row.pop("class_supervision"), row.pop("obstacle_roi"), row.pop("source_loss_weight"))
            for row in batch
        ]
        result = YOLODataset.collate_fn(batch)
        result.update(
            class_supervision=torch.stack([row[0] for row in extra]),
            obstacle_roi=torch.stack([row[1] for row in extra]),
            source_loss_weight=torch.stack([row[2] for row in extra]),
        )
        return result


def source_results(box, known_count):
    result = {}
    for source, predicate in (
        ("acdc", lambda x: x < known_count),
        ("lost_and_found", lambda x: x == known_count),
    ):
        rows = [index for index, cls in enumerate(box.ap_class_index) if predicate(int(cls))]
        aps = box.all_ap[rows] if rows else np.zeros((1, 10))
        result[source] = {"mAP50": float(aps[:, 0].mean()), "mAP50_95": float(aps.mean())}
    return result


class JointMetrics(DetMetrics):
    @property
    def results_dict(self):
        result = super().results_dict
        sources = source_results(self.box, self.known_count)
        result["fitness"] = sum(
            self.source_weights[source] * (0.1 * score["mAP50"] + 0.9 * score["mAP50_95"])
            for source, score in sources.items()
        )
        for source, scores in sources.items():
            result.update({f"{source}/{key}": value for key, value in scores.items()})
        return result


class DeploymentValidator(DetectionValidator):
    def postprocess(self, predictions):
        # YOLO.predict uses one class per anchor. Validation must use the same
        # class selection, rather than recover a lower-scoring second class.
        outputs = ops.non_max_suppression(
            predictions,
            self.args.conf,
            self.args.iou,
            nc=0,
            multi_label=False,
            agnostic=self.args.single_cls or self.args.agnostic_nms,
            max_det=self.args.max_det,
            end2end=self.end2end,
        )
        return [
            {"bboxes": value[:, :4], "conf": value[:, 4], "cls": value[:, 5], "extra": value[:, 6:]}
            for value in outputs
        ]


class JointValidator(DeploymentValidator):
    def init_metrics(self, model):
        self.manifest, self.index = load_joint_index(self.data)
        self.metrics = JointMetrics()
        self.metrics.known_count = self.manifest["known_class_count"]
        self.metrics.source_weights = self.manifest["source_weights"]
        super().init_metrics(model)

    def build_dataset(self, img_path, mode="val", batch=None):
        return JointDataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch or self.args.batch,
            augment=False,
            hyp=self.args,
            rect=True,
            stride=self.stride,
            pad=0.5,
            data=self.data,
            task="detect",
        )

    def _prepare_batch(self, si, batch):
        result = super()._prepare_batch(si, batch)
        result["joint_row"] = self.index[str(Path(batch["im_file"][si]).resolve())]
        return result

    def _prepare_pred(self, prediction, batch):
        result = super()._prepare_pred(prediction, batch)
        row = batch["joint_row"]
        supported = torch.tensor(row["supervised_classes"], device=result["cls"].device)
        keep = torch.isin(result["cls"].long(), supported)
        if row["roi_mask"] and len(result["cls"]):
            roi = native_roi(row, self.manifest)
            boxes = result["bboxes"].detach().cpu().numpy()
            centers = ((boxes[:, :2] + boxes[:, 2:]) / 2).astype(int)
            inside = (
                (centers[:, 0] >= 0)
                & (centers[:, 0] < roi.shape[1])
                & (centers[:, 1] >= 0)
                & (centers[:, 1] < roi.shape[0])
            )
            valid = np.zeros(len(boxes), dtype=bool)
            valid[inside] = roi[centers[inside, 1], centers[inside, 0]]
            keep &= torch.as_tensor(valid, device=keep.device)
        return {key: value[keep] for key, value in result.items()}


def joint_trainer(resume_state, foreground_exclusive=True):
    base = detection_trainer_for_resume(resume_state)

    class Trainer(base):
        def build_dataset(self, img_path, mode="train", batch=None):
            if mode == "train":
                # ROI must undergo the same geometric transform as the image.
                # This version allows colour/blur augmentation, not spatial mixing.
                for name in (
                    "mosaic",
                    "mixup",
                    "copy_paste",
                    "cutmix",
                    "degrees",
                    "translate",
                    "scale",
                    "shear",
                    "perspective",
                    "fliplr",
                    "flipud",
                    "multi_scale",
                ):
                    if getattr(self.args, name, 0):
                        raise ValueError(f"Joint ROI supervision requires {name}=0")
            return JointDataset(
                img_path=img_path,
                imgsz=self.args.imgsz,
                batch_size=batch or self.args.batch,
                augment=mode == "train",
                hyp=self.args,
                rect=mode != "train",
                stride=32,
                pad=0.0 if mode == "train" else 0.5,
                data=self.data,
                task="detect",
            )

        def get_model(self, cfg=None, weights=None, verbose=True):
            model = super().get_model(cfg, weights, verbose)
            manifest, _ = load_joint_index(self.data)
            preserve_classification_outputs(weights, model, manifest["known_class_count"])
            model.joint_detection = True
            model.joint_obstacle_class = manifest["known_class_count"]
            model.joint_foreground_exclusive = foreground_exclusive
            return model

        def set_model_attributes(self):
            super().set_model_attributes()
            self.model.criterion = JointDetectionLoss(
                self.model, self.model.joint_obstacle_class, self.model.joint_foreground_exclusive
            )

        def get_validator(self):
            self.loss_names = "box_loss", "cls_loss", "dfl_loss"
            return JointValidator(
                self.test_loader,
                save_dir=self.save_dir,
                args=copy(self.args),
                _callbacks=self.callbacks,
            )

    return Trainer
