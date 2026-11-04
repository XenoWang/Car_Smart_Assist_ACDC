"""Optional center-assisted box regression, in input pixels, for joint YOLO.

Adapted from arXiv:2603.03788v1 Eq.8: 1-exp(-EuclideanDistance/C).
The paper leaves average object size unspecified. Here C is the train-only mean
sqrt(box area) after the existing square training resize/letterbox.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from car_smart_assist.config.detection import center_loss_config


def validate_center_recipe(recipe):
    if not isinstance(recipe, dict) or type(recipe.get("enabled")) is not bool:
        raise ValueError("Center loss resume recipe must be a mapping")
    options = center_loss_config({key: recipe[key] for key in ("enabled", "weight", "normalization") if key in recipe})
    if not options["enabled"]:
        return options
    if (type(recipe.get("version")) is not int or recipe["version"] != 1
            or type(recipe.get("imgsz")) is not int or recipe["imgsz"] <= 0
            or type(recipe.get("ground_truth_count")) is not int or recipe["ground_truth_count"] <= 0
            or not isinstance(recipe.get("training_data_signature"), str) or not recipe["training_data_signature"]):
        raise ValueError("Center loss resume recipe lacks train-only normalization identity")
    scale = recipe.get("scale_pixels")
    try:
        valid_scale = type(scale) in (int, float) and math.isfinite(scale) and scale > 0
    except OverflowError:
        valid_scale = False
    if not valid_scale:
        raise ValueError("Center loss scale_pixels must be finite and positive")
    expected = {"version", "enabled", "weight", "normalization", "scale_pixels", "ground_truth_count", "imgsz", "training_data_signature"}
    if set(recipe) != expected:
        raise ValueError("Center loss resume recipe has missing or unknown fields")
    return {**recipe, **options, "scale_pixels": float(scale)}


def require_center_recipe(record, options, *, resolved=None):
    """Legacy states without the optional objective are compatible only when off."""
    wanted = center_loss_config(options)
    saved = validate_center_recipe(record.get("center_loss", {"enabled": False}))
    saved_options = center_loss_config({key: saved[key] for key in ("enabled", "weight", "normalization") if key in saved})
    if saved_options != wanted:
        raise ValueError("Center loss configuration changed or missing; use --fresh or a new run directory")
    if resolved is not None and saved != validate_center_recipe(resolved):
        raise ValueError("Center loss training normalization changed; use --fresh or a new run directory")
    return saved


def train_center_recipe(labels, imgsz, data_signature, options):
    """Consume only the actual training dataset labels, never validation batches."""
    options = center_loss_config(options)
    if not options["enabled"]:
        return options
    if type(imgsz) is not int or imgsz <= 0:
        raise ValueError("Center loss requires a positive square training imgsz")
    sizes = []
    for label in labels:
        height, width = label["shape"]
        if min(height, width) <= 0 or not all(math.isfinite(v) for v in (height, width)):
            raise ValueError("Center loss requires valid training image shapes")
        if label.get("bbox_format", "xywh") != "xywh" or label.get("normalized", True) is not True:
            raise ValueError("Center loss requires normalized xywh training labels")
        ratio = imgsz / max(height, width)
        resized_w = min(math.ceil(width * ratio), imgsz)
        resized_h = min(math.ceil(height * ratio), imgsz)
        for box in label["bboxes"]:
            if len(box) != 4 or not all(math.isfinite(float(v)) for v in box):
                raise ValueError("Center loss requires finite xywh training boxes")
            x, y, w, h = map(float, box)
            if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                raise ValueError("Center loss requires positive normalized training box sizes")
            sizes.append(math.sqrt((w * resized_w) * (h * resized_h)))
    if not sizes:
        raise ValueError("Center loss requires at least one labeled training box")
    return validate_center_recipe({
        "version": 1, **options, "scale_pixels": math.fsum(sizes) / len(sizes),
        "ground_truth_count": len(sizes), "imgsz": imgsz,
        "training_data_signature": data_signature,
    })


def center_distance_loss(pred_boxes, target_boxes, strides, target_weights, score_sum, scale_pixels):
    """Assigned xyxy grid boxes -> target-score-weighted bounded pixel loss."""
    # Keep the geometry in FP32 under AMP; vector_norm has zero gradient at zero.
    predicted, target = pred_boxes.float(), target_boxes.float()
    delta = ((predicted[..., :2] + predicted[..., 2:])
             - (target[..., :2] + target[..., 2:])) * 0.5
    distance = torch.linalg.vector_norm(delta * strides.float(), dim=-1)
    bounded = -torch.expm1(-distance / scale_pixels)
    return (bounded * target_weights.float()).sum() / score_sum


class CenterAssistedBboxLoss(nn.Module):
    """Add the optional term to the vendor's box component, preserving its DFL."""

    def __init__(self, base, recipe):
        super().__init__()
        self.base = base
        self.recipe = validate_center_recipe(recipe)
        if not self.recipe["enabled"]:
            raise ValueError("CenterAssistedBboxLoss requires an enabled recipe")
        self.anchor_strides = None

    def forward(self, pred_dist, pred_boxes, anchors, targets, scores, score_sum, foreground):
        box, dfl = self.base(pred_dist, pred_boxes, anchors, targets, scores, score_sum, foreground)
        if self.recipe["weight"] == 0:
            return box, dfl
        if self.anchor_strides is None or self.anchor_strides.shape != (pred_boxes.shape[1], 1):
            raise ValueError("Center loss requires per-anchor pixel strides")
        strides = self.anchor_strides.unsqueeze(0).expand(pred_boxes.shape[0], -1, -1)[foreground]
        auxiliary = center_distance_loss(
            pred_boxes[foreground], targets[foreground], strides,
            scores.sum(-1)[foreground], score_sum, self.recipe["scale_pixels"],
        )
        return box + self.recipe["weight"] * auxiliary, dfl
