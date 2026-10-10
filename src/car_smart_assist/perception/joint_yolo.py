"""BCE 按数据来源选择监督范围；未标注的类别和 ROI 不作为负样本。"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class MaskedClassificationLoss(nn.Module):
    def __init__(self, foreground_exclusive=True):
        super().__init__()
        self.valid = None
        self.weights = None
        self.foreground_exclusive = foreground_exclusive

    def forward(self, predictions, targets):
        if self.valid is None:
            raise ValueError("Joint detection requires explicit supervision coverage")
        # 正样本分配可能落在稀疏像素 ROI 之外，这种情况要保留它的目标。
        positives = targets > 0
        if getattr(self, "foreground_exclusive", False):
            # 在已标注的目标上，这九类互斥：
            # 常规交通类别与已标注的杂类货物。
            # 缺少背景标注时仍然不产生类别负样本。
            positives = positives.any(dim=-1, keepdim=True)
        allowed = self.valid | positives
        return (
            F.binary_cross_entropy_with_logits(predictions, targets, reduction="none")
            * allowed
            * self.weights
        )


class JointDetectionLoss:
    def __init__(self, model, obstacle_class, foreground_exclusive=True):
        from ultralytics.utils.loss import v8DetectionLoss

        self.base = v8DetectionLoss(model)
        self.base.bce = MaskedClassificationLoss(foreground_exclusive)
        self.obstacle_class = obstacle_class

    def __call__(self, predictions, batch):
        features = predictions[1] if isinstance(predictions, tuple) else predictions
        device = features[0].device
        coverage = batch["class_supervision"].to(device=device, dtype=torch.bool)
        roi = batch["obstacle_roi"].to(device=device, dtype=torch.bool)
        sampled = []
        for feature in features:
            height, width = feature.shape[-2:]
            y = ((torch.arange(height, device=device) + 0.5) * roi.shape[-2] / height).long()
            x = ((torch.arange(width, device=device) + 0.5) * roi.shape[-1] / width).long()
            sampled.append(roi[:, y[:, None], x[None, :]].flatten(1))
        anchors = torch.cat(sampled, dim=1)
        valid = coverage[:, None, :].expand(-1, anchors.shape[1], -1).clone()
        valid[..., self.obstacle_class] &= anchors
        self.base.bce.valid = valid
        self.base.bce.weights = batch["source_loss_weight"].to(device)[:, None, None]
        try:
            return self.base(predictions, batch)
        finally:
            self.base.bce.valid = self.base.bce.weights = None


def preserve_classification_outputs(source, target, known_classes):
    """加第九类时要保留原来八类的输出权重行，不能重新初始化。"""
    if source is None or source.model[-1].nc != known_classes:
        return 0
    copied = 0
    with torch.no_grad():
        for old, new in zip(source.model[-1].cv3, target.model[-1].cv3, strict=True):
            old_layer, new_layer = old[-1], new[-1]
            if old_layer.weight.shape[1:] != new_layer.weight.shape[1:]:
                raise ValueError(
                    "Classifier channel shape changed; cannot preserve original classes"
                )
            new_layer.weight[:known_classes].copy_(old_layer.weight)
            new_layer.bias[:known_classes].copy_(old_layer.bias)
            copied += 1
    return copied
