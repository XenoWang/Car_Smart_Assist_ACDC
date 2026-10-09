"""Rain-focused fine-tuning with a frozen, multi-label weather teacher."""

from __future__ import annotations

import hashlib
import json
import logging
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from car_smart_assist.config.weather import WeatherConfig
from car_smart_assist.data.preprocessing import load_invalid_entries
from car_smart_assist.perception.weather import WeatherClassifier, select_device, visual_cues
from car_smart_assist.perception.weather_training import (
    FEATURE_NAMES,
    WeatherDataset,
    WeatherSample,
    _save_checkpoint,
    attribute_metrics,
    checkpoint_model_config,
    list_condition_images,
    list_pixel_accurate_images,
    split_by_sequence,
)

logger = logging.getLogger(__name__)


def source_of(sample: WeatherSample) -> str:
    return "acdc" if sample.path is not None else "pixel_accurate"


def enhanced_splits(cfg: WeatherConfig, root: Path) -> dict[str, list[WeatherSample]]:
    """Keep recording families/scenes disjoint; never read official test here."""
    t = cfg.train
    invalid = load_invalid_entries(root / t["acdc_cleaning_manifest"], root)
    records = [sample for split in ("train", "val") for sample in list_condition_images(
        root / t["acdc_root"], split, cfg.attributes, invalid)]
    train, val = split_by_sequence(records, float(t["validation_fraction"]), int(t["seed"]))
    fraction = float(t["calibration_fraction"]) / (1 - float(t["validation_fraction"]))
    train, calibration = split_by_sequence(train, fraction, int(t["seed"]) + 1)
    if t["use_pixel_accurate"]:
        val_scene, cal_scene = t["pixel_accurate_validation_scene"], t["pixel_accurate_calibration_scene"]
        if val_scene == cal_scene:
            raise ValueError("Pixel 验证和校准场景必须不同")
        pixel_train, pixel_val = list_pixel_accurate_images(
            root / t["pixel_accurate_zip"], cfg.attributes, val_scene,
            load_invalid_entries(root / t["pixel_cleaning_manifest"], root))
        pixel_cal = [s for s in pixel_train if s.group == f"pixel-accurate:scene{cal_scene}"]
        if not pixel_cal:
            raise ValueError("Pixel 校准场景没有样本")
        train.extend(s for s in pixel_train if s.group != f"pixel-accurate:scene{cal_scene}")
        val.extend(pixel_val)
        calibration.extend(pixel_cal)
    splits = {"train": train, "validation": val, "calibration": calibration}
    groups = [set(s.group for s in samples) for samples in splits.values()]
    if any(groups[i] & groups[j] for i in range(3) for j in range(i)):
        raise ValueError("天气训练／验证／校准存在重复录制组")
    return splits


class EnhancedDataset(Dataset):
    """Augment training rain images only; cues follow the augmented pixels."""

    def __init__(self, base: WeatherDataset, options: dict[str, Any]):
        self.base, self.options = base, options
        self.rain_index = base.cfg.attributes.index("rain")
        self.mask = np.ones_like(base.targets)
        if options["acdc_positive_only"]:
            for i, sample in enumerate(base.samples):
                if source_of(sample) == "acdc":
                    self.mask[i] = base.targets[i]

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        original, original_cues, labels = self.base[index]
        pixels, cues = original, original_cues
        if labels[self.rain_index] and torch.rand(()) < self.options["augmentation_probability"]:
            image = Image.fromarray(np.array(self.base.images[index], copy=True))
            for transform, key in ((ImageEnhance.Brightness, "brightness_range"),
                                   (ImageEnhance.Contrast, "contrast_range")):
                lo, hi = self.options[key]
                factor = lo + float(torch.rand(())) * (hi - lo)
                image = transform(image).enhance(factor)
            array = np.asarray(image, dtype=np.uint8)
            pixels = torch.from_numpy(np.array(array, copy=True)).permute(2, 0, 1).float() / 255
            values = visual_cues(array, self.base.cfg)
            cues = torch.tensor([values[name] for name in FEATURE_NAMES])
        return pixels, cues, labels, torch.from_numpy(self.mask[index].copy()), original, original_cues


def probabilities(model, dataset, device, batch_size):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    values = []
    model.eval()
    with torch.inference_mode():
        for images, cues, _ in loader:
            logits = model(images.to(device), cues.to(device))
            values.append(logits.sigmoid().cpu().numpy())
    result = np.concatenate(values)
    if not np.isfinite(result).all():
        raise ValueError("天气概率包含非有限值")
    return result


def metrics_by_source(dataset, values, cfg):
    result = {}
    for source in sorted({source_of(s) for s in dataset.samples}):
        indices = [i for i, s in enumerate(dataset.samples) if source_of(s) == source]
        metrics = attribute_metrics(dataset.targets[indices], values[indices], cfg.attributes,
                                    cfg.decision_thresholds)
        for j, name in enumerate(cfg.attributes):
            negatives = dataset.targets[indices, j] == 0
            predicted = values[indices, j] >= cfg.decision_thresholds[name]
            metrics["per_attribute"][name]["false_positive_rate"] = (
                float(predicted[negatives].mean()) if negatives.any() else 0.0)
        result[source] = metrics
    return result


def retained(candidate, baseline, options):
    """Evaluate preservation per source; absent positive classes use FP rate."""
    reasons = []
    for source, old in baseline.items():
        new = candidate[source]
        if new["macro_f1"] + options["max_source_macro_f1_drop"] + 1e-12 < old["macro_f1"]:
            reasons.append(f"{source}:macro_f1")
        for name in options["protected_attributes"]:
            a, b = new["per_attribute"][name], old["per_attribute"][name]
            if b["support"]:
                for metric, tolerance in (("f1", "max_f1_drop"), ("recall", "max_recall_drop")):
                    if a[metric] + options[tolerance] + 1e-12 < b[metric]:
                        reasons.append(f"{source}:{name}:{metric}")
            elif a["false_positive_rate"] > b["false_positive_rate"] + 1e-12:
                reasons.append(f"{source}:{name}:false_positive_rate")
    return not reasons, reasons


def rain_score(metrics):
    supported = [m["per_attribute"]["rain"]["f1"] for m in metrics.values()
                 if m["per_attribute"]["rain"]["support"]]
    return float(np.mean(supported)) if supported else 0.0


def distillation_loss(student, teacher, labels, mask, cfg, options):
    """Bernoulli KL; known wrong/uncertain teacher attributes contribute no loss."""
    p = teacher.sigmoid()
    thresholds = torch.tensor([cfg.decision_thresholds[name] for name in cfg.attributes], device=p.device)
    correct = (p >= thresholds) == (labels > 0.5)
    confident = torch.maximum(p, 1 - p) >= options["teacher_confidence"]
    valid = mask * correct * confident
    if options.get("distill_unlabeled", False):
        # 未标注属性只使用高置信原模型作软约束，不把它们伪装成已确认负标签。
        valid = valid + (1 - mask) * confident
    temperature = options["distillation_temperature"]
    soft = (teacher / temperature).sigmoid()
    cross_entropy = F.binary_cross_entropy_with_logits(student / temperature, soft, reduction="none")
    entropy = F.binary_cross_entropy_with_logits(teacher / temperature, soft, reduction="none")
    return ((cross_entropy - entropy) * valid).sum() / valid.sum().clamp_min(1) * temperature**2


def sampling_weights(dataset, teacher_values, cfg, options):
    keys = [(source_of(s), s.labels) for s in dataset.samples]
    counts = Counter(keys)
    weights = np.asarray([1 / counts[key] for key in keys], dtype=np.float64)
    index = cfg.attributes.index("rain")
    rain = dataset.targets[:, index] == 1
    hard = rain & (teacher_values[:, index] < cfg.decision_thresholds["rain"])
    weights[rain] *= options["rain_sampling_multiplier"]
    weights[hard] *= options["hard_rain_multiplier"]
    return torch.from_numpy(weights)


def validate_options(cfg):
    t, options = cfg.train, cfg.train["enhanced"]
    if cfg.rain_adapter_channels <= 0:
        raise ValueError("雨天增强训练需要 rain_adapter_channels > 0")
    for key in ("batch_size", "epochs", "patience", "resume_extra_epochs"):
        if type(t[key]) is not int or t[key] <= 0:
            raise ValueError(f"{key} 必须为正整数")
    if t["num_workers"] != 0:
        raise ValueError("增强天气训练目前要求 num_workers=0，以完整恢复增强随机状态")
    if not 0 < t["validation_fraction"] < 1 or not 0 < t["calibration_fraction"] < 1 - t["validation_fraction"]:
        raise ValueError("天气验证／校准比例无效")
    positive = ("rain_sampling_multiplier", "hard_rain_multiplier", "rain_loss_multiplier",
                "distillation_temperature")
    for key in (*positive, "distillation_weight", "max_f1_drop", "max_recall_drop", "max_source_macro_f1_drop"):
        if not np.isfinite(options[key]) or options[key] < 0 or (key in positive and options[key] == 0):
            raise ValueError(f"enhanced.{key} 无效")
    if not 0.5 <= options["teacher_confidence"] <= 1 or not 0 <= options["augmentation_probability"] <= 1:
        raise ValueError("原模型置信度／增强概率无效")
    for key in ("brightness_range", "contrast_range"):
        bounds = options[key]
        if len(bounds) != 2 or not np.isfinite(bounds).all() or not 0 < bounds[0] <= bounds[1]:
            raise ValueError(f"enhanced.{key} 无效")
    if set(options["protected_attributes"]) - set(cfg.attributes):
        raise ValueError("受保护天气属性无效")
    if not 0 <= t["label_smoothing"] < 1 or t["learning_rate"] <= 0 or t["weight_decay"] < 0:
        raise ValueError("天气优化器／平滑参数无效")


def train_enhanced_weather(cfg: WeatherConfig, root: Path, *, resume=True):
    validate_options(cfg)
    device = select_device(cfg.device)
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(cfg.train["seed"])
        return _train(cfg, root, device, resume)


def _train(cfg, root, device, resume):
    t, options = cfg.train, cfg.train["enhanced"]
    teacher_path, best_path = root / options["teacher_checkpoint"], root / cfg.checkpoint
    last_path = best_path.with_name("last.pt")
    if teacher_path.resolve() in (best_path.resolve(), last_path.resolve()):
        raise ValueError("增强候选不能覆盖原模型权重")
    teacher_hash = hashlib.sha256(teacher_path.read_bytes()).hexdigest()
    payload = torch.load(teacher_path, map_location="cpu", weights_only=True)
    teacher_cfg = replace(cfg, rain_adapter_channels=0)
    if payload.get("format_version") != 2 or payload["model_config"] != checkpoint_model_config(teacher_cfg):
        raise ValueError("蒸馏原模型必须是兼容的多标签天气模型")
    teacher = WeatherClassifier(teacher_cfg).to(device)
    teacher.load_state_dict(payload["model_state"])
    teacher.eval().requires_grad_(False)
    model = WeatherClassifier(cfg).to(device)
    restored = model.load_state_dict(payload["model_state"], strict=False)
    if restored.unexpected_keys or any(not name.startswith("rain_adapter.") for name in restored.missing_keys):
        raise ValueError("学生模型与原模型的原结构不兼容")
    model.requires_grad_(False)
    model.rain_adapter.requires_grad_(True)
    splits = enhanced_splits(cfg, root)
    sets = {name: WeatherDataset(samples, cfg, root / t["cache_dir"] / f"enhanced_{name}.npy")
            for name, samples in splits.items()}
    signatures = {name: dataset.signature for name, dataset in sets.items()}
    training = EnhancedDataset(sets["train"], options)
    for i, name in enumerate(cfg.attributes):
        known = training.mask[:, i] > 0
        if len(set(training.base.targets[known, i])) < 2:
            raise ValueError(f"增强训练的 {name} 缺少已标注正／负样本")
    teacher_train = probabilities(teacher, sets["train"], device, t["batch_size"])
    baseline = metrics_by_source(sets["validation"], probabilities(
        teacher, sets["validation"], device, t["batch_size"]), cfg)
    generator = torch.Generator().manual_seed(t["seed"])
    sampler = WeightedRandomSampler(sampling_weights(sets["train"], teacher_train, cfg, options),
                                    len(training), replacement=True, generator=generator)
    loader = DataLoader(training, batch_size=t["batch_size"], sampler=sampler, num_workers=0,
                        pin_memory=device.type == "cuda")
    positive = (training.base.targets * training.mask).sum(axis=0)
    negative = ((1 - training.base.targets) * training.mask).sum(axis=0)
    pos_weight = torch.as_tensor(np.minimum(negative / positive, t["pos_weight_cap"]),
                                 dtype=torch.float32, device=device)
    loss_weights = torch.ones(len(cfg.attributes), device=device)
    loss_weights[cfg.attributes.index("rain")] = options["rain_loss_multiplier"]
    optimizer = torch.optim.AdamW(model.rain_adapter.parameters(), lr=t["learning_rate"], weight_decay=t["weight_decay"])
    recipe = {key: value for key, value in t.items() if key not in ("epochs", "resume_extra_epochs", "patience")}
    recipe["decision_thresholds"] = cfg.decision_thresholds
    recipe["model_config"] = checkpoint_model_config(cfg)
    start, best_epoch, wait = 0, 0, 0
    best_score, best_metrics = rain_score(baseline), baseline
    history = []
    completed = False
    planned_epochs = t["epochs"]

    def save_best(epoch, metrics):
        _save_checkpoint({"format_version": 2, "model_config": checkpoint_model_config(cfg),
            "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "epoch": epoch, "val_metrics_by_source": metrics,
            "teacher_sha256": teacher_hash, "dataset_signatures": signatures,
            "decision_thresholds": cfg.decision_thresholds}, best_path)

    if resume and last_path.is_file():
        previous = torch.load(last_path, map_location="cpu", weights_only=True)
        if (previous["teacher_sha256"] != teacher_hash or previous["dataset_signatures"] != signatures
                or previous["recipe"] != recipe):
            raise ValueError("原模型／天气数据／增强策略变化，请使用 --fresh 开始新的增强运行")
        if not best_path.is_file() or hashlib.sha256(best_path.read_bytes()).hexdigest() != previous["best_sha256"]:
            raise ValueError("增强 last 引用的 best 权重缺失或被修改")
        model.load_state_dict(previous["model_state"])
        optimizer.load_state_dict(previous["optimizer_state"])
        generator.set_state(previous["shuffle_state"])
        torch.set_rng_state(previous["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(previous["cuda_rng"], device)
        start, best_epoch, best_score, wait = (previous[k] for k in ("epoch", "best_epoch", "best_score", "wait"))
        best_metrics, history = previous["best_metrics"], previous["history"]
        completed = previous.get("training_complete", wait >= t["patience"] or start >= t["epochs"])
        planned_epochs = previous.get("planned_epochs", t["epochs"])
    else:
        save_best(0, baseline)  # No admissible improvement keeps the original teacher weights.
    epochs = start + t["resume_extra_epochs"] if completed else planned_epochs
    if completed:
        wait = 0
    for epoch in range(start + 1, epochs + 1):
        # 骨干、BN 统计与原输出保持冻结，只有雨天残差分支参与更新。
        model.eval()
        model.rain_adapter.train()
        total = 0.0
        for images, cues, labels, mask, originals, original_cues in loader:
            images, cues, labels, mask = (x.to(device) for x in (images, cues, labels, mask))
            with torch.no_grad():
                teacher_logits = teacher(originals.to(device), original_cues.to(device))
            optimizer.zero_grad(set_to_none=True)
            logits = model(images, cues)
            smooth = labels * (1 - t["label_smoothing"]) + 0.5 * t["label_smoothing"]
            supervised = F.binary_cross_entropy_with_logits(logits, smooth, pos_weight=pos_weight, reduction="none")
            supervised = (supervised * mask * loss_weights).sum() / (mask * loss_weights).sum()
            loss = supervised + options["distillation_weight"] * distillation_loss(
                logits, teacher_logits, labels, mask, cfg, options)
            if not torch.isfinite(loss):
                raise ValueError("增强训练损失非有限值，停止写入检查点")
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
        metrics = metrics_by_source(sets["validation"], probabilities(
            model, sets["validation"], device, t["batch_size"]), cfg)
        admissible, reasons = retained(metrics, baseline, options)
        score = rain_score(metrics)
        improved = admissible and score > best_score + 1e-12
        if improved:
            best_epoch, best_score, best_metrics, wait = epoch, score, metrics, 0
            save_best(epoch, metrics)
        else:
            wait += 1
        row = {"epoch": epoch, "loss": total / len(loader), "rain_f1": score,
               "admissible": admissible, "rejected_by": reasons, "metrics": metrics}
        history.append(row)
        logger.info("天气增强 epoch %d/%d loss=%.4f rain-F1=%.4f 保持旧能力=%s best=%d",
                    epoch, epochs, row["loss"], score, admissible, best_epoch)
        _save_checkpoint({"format_version": 2, "model_config": checkpoint_model_config(cfg),
            "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "optimizer_state": optimizer.state_dict(), "epoch": epoch, "best_epoch": best_epoch,
            "planned_epochs": epochs, "training_complete": wait >= t["patience"] or epoch == epochs,
            "best_score": best_score, "best_metrics": best_metrics, "wait": wait, "history": history,
            "shuffle_state": generator.get_state(), "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
            "teacher_sha256": teacher_hash, "dataset_signatures": signatures, "recipe": recipe,
            "best_sha256": hashlib.sha256(best_path.read_bytes()).hexdigest()}, last_path)
        if wait >= t["patience"]:
            break
    report = {"checkpoint": str(best_path), "teacher_checkpoint": str(teacher_path),
        "teacher_sha256": teacher_hash, "best_epoch": best_epoch, "trained_through_epoch": history[-1]["epoch"],
        "accepted_improvement": best_epoch > 0, "device": str(device), "thresholds": cfg.decision_thresholds,
        "split_counts": {name: dict(Counter(source_of(s) for s in samples)) for name, samples in splits.items()},
        "split_groups": {name: sorted({s.group for s in samples}) for name, samples in splits.items()},
        "dataset_signatures": signatures, "teacher_validation": baseline, "best_validation": best_metrics,
        "history": history, "evaluation_limit": "旧原模型曾接触部分新验证／校准数据；此处是开发回归，不是完全独立泛化证明。官方 test 未用于本轮训练／选模。"}
    output = root / options["report_dir"] / "training_comparison.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if hashlib.sha256(teacher_path.read_bytes()).hexdigest() != teacher_hash:
        raise ValueError("增强训练期间原模型检查点被外部修改")
    return {k: v for k, v in report.items() if k not in ("history", "split_groups")}
