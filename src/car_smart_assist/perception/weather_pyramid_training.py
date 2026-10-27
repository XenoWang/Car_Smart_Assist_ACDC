"""同一四分数据、初始化和训练预算，比较 Focal、ASL 与旧天气模型。"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import shutil
import time
from collections import defaultdict
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, WeightedRandomSampler

from car_smart_assist.config.weather import load_weather_config
from car_smart_assist.utils.stop import StopController
from car_smart_assist.config.weather_pyramid import WeatherPyramidConfig, pyramid_model_config
from car_smart_assist.data.weather_pyramid import (
    PyramidDataset,
    prepare_pyramid_splits,
    pyramid_sampling_weights,
    sample_identifier,
    weather_source,
)
from car_smart_assist.perception.weather import WeatherPredictor, select_device
from car_smart_assist.perception.weather_pyramid import (
    WeatherPyramid,
    WeatherPyramidPredictor,
    masked_weather_loss,
)
from car_smart_assist.perception.weather_training import (
    WeatherDataset,
    _acdc_split,
    _save_checkpoint,
)

logger = logging.getLogger(__name__)


def state_hash(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def masked_metrics(labels, mask, values, attributes, thresholds):
    result = {"samples": len(labels), "per_attribute": {}}
    for index, name in enumerate(attributes):
        known = mask[:, index] > 0
        truth, probability = labels[known, index] > 0.5, values[known, index]
        # 与部署中 Python float 概率/阈值的比较保持一致。
        prediction = probability.astype(np.float64, copy=False) >= float(thresholds[name])
        tp, fp = int((truth & prediction).sum()), int((~truth & prediction).sum())
        fn, tn = int((truth & ~prediction).sum()), int((~truth & ~prediction).sum())
        both = bool(truth.any() and (~truth).any())
        result["per_attribute"][name] = {
            "known_positive": tp + fn,
            "known_negative": fp + tn,
            "unknown": int((~known).sum()),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "threshold": thresholds[name],
            "recall": tp / (tp + fn) if tp + fn else None,
            "precision": tp / (tp + fp) if both and tp + fp else (0.0 if both else None),
            "f1": 2 * tp / (2 * tp + fp + fn) if both else None,
            "false_positive_rate": fp / (fp + tn) if fp + tn else None,
            "average_precision": float(average_precision_score(truth, probability))
            if both
            else None,
            "unverified_positive_rate": float(
                (values[~known, index].astype(np.float64, copy=False) >= float(thresholds[name])).mean()
            )
            if (~known).any()
            else None,
        }
    for metric, output in (("f1", "macro_f1"), ("average_precision", "macro_ap")):
        valid = [row[metric] for row in result["per_attribute"].values() if row[metric] is not None]
        result[output] = float(np.mean(valid)) if valid else None
    return result


def grouped_metrics(dataset, values, cfg, thresholds):
    result = masked_metrics(dataset.targets, dataset.mask, values, cfg.attributes, thresholds)
    result["by_source"] = {}
    for source in sorted({weather_source(s) for s in dataset.samples}):
        indices = [i for i, s in enumerate(dataset.samples) if weather_source(s) == source]
        result["by_source"][source] = masked_metrics(
            dataset.targets[indices],
            dataset.mask[indices],
            values[indices],
            cfg.attributes,
            thresholds,
        )
    contexts = defaultdict(list)
    for index, sample in enumerate(dataset.samples):
        if sample.path is None and sample.labels[cfg.attributes.index("rain")]:
            contexts[sample.stratum].append(index)
    result["rain_contexts"] = {
        name: masked_metrics(
            dataset.targets[idx], dataset.mask[idx], values[idx], cfg.attributes, thresholds
        )["per_attribute"]["rain"]
        for name, idx in sorted(contexts.items())
    }
    return result


def calibrate_thresholds(dataset, probabilities, cfg):
    thresholds = {}
    grid = np.linspace(0, 1, cfg.evaluation["threshold_steps"] + 2)[1:-1]
    for index, name in enumerate(cfg.attributes):
        known = dataset.mask[:, index] > 0
        labels = dataset.targets[known, index] > 0.5
        values = probabilities[known, index]
        if not labels.any() or labels.all():
            raise ValueError(f"校准集的 {name} 缺少已确认正／负标签")
        scores = []
        for threshold in grid:
            predicted = values >= threshold
            tp = int((labels & predicted).sum())
            scores.append(2 * tp / (int(labels.sum()) + int(predicted.sum())))
        maximum = max(scores)
        optimal = [t for t, score in zip(grid, scores, strict=True) if abs(score - maximum) < 1e-12]
        # 避免把宽阈值平台推到边缘；仍只使用校准标签。
        selected = (
            max(optimal)
            if cfg.evaluation["threshold_tie_break"] == "highest"
            else min(optimal, key=lambda t: (abs(t - float(np.median(optimal))), t))
        )
        thresholds[name] = float(selected)
    return thresholds


def _loader(dataset, cfg, device, *, sampler=None, generator=None):
    return DataLoader(
        dataset,
        batch_size=cfg.train["batch_size"] if sampler is not None else cfg.evaluation["batch_size"],
        sampler=sampler,
        num_workers=cfg.train["num_workers"],
        generator=generator,
        pin_memory=device.type == "cuda",
    )


def pyramid_probabilities(model, dataset, cfg, device):
    outputs = []
    model.eval()
    with torch.inference_mode():
        for images, _, _ in _loader(dataset, cfg, device):
            images = images.to(device, dtype=torch.float32, non_blocking=True).div_(255)
            outputs.append(model(images).float().sigmoid().cpu().numpy())
    values = np.concatenate(outputs)
    if not np.isfinite(values).all():
        raise ValueError("新天气模型概率非有限")
    return values


def _augment(images, options):
    batch = len(images)
    flip = torch.rand(batch, device=images.device) < options["horizontal_flip_probability"]
    images = torch.where(flip[:, None, None, None], images.flip(-1), images)
    for key, center in (("brightness_range", 0), ("contrast_range", None)):
        low, high = options[key]
        factor = low + (high - low) * torch.rand((batch, 1, 1, 1), device=images.device)
        midpoint = images.mean(dim=(1, 2, 3), keepdim=True) if center is None else center
        images = ((images - midpoint) * factor + midpoint).clamp(0, 1)
    return images


def _selection_score(metrics, cfg):
    total, weight_sum = 0.0, 0.0
    for name, row in metrics["per_attribute"].items():
        if row["average_precision"] is None:
            raise ValueError(f"验证集的 {name} 缺少已确认正／负标签")
        weight = cfg.train["rain_selection_weight"] if name == "rain" else 1
        total += weight * row["average_precision"]
        weight_sum += weight
    return total / weight_sum


def _train_variant(cfg, root, datasets, split_info, name, device, fresh, total_epochs=None, stop_controller=None, continue_only=False):
    from car_smart_assist.perception.weather_training import require_independent_weather_output

    output = root / cfg.checkpoint_root / name
    require_independent_weather_output(root, output)
    best_path, last_path = output / "best.pt", output / "last.pt"
    if not fresh and best_path.exists() and not last_path.exists():
        raise ValueError("已有 best 但缺少配对 last；保留原件，使用 --fresh 或独立输出目录")
    stop_controller = stop_controller or StopController(default_directory=output)
    stop_controller.watch_directory(output)
    stop_controller.require_running()
    if fresh and output.exists():
        files = [p for p in (best_path, last_path) if p.exists()]
        if files:
            history = output / "history" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            history.mkdir(parents=True)
            for path in files:
                shutil.copy2(path, history / path.name)
    torch.manual_seed(cfg.data["seed"])
    model = WeatherPyramid(cfg).to(device)
    initial_hash = state_hash(model)
    generator = torch.Generator().manual_seed(cfg.data["seed"])
    sampler = WeightedRandomSampler(
        pyramid_sampling_weights(datasets["train"]),
        len(datasets["train"]),
        replacement=True,
        generator=generator,
    )
    loader = _loader(datasets["train"], cfg, device, sampler=sampler, generator=generator)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.train["learning_rate"], weight_decay=cfg.train["weight_decay"]
    )
    amp_enabled = device.type == "cuda" and cfg.train["amp"]
    amp_choice = cfg.train["amp_dtype"]
    use_bfloat16 = amp_enabled and (
        amp_choice == "bfloat16" or (amp_choice == "auto" and torch.cuda.is_bf16_supported())
    )
    if use_bfloat16 and not torch.cuda.is_bf16_supported():
        raise ValueError("当前 CUDA 设备不支持请求的 BF16")
    amp_dtype = torch.bfloat16 if use_bfloat16 else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and not use_bfloat16)
    logger.info("%s 设备=%s 混合精度=%s", name, device, str(amp_dtype) if amp_enabled else "关闭")
    recipe = {
        "model": pyramid_model_config(cfg),
        "loss_name": name,
        "loss": cfg.loss,
        "train": {
            k: v
            for k, v in cfg.train.items()
            if k not in ("epochs", "resume_extra_epochs", "force_epochs")
        },
        "dataset_signatures": {key: ds.signature for key, ds in datasets.items()},
        "split_signature": split_info["signature"],
        "seed": cfg.data["seed"],
    }
    epoch, wait, best_epoch, best_score, history = 0, 0, 0, -1.0, []
    planned_epochs = cfg.train["epochs"]
    force_epochs = cfg.train["force_epochs"]
    completed_stage = False
    if continue_only and (fresh or not last_path.is_file()):
        raise ValueError("--continue 需要本训练输出目录的配对 last.pt")
    if not fresh and last_path.exists():
        previous = torch.load(last_path, map_location="cpu", weights_only=True)
        if previous["recipe"] != recipe:
            raise ValueError(f"{name} 数据或训练配置已改变，请用 --fresh 开新实验")
        if (
            not best_path.exists()
            or hashlib.sha256(best_path.read_bytes()).hexdigest() != previous["best_sha256"]
        ):
            raise ValueError(f"{name} last 引用的 best 已丢失或改变")
        model.load_state_dict(previous["model_state"])
        optimizer.load_state_dict(previous["optimizer_state"])
        scaler.load_state_dict(previous["scaler_state"])
        generator.set_state(previous["generator_state"])
        torch.set_rng_state(previous["torch_rng"])
        if device.type == "cuda" and previous["cuda_rng"] is not None:
            torch.cuda.set_rng_state(previous["cuda_rng"], device)
        epoch, wait, best_epoch, best_score, history, initial_hash = (
            previous[key]
            for key in (
                "epoch",
                "wait",
                "best_epoch",
                "best_score",
                "history",
                "initial_state_sha256",
            )
        )
        planned_epochs = previous["planned_epochs"]
        force_epochs = previous.get("force_epochs", force_epochs)
        completed_stage = continue_only and bool(previous["training_complete"])
        if previous["training_complete"] and not continue_only:
            planned_epochs = epoch + cfg.train["resume_extra_epochs"]
            force_epochs = cfg.train["force_epochs"]
            wait = 0
    if total_epochs is not None:
        if continue_only and total_epochs != planned_epochs:
            raise ValueError("--continue 不允许改变原累计训练目标")
        if type(total_epochs) is not int or total_epochs <= 0 or total_epochs < epoch:
            raise ValueError("累计训练目标必须为正整数，且不能小于已完成轮数")
        if not continue_only:
            planned_epochs, force_epochs = total_epochs, True
            wait = 0
    logger.info(
        "%s 恢复轮数=%d 累计目标=%d 提前早停=%s", name, epoch, planned_epochs, not force_epochs
    )
    defaults = dict.fromkeys(cfg.attributes, 0.5)
    started = time.perf_counter()
    remaining = () if completed_stage else range(epoch + 1, planned_epochs + 1)
    for current in remaining:
        if stop_controller.requested:
            break
        model.train()
        total, batches = 0.0, 0
        for images, labels, mask in loader:
            images = _augment(
                images.to(device, dtype=torch.float32, non_blocking=True).div_(255), cfg.train
            )
            labels, mask = labels.to(device, non_blocking=True), mask.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp_enabled, dtype=amp_dtype):
                logits = model(images)
                loss = masked_weather_loss(logits, labels, mask, name, cfg)
            if not torch.isfinite(loss):
                raise ValueError("天气损失非有限；未覆盖检查点")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                cfg.train["gradient_clip"],
                error_if_nonfinite=not scaler.is_enabled(),
            )
            if scaler.is_enabled() and not torch.isfinite(gradient_norm):
                logger.warning("%s FP16梯度溢出：GradScaler跳过本次更新并降低缩放", name)
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.detach())
            batches += 1
        probabilities = pyramid_probabilities(model, datasets["validation"], cfg, device)
        metrics = grouped_metrics(datasets["validation"], probabilities, cfg, defaults)
        score = _selection_score(metrics, cfg)
        if score > best_score + 1e-12:
            best_score, best_epoch, wait = score, current, 0
            _save_checkpoint(
                {
                    "format_version": 3,
                    "model_config": pyramid_model_config(cfg),
                    "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "epoch": current,
                    "loss_name": name,
                    "validation_metrics": metrics,
                    "split_signature": split_info["signature"],
                    "initial_state_sha256": initial_hash,
                    "decision_thresholds": defaults,
                },
                best_path,
            )
        else:
            wait += 1
        complete = current == planned_epochs or (
            not force_epochs
            and current >= cfg.train["min_epochs"]
            and wait >= cfg.train["patience"]
        )
        row = {
            "epoch": current,
            "loss": total / batches,
            "selection_score": score,
            "validation_macro_ap": metrics["macro_ap"],
            "rain_ap": metrics["per_attribute"]["rain"]["average_precision"],
            "best_epoch": best_epoch,
            "metrics": metrics,
        }
        history.append(row)
        logger.info(
            "%s %d/%d loss=%.4f val-mAP=%.4f rain-AP=%.4f best=%d elapsed=%.0fs",
            name,
            current,
            planned_epochs,
            row["loss"],
            row["validation_macro_ap"],
            row["rain_ap"],
            best_epoch,
            time.perf_counter() - started,
        )
        _save_checkpoint(
            {
                "format_version": 3,
                "model_config": pyramid_model_config(cfg),
                "recipe": recipe,
                "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "optimizer_state": optimizer.state_dict(),
                "scaler_state": scaler.state_dict(),
                "epoch": current,
                "planned_epochs": planned_epochs,
                "force_epochs": force_epochs,
                "training_complete": complete,
                "termination_reason": "budget_completed" if current == planned_epochs else (
                    "early_stopping" if complete else None
                ),
                "manual_stop": False,
                "wait": wait,
                "best_epoch": best_epoch,
                "best_score": best_score,
                "history": history,
                "initial_state_sha256": initial_hash,
                "generator_state": generator.get_state(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
                "best_sha256": hashlib.sha256(best_path.read_bytes()).hexdigest(),
            },
            last_path,
        )
        if stop_controller.requested or complete:
            break
    def pause_if_requested():
        if not stop_controller.requested:
            return None
        if not last_path.is_file():
            stop_controller.require_running()
        last = torch.load(last_path, map_location="cpu", weights_only=True)
        last.update(termination_reason="manual_stop", manual_stop=True)
        _save_checkpoint(last, last_path)
        return model.eval(), {
            "checkpoint": str(best_path), "best_epoch": best_epoch,
            "trained_through_epoch": last["epoch"],
            "training_complete": last["training_complete"],
            "manual_stop": True, "termination_reason": "manual_stop",
            "history": history, "initial_state_sha256": initial_hash,
            "training_seconds": time.perf_counter() - started, "calibrated": False,
        }
    paused = pause_if_requested()
    if paused is not None:
        return paused
    payload = torch.load(best_path, map_location="cpu", weights_only=True)
    model.load_state_dict(payload["model_state"])
    values = pyramid_probabilities(model, datasets["calibration"], cfg, device)
    paused = pause_if_requested()
    if paused is not None:
        return paused
    thresholds = calibrate_thresholds(datasets["calibration"], values, cfg)
    payload.update(
        {
            "decision_thresholds": thresholds,
            "calibration_signature": datasets["calibration"].signature,
            "calibration_metrics": grouped_metrics(
                datasets["calibration"], values, cfg, thresholds
            ),
        }
    )
    _save_checkpoint(payload, best_path)
    previous = torch.load(last_path, map_location="cpu", weights_only=True)
    previous["best_sha256"] = hashlib.sha256(best_path.read_bytes()).hexdigest()
    if continue_only and previous["training_complete"]:
        previous.update(manual_stop=False, termination_reason=(
            "budget_completed" if previous["epoch"] == planned_epochs else "early_stopping"
        ))
    _save_checkpoint(previous, last_path)
    paused = pause_if_requested()
    if paused is not None:
        return paused
    report = {
        "checkpoint": str(best_path),
        "best_epoch": best_epoch,
        "trained_through_epoch": history[-1]["epoch"],
        "training_complete": previous["training_complete"],
        "termination_reason": previous.get("termination_reason") or (
            "budget_completed" if previous["epoch"] == planned_epochs else "early_stopping"
        ),
        "manual_stop": False,
        "thresholds": thresholds,
        "initial_state_sha256": initial_hash,
        "parameters": sum(p.numel() for p in model.parameters()),
        "validation": grouped_metrics(
            datasets["validation"],
            pyramid_probabilities(model, datasets["validation"], cfg, device),
            cfg,
            defaults,
        ),
        "calibration": payload["calibration_metrics"],
        "history": history,
        "training_seconds": time.perf_counter() - started,
    }
    paused = pause_if_requested()
    if paused is not None:
        return paused
    return model.eval(), report


def train_pyramid_variant(cfg, root, datasets, split_info, name, *, fresh=False, total_epochs=None, stop_controller=None, continue_only=False):
    device = select_device(cfg.device)
    devices = (
        [device.index if device.index is not None else torch.cuda.current_device()]
        if device.type == "cuda"
        else []
    )
    controller = stop_controller or StopController(default_directory=root / cfg.checkpoint_root / name)
    controller.watch_directory(root / cfg.checkpoint_root / name)
    with controller, torch.random.fork_rng(devices=devices):
        controller.require_running()
        return _train_variant(cfg, root, datasets, split_info, name, device, fresh, total_epochs, controller, continue_only)


def benchmark_model(model, image, cfg, *, cues=None):
    results = {}
    old_threads, original_device = torch.get_num_threads(), next(model.parameters()).device
    try:
        torch.set_num_threads(cfg.evaluation["cpu_threads"])
        for name in ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",):
            device = torch.device(name)
            model.to(device).eval()
            pixels = image.to(device)
            cue_tensor = cues.to(device) if cues is not None else None
            with torch.inference_mode():
                for _ in range(cfg.evaluation["benchmark_warmup"]):
                    model(pixels, cue_tensor) if cues is not None else model(pixels)
                if name == "cuda":
                    torch.cuda.synchronize()
                started = time.perf_counter()
                for _ in range(cfg.evaluation["benchmark_iterations"]):
                    model(pixels, cue_tensor) if cues is not None else model(pixels)
                if name == "cuda":
                    torch.cuda.synchronize()
            results[name] = (
                (time.perf_counter() - started) * 1000 / cfg.evaluation["benchmark_iterations"]
            )
    finally:
        torch.set_num_threads(old_threads)
        model.to(original_device)
    return results


def _old_probabilities(predictor, dataset, batch_size):
    outputs = []
    with torch.inference_mode():
        for images, cues, _ in DataLoader(dataset, batch_size=batch_size, shuffle=False):
            outputs.append(
                predictor.model(images.to(predictor.device), cues.to(predictor.device))
                .float()
                .sigmoid()
                .cpu()
                .numpy()
            )
    result = np.concatenate(outputs)
    if not np.isfinite(result).all():
        raise ValueError("旧天气概率非有限")
    return result


def compare_pyramid_models(
    cfg: WeatherPyramidConfig,
    root: Path,
    *,
    fresh=False,
    variants=("focal", "asl"),
    evaluate_only=False,
    total_epochs=None,
    stop_controller=None,
    continue_only=False,
):
    stop_controller = stop_controller or StopController(default_directory=root / cfg.checkpoint_root)
    stop_controller.require_running()
    old_path = root / cfg.evaluation["old_checkpoint"]
    old_hash = hashlib.sha256(old_path.read_bytes()).hexdigest()
    splits, split_info = prepare_pyramid_splits(cfg, root)
    datasets = {
        name: PyramidDataset(samples, cfg, root / cfg.data["cache_dir"])
        for name, samples in splits.items()
    }
    device = select_device(cfg.device)
    models, reports = {}, {}
    state_path = root / cfg.checkpoint_root / "training_progress.json"
    identity = json.loads(json.dumps({"variants": list(variants), "split_signature": split_info["signature"],
                                      "model": pyramid_model_config(cfg), "train": cfg.train, "loss": cfg.loss}))
    if continue_only:
        if not state_path.is_file():
            raise ValueError("--continue 缺少金字塔组状态，不能推测缺失 last 是未开始")
        suite = json.loads(state_path.read_text(encoding="utf-8"))
        if suite.get("identity") != identity:
            raise ValueError("金字塔组配置、划分或原目标已改变")
    else:
        if fresh and state_path.is_file():
            archive = state_path.parent / "history" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            archive.mkdir(parents=True)
            original = state_path.read_bytes()
            shutil.copy2(state_path, archive / state_path.name)
            if (archive / state_path.name).read_bytes() != original:
                raise OSError("金字塔组进度归档回读不一致")
        suite = {"identity": identity, "groups": {n: {"status": "pending"} for n in variants}}

    def record_group(name, status, report=None):
        row = {"status": status}
        if report is not None:
            row["report"] = report
            row["last_sha256"] = hashlib.sha256(
                (root / cfg.checkpoint_root / name / "last.pt").read_bytes()
            ).hexdigest()
            row["best_sha256"] = hashlib.sha256(
                (root / cfg.checkpoint_root / name / "best.pt").read_bytes()
            ).hexdigest()
        suite["groups"][name] = row
        state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(suite, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(state_path)

    for name in variants:
        stop_controller.require_running()
        if evaluate_only:
            path = root / cfg.checkpoint_root / name / "best.pt"
            predictor = WeatherPyramidPredictor.from_checkpoint(path, cfg)
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if (
                payload["split_signature"] != split_info["signature"]
                or payload.get("calibration_signature") != datasets["calibration"].signature
            ):
                raise ValueError("天气模型的数据划分或校准数据已改变，不能复用旧测试结果")
            models[name] = predictor.model
            reports[name] = {
                "checkpoint": str(path),
                "best_epoch": payload["epoch"],
                "thresholds": predictor.thresholds,
                "initial_state_sha256": payload["initial_state_sha256"],
                "parameters": sum(p.numel() for p in predictor.model.parameters()),
            }
        else:
            row = suite["groups"].get(name, {})
            group_dir = root / cfg.checkpoint_root / name
            group_continue = continue_only and row.get("status") != "pending"
            if continue_only:
                if row.get("status") not in {"pending", "running", "paused", "completed"}:
                    raise ValueError(f"{name} 的原组进度缺失或无效")
                if row.get("status") == "pending":
                    if any((group_dir / item).exists() for item in ("last.pt", "best.pt")):
                        raise ValueError(f"{name} 标记未开始但已有检查点，拒绝猜测恢复")
                elif not all((group_dir / item).is_file() for item in ("last.pt", "best.pt")):
                    raise ValueError(f"{name} 缺少已开始组的配对 last/best")
                if row.get("status") == "completed":
                    for item, key in (("last.pt", "last_sha256"), ("best.pt", "best_sha256")):
                        if hashlib.sha256((group_dir / item).read_bytes()).hexdigest() != row.get(key):
                            raise ValueError(f"{name} 已完成检查点身份已改变")
                    reports[name] = row["report"]
                    models[name] = WeatherPyramidPredictor.from_checkpoint(group_dir / "best.pt", cfg).model
                    continue
            record_group(name, "running")
            models[name], reports[name] = train_pyramid_variant(
                cfg, root, datasets, split_info, name, fresh=fresh, total_epochs=total_epochs,
                stop_controller=stop_controller, continue_only=group_continue
            )
            if reports[name].get("manual_stop") or stop_controller.requested:
                record_group(name, "paused")
                return {"status": "手动停止", "manual_stop": True,
                        "termination_reason": "manual_stop", "models": reports}
            record_group(name, "completed", reports[name])
    stop_controller.require_running()
    if len({r["initial_state_sha256"] for r in reports.values()}) > 1:
        raise ValueError("Focal 和 ASL 初始参数不一致")
    old_cfg = replace(load_weather_config(root / cfg.evaluation["old_config"]), device=cfg.device)
    old = WeatherPredictor.from_checkpoint(old_path, old_cfg)
    old_sets = {
        name: WeatherDataset(
            splits[name], old_cfg, root / cfg.data["cache_dir"] / f"old_{name}.npy"
        )
        for name in ("calibration", "test")
    }
    old_calibration = _old_probabilities(old, old_sets["calibration"], cfg.evaluation["batch_size"])
    old_thresholds = calibrate_thresholds(datasets["calibration"], old_calibration, cfg)
    test_values = {}
    for name, model in models.items():
        stop_controller.require_running()
        test_values[name] = pyramid_probabilities(model, datasets["test"], cfg, device)
    stop_controller.require_running()
    test_values["original"] = _old_probabilities(
        old, old_sets["test"], cfg.evaluation["batch_size"]
    )
    reports["original"] = {
        "checkpoint": str(old_path),
        "sha256": old_hash,
        "thresholds": old_cfg.decision_thresholds,
        "recalibrated_thresholds": old_thresholds,
        "parameters": sum(p.numel() for p in old.model.parameters()),
    }
    known_development = {
        s.group
        for part in ("train", "val")
        for s in _acdc_split(root / cfg.data["acdc_root"], part, cfg.attributes)
    }
    unseen_indices = [
        i
        for i, s in enumerate(splits["test"])
        if s.path is not None and s.group not in known_development
    ]
    for name, values in test_values.items():
        stop_controller.require_running()
        thresholds = reports[name]["thresholds"]
        reports[name]["test"] = grouped_metrics(datasets["test"], values, cfg, thresholds)
        reports[name]["test_at_0_5"] = grouped_metrics(
            datasets["test"], values, cfg, dict.fromkeys(cfg.attributes, 0.5)
        )
        if unseen_indices:
            ds = datasets["test"]
            reports[name]["old_unseen_recordings"] = masked_metrics(
                ds.targets[unseen_indices],
                ds.mask[unseen_indices],
                values[unseen_indices],
                cfg.attributes,
                thresholds,
            )
        if name == "original":
            reports[name]["test_recalibrated"] = grouped_metrics(
                datasets["test"], values, cfg, old_thresholds
            )
            pixels, cues, _ = old_sets["test"][0]
            reports[name]["forward_ms"] = benchmark_model(
                old.model, pixels.unsqueeze(0), cfg, cues=cues.unsqueeze(0)
            )
        else:
            pixels = (
                torch.from_numpy(np.array(datasets["test"].images[0], copy=True))
                .permute(2, 0, 1)
                .unsqueeze(0)
                .float()
                .div_(255)
            )
            reports[name]["forward_ms"] = benchmark_model(models[name], pixels, cfg)
    if hashlib.sha256(old_path.read_bytes()).hexdigest() != old_hash:
        raise ValueError("对照期间旧天气权重被外部修改")
    stop_controller.require_running()
    report = {
        "protocol": "project_fourfold",
        "official_test_protocol": False,
        "split_manifest": split_info["manifest"],
        "split_signature": split_info["signature"],
        "target_ratios": split_info["target_ratios"],
        "seed": cfg.data["seed"],
        "device": str(device),
        "image_size": list(cfg.image_size),
        "default_pipeline_promoted": False,
        "evaluation_only": evaluate_only,
        "benchmark_cpu_threads": cfg.evaluation["cpu_threads"],
        "splits": {
            name: {key: value for key, value in row.items() if key != "records"}
            for name, row in split_info["splits"].items()
        },
        "models": reports,
        "old_unseen_recording_test_samples": len(unseen_indices),
        "limits": [
            "旧模型接触过部分新四分数据，同图比较是开发回归，不是双方完全独立的泛化证明。",
            "ACDC 标签依据原论文的单一恶劣条件／正常条件采集协议，不能代表真实道路全部组合天气。",
            "只有四个 Pixel Accurate 场景；每折只有一个场景，不能代表所有实际道路。",
            "延迟只计单帧模型前向，不包含图像解码、预处理或整条 pipeline。",
        ],
    }
    directory = root / cfg.report_dir
    directory.mkdir(parents=True, exist_ok=True)
    (directory / ("evaluation.json" if evaluate_only else "comparison.json")).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (directory / "test_predictions.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        fields = [
            "id",
            "source",
            "group",
            "context",
            "model",
            "attributes",
            *[f"gt_{a}" for a in cfg.attributes],
            *[f"probability_{a}" for a in cfg.attributes],
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, sample in enumerate(splits["test"]):
            for name, values in test_values.items():
                row = {
                    "id": sample_identifier(sample),
                    "source": weather_source(sample),
                    "group": sample.group,
                    "context": sample.stratum,
                    "model": name,
                    "attributes": "+".join(
                        a
                        for j, a in enumerate(cfg.attributes)
                        if values[index, j] >= reports[name]["thresholds"][a]
                    )
                    or "none",
                }
                row.update(
                    {
                        f"gt_{a}": int(sample.labels[j])
                        if datasets["test"].mask[index, j]
                        else "unknown"
                        for j, a in enumerate(cfg.attributes)
                    }
                )
                row.update(
                    {
                        f"probability_{a}": float(values[index, j])
                        for j, a in enumerate(cfg.attributes)
                    }
                )
                writer.writerow(row)
    return report
