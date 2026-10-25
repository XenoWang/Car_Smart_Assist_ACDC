"""实际训练、分组评估和中断恢复；测试标签不参与选模型或阈值。"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import shutil
import time
from collections import Counter
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from car_smart_assist.config.weather import load_weather_config
from car_smart_assist.config.weather_attention import (
    attention_model_config,
    load_weather_attention_config,
)
from car_smart_assist.config.weather_pyramid import load_weather_pyramid_config
from car_smart_assist.data.weather_pyramid import (
    PyramidDataset,
    prepare_pyramid_splits,
    sample_identifier,
    weather_source,
)
from car_smart_assist.perception.weather import WeatherPredictor, select_device
from car_smart_assist.perception.weather_attention import (
    WeatherAttention,
    WeatherAttentionPredictor,
    ensure_pretrained_weights,
    optimize_attention_inference,
)
from car_smart_assist.perception.weather_continuation import load_continuation
from car_smart_assist.perception.weather_pyramid import WeatherPyramidPredictor, masked_weather_loss
from car_smart_assist.perception.weather_pyramid_training import (
    _augment,
    _old_probabilities,
    benchmark_model,
    grouped_metrics,
    masked_metrics,
    state_hash,
)
from car_smart_assist.perception.weather_swad import (
    SWADTracker,
    finish_swad_checkpoints,
    source_balanced_bce,
)
from car_smart_assist.perception.weather_training import (
    WeatherDataset,
    _save_checkpoint,
    require_independent_weather_output,
)

from car_smart_assist.utils.stop import STOP_EXIT_CODE, StopController

logger = logging.getLogger(__name__)


class BackboneL2SP:
    """只约束可训练骨干接近 ImageNet 起点；参考在恢复权重之前保存。"""

    def __init__(self, model, coefficient):
        self.model, self.coefficient = model, coefficient
        self.references = {
            name: p.detach().float().clone() for name, p in model.backbone.named_parameters()
        }
        self.names = None
        self.parameters = []
        self.reference_vector = None

    def __call__(self):
        active = [
            (name, p) for name, p in self.model.backbone.named_parameters() if p.requires_grad
        ]
        names = tuple(name for name, _ in active)
        if not names:
            return next(self.model.parameters()).new_zeros(())
        if names != self.names:
            self.names = names
            self.parameters = [p for _, p in active]
            self.reference_vector = torch.cat([self.references[name].reshape(-1) for name in names])
        current = torch.cat([p.float().reshape(-1) for p in self.parameters])
        return self.coefficient * 0.5 * (current - self.reference_vector).square().sum()


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    temporary.replace(path)


def save_attention_best(payload, path):
    """best 和 last 分两次写；保留上一份 best，防止两次写入之间中断。"""
    if path.exists():
        rollback = path.with_name("best.rollback.pt")
        temporary = rollback.with_suffix(".tmp")
        shutil.copy2(path, temporary)
        temporary.replace(rollback)
    _save_checkpoint(payload, path)


def source_balanced_weights(dataset, known=None):
    sources = np.asarray([weather_source(s) for s in dataset.samples])
    known = np.ones(len(sources), dtype=bool) if known is None else known
    counts = Counter(sources[known])
    return np.asarray(
        [1 / counts[source] if keep else 0 for source, keep in zip(sources, known, strict=True)]
    )


def attention_sampling_weights(dataset, cfg):
    keys = [(weather_source(s), s.labels, s.group) for s in dataset.samples]
    counts = Counter(keys)
    groups = Counter((source, labels) for source, labels, _ in counts)
    combinations = Counter(source for source, _ in groups)
    return torch.tensor(
        [
            cfg.train["source_sampling_weights"][source]
            / combinations[source]
            / groups[source, labels]
            / counts[source, labels, group]
            for source, labels, group in keys
        ],
        dtype=torch.float64,
    )


def batch_images(dataset, indices, device, cfg):
    # 连续评估直接读 NHWC；随机训练使用本机实测更快的 stack 路径。
    if isinstance(indices, slice):
        array = np.array(dataset.images[indices], copy=True)
        pixels = torch.from_numpy(array).permute(0, 3, 1, 2)
    else:
        indices = np.asarray(indices)
        if np.any(indices < 0) or np.any(indices >= len(dataset)):
            raise IndexError("批量索引超出图片范围")
        pixels = torch.stack(
            [
                torch.from_numpy(np.array(dataset.images[i], copy=True)).permute(2, 0, 1)
                for i in indices
            ]
        )
    if device.type == "cuda":
        pixels = pixels.pin_memory()
    pixels = pixels.to(device, dtype=torch.float32, non_blocking=True).div_(255)
    pixels = pixels.contiguous(
        memory_format=torch.channels_last if cfg.train["channels_last"] else torch.contiguous_format
    )
    return pixels


def attention_probabilities(model, dataset, cfg, device):
    output = np.empty((len(dataset), len(cfg.attributes)), np.float32)
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(dataset), cfg.evaluation["batch_size"]):
            stop = min(start + cfg.evaluation["batch_size"], len(dataset))
            images = batch_images(dataset, slice(start, stop), device, cfg)
            # 校准、评估和 predict 都用 FP32，避免临界阈值受精度路径影响。
            logits = model(images)
            output[start:stop] = logits.float().sigmoid().cpu().numpy()
    if not np.isfinite(output).all():
        raise ValueError("天气概率包含非有限值")
    return output


def balanced_thresholds(dataset, probabilities, cfg):
    """每来源总权重相同；返回校准阈值或验证选模型分数。"""
    thresholds, rows = {}, {}
    grid = np.linspace(0, 1, cfg.evaluation["threshold_steps"] + 2)[1:-1]
    for index, name in enumerate(cfg.attributes):
        known = dataset.mask[:, index] > 0
        truth = dataset.targets[known, index] > 0.5
        values = probabilities[known, index]
        if not truth.any() or truth.all():
            raise ValueError(f"{name} 缺少已确认的正／负样本")
        weights = source_balanced_weights(dataset, known)[known]
        positive = weights[truth].sum()
        prediction = values[None, :] >= grid[:, None]
        true_positive = (prediction * (weights * truth)[None, :]).sum(1)
        predicted_positive = (prediction * weights[None, :]).sum(1)
        f1 = 2 * true_positive / (positive + predicted_positive)
        best = np.flatnonzero(np.isclose(f1, f1.max(), atol=1e-12, rtol=0))
        selected = min(best, key=lambda i: (abs(grid[i] - np.median(grid[best])), grid[i]))
        thresholds[name] = float(grid[selected])
        rows[name] = {
            "source_balanced_ap": float(
                average_precision_score(truth, values, sample_weight=weights)
            ),
            "source_balanced_max_f1": float(f1[selected]),
        }
    fraction = cfg.evaluation["ap_selection_fraction"]
    weights = cfg.evaluation["selection_weights"]
    score = sum(
        weights[n]
        * (
            fraction * rows[n]["source_balanced_ap"]
            + (1 - fraction) * rows[n]["source_balanced_max_f1"]
        )
        for n in cfg.attributes
    ) / sum(weights.values())
    return thresholds, {"score": float(score), "per_attribute": rows}


def detailed_metrics(dataset, probabilities, cfg, thresholds):
    result = grouped_metrics(dataset, probabilities, cfg, thresholds)
    groups = {}
    for index, sample in enumerate(dataset.samples):
        groups.setdefault(sample.group, []).append(index)
    result["by_recording"] = {
        group: masked_metrics(
            dataset.targets[idx], dataset.mask[idx], probabilities[idx], cfg.attributes, thresholds
        )
        for group, idx in sorted(groups.items())
    }
    for index, name in enumerate(cfg.attributes):
        known = dataset.mask[:, index] > 0
        truth = dataset.targets[known, index]
        values = np.clip(probabilities[known, index], 1e-7, 1 - 1e-7)
        result["per_attribute"][name]["brier"] = float(np.mean((values - truth) ** 2))
        result["per_attribute"][name]["nll"] = float(
            np.mean(-truth * np.log(values) - (1 - truth) * np.log(1 - values))
        )
    return result


def learning_rate_factor(epoch, cfg):
    if "continuation" in cfg.train:
        stage = cfg.train["continuation"]
        position = epoch - stage["source_epoch"] - 1
        if position <= 0:
            return stage["start_factor"]
        progress = min(1.0, position / max(1, stage["additional_epochs"] - 1))
        minimum = cfg.train["minimum_lr_ratio"]
        return minimum + (stage["start_factor"] - minimum) * (1 + math.cos(math.pi * progress)) / 2
    warmup = cfg.train["warmup_epochs"]
    planned = cfg.train.get("schedule_epochs", cfg.train["epochs"])
    if epoch <= warmup:
        return epoch / max(1, warmup)
    progress = (epoch - warmup) / max(1, planned - warmup)
    minimum = cfg.train["minimum_lr_ratio"]
    return minimum + (1 - minimum) * (1 + math.cos(math.pi * progress)) / 2


def prepare_attention_datasets(cfg, root):
    splits, info = prepare_pyramid_splits(cfg, root)
    datasets = {
        name: PyramidDataset(samples, cfg, root / cfg.data["cache_dir"])
        for name, samples in splits.items()
    }
    return datasets, info


def model_report(cfg, root, progress, comparison=None):
    stopping = cfg.train.get("early_stopping", {})
    stopping_text = (
        f"验证评分连续 {stopping['patience']} 轮未超过参考分数 {stopping['min_delta']:g} 时早停"
        if stopping.get("enabled", False)
        else "不提前早停"
    )
    fraction = cfg.evaluation["ap_selection_fraction"]
    selection_weights = "、".join(
        f"{name}={cfg.evaluation['selection_weights'][name]:g}" for name in cfg.attributes
    )
    lines = [
        "# 天气模型报告",
        "",
        f"更新：{datetime.now().isoformat(timespec='seconds')}",
        "",
        "## 当前状态",
        "",
        f"- 状态：{progress['status']}",
        f"- 已完成轮数：{progress.get('epoch', 0)}／{cfg.train['epochs']}；{stopping_text}。",
        f"- 终止原因：{progress.get('termination_reason') or '尚未终止'}。",
        f"- 验证集最佳轮次：{progress.get('best_epoch', '尚未选出')}。",
        "- 新候选：预训练 MobileNetV3 三层特征＋四属性注意力，使用 ASL。",
        "- 原模型、已训练的金字塔 ASL 仅作对照，不强制蒸馏其错误输出。",
        f"- 沿用固定种子 {cfg.data['seed']} 的录制组四分；"
        + ("ACDC 正常参考及原图绑定" if cfg.data.get("include_normal_references", True) else "未加入 ACDC 正常参考图")
        + "，原始数据保留。",
        f"- 验证选模型，校准选阈值，最后评估开发测试；不在 {cfg.train['epochs']} 轮中反复挑测试分数。",
        f"- 验证评分按来源等权，AP 占 {100 * fraction:g}%、最优 F1 占 {100 * (1 - fraction):g}%；属性权重：{selection_weights}。验证阈值不用于部署。",
        f"- {cfg.train['epochs']} 轮是累计训练预算，不保证最后一轮或更长训练更好。"
        + (
            "追加阶段由收尾脚本按验证评分处理部署或回退。"
            if "continuation" in cfg.train
            else "默认 pipeline 不自动切换。"
        ),
        "",
    ]
    if cfg.train.get("swad", {}).get("enabled", False):
        options = cfg.train["swad"]
        lines += [
            "## 本轮 SWAD",
            "",
            f"从第 {options['start_epoch']} 轮起，在每次有效参数更新后密集平均；每轮验证一次，收敛／容忍窗口为 {options['n_converge']}／{options['n_tolerance']}，容忍比例 {options['tolerance_ratio']}。",
            "验证监测使用来源等权 BCE，雨雾权重沿用现有验证配置。ASL 训练目标、余弦学习率和冻结 BN 保留；谷底结束只停止收集权重，训练终止仍由上述预算与早停配置控制。这是对当前多标签模型的适配，没有完整复现原论文的分类与学习率配方。",
            "单点权重保存在 single_best.pt，平均权重在成功找到谷底后保存为 swad.pt。两个版本各自独立校准，按相同验证评分选择 best.pt，不使用测试分数来选择。",
            "",
        ]
        if comparison:
            lines += [
                f"本轮最终导出：`{comparison['selected_kind']}`。平均模型的 epoch 表示所含更新的终点，不是单点最优轮次。",
                "",
                "```json",
                json.dumps(comparison["swad"], ensure_ascii=False, indent=2),
                "```",
                "",
            ]
    if "continuation" in cfg.train:
        stage = cfg.train["continuation"]
        lines += [
            f"本阶段从第 {stage['source_epoch']} 轮 last 追加 {stage['additional_epochs']} 轮；保留优化器与随机状态，不重复冻结，学习率从源末轮平滑衰减。",
            "",
        ]
    preflight = root / cfg.report_dir / "preflight.json"
    if preflight.exists():
        profile = json.loads(preflight.read_text(encoding="utf-8"))
        lines += [
            "## 训练前效率检查",
            "",
            "| 模型 | 参数数 | CPU 单帧 ms | GPU 单帧 ms |",
            "|---|---:|---:|---:|",
        ]
        for name, row in profile["models"].items():
            lines.append(
                f"| {name} | {row['parameters']} | {row['forward_ms'].get('cpu', 0):.3f} | {row['forward_ms'].get('cuda', 0):.3f} |"
            )
        lines += [
            "",
            f"CPU 前向预检线程数：{profile.get('cpu_threads', '未记录（旧预检）')}；当前配置为 {cfg.evaluation['cpu_threads']}。只计模型前向，未计整条车载 pipeline。预训练模型尚未训练时的输出不代表天气能力。",
            "",
            f"随机批读取与传到 GPU：向量路径 {profile['batch_transfer_ms']['vector']:.2f} ms，选用 stack 路径 {profile['batch_transfer_ms']['selected_stack']:.2f} ms；两种像素一致。",
            "",
        ]
    if comparison:
        lines += [
            "## 最佳权重的分数",
            "",
            "以下为校准阈值下的项目开发测试结果；旧模型曾见过部分图片，这不是三者完全未见的官方测试。",
            "",
            "| 模型 | 平均 AP | 平均 F1 | 雨 P | 雨 R | 雨 F1 | 雾 P | 雾 R | 雾 F1 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for name, row in comparison["models"].items():
            test = row["test"]
            rain, fog = test["per_attribute"]["rain"], test["per_attribute"]["fog"]
            values = [
                test["macro_ap"],
                test["macro_f1"],
                rain["precision"],
                rain["recall"],
                rain["f1"],
                fog["precision"],
                fog["recall"],
                fog["f1"],
            ]
            lines.append(f"| {name} | " + " | ".join(f"{100 * v:.2f}%" for v in values) + " |")
        lines += [
            "",
            "### 四个属性分别看",
            "",
            "| 模型 | 属性 | AP | P | R | F1 | 阈值 |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
        for name, row in comparison["models"].items():
            for attribute, metrics in row["test"]["per_attribute"].items():
                scores = [metrics[k] for k in ("average_precision", "precision", "recall", "f1")]
                lines.append(
                    f"| {name} | {attribute} | "
                    + " | ".join(f"{100 * v:.2f}%" for v in scores)
                    + f" | {metrics['threshold']:.2f} |"
                )
        lines += [
            "",
            "### 误报和漏报",
            "",
            "| 模型 | 雨漏判／雨图 | 雨误报／无雨图 | 雾漏判／雾图 | 雾误报／无雾图 |",
            "|---|---:|---:|---:|---:|",
        ]
        for name, row in comparison["models"].items():
            rain, fog = (row["test"]["per_attribute"][n] for n in ("rain", "fog"))
            lines.append(
                f"| {name} | {rain['fn']}/{rain['known_positive']} | {rain['fp']}/{rain['known_negative']} | {fog['fn']}/{fog['known_positive']} | {fog['fp']}/{fog['known_negative']} |"
            )
        lines += [
            "",
            "两份对照报告保留各自原阈值，也另外给出同一校准策略的结果；详细结果含来源、录制组、昼夜雨条件、Brier 和 NLL。",
            "",
        ]
    else:
        lines += [
            "## 待完成的成绩",
            "",
            "新候选尚未完成本轮训练和独立校准，暂不填写新模型测试分数。训练正常结束（预算完成或早停）后，程序会写入最佳权重与对照的分数。",
            "",
        ]
    if progress.get("error"):
        lines += ["## 中断原因", "", progress["error"], ""]
    lines += [
        "## 文件",
        "",
        "- 配置来源：调用方传入的 WeatherAttentionConfig；本次有效配置快照见下方。",
        f"- 实际进度：`{cfg.report_dir}/progress.json`。",
        f"- 每轮验证：`{cfg.report_dir}/history.json`。",
        f"- 完整对照：`{cfg.report_dir}/comparison.json`（训练完成后生成）。",
        f"- 逐图分数：`{cfg.report_dir}/test_predictions.csv`（训练完成后生成）。",
        f"- best／last：`{cfg.checkpoint_root}`。",
        "",
        "输入仍是单帧视觉估计，没有新加水深、雨量或水花实测输出。属性不确定不单独请求接管。",
        "",
        "### 本次有效配置",
        "",
        "```json",
        json.dumps(asdict(cfg), ensure_ascii=False, indent=2, allow_nan=False),
        "```",
    ]
    destination = root / cfg.report_file
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp.md")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(destination)


def require_independent_weather_reports(cfg, root):
    """Preflight and training must preserve the deployed model's report artifacts."""
    import yaml

    route_path = root / "configs/model/weather_default.yaml"
    if not route_path.is_file():
        return
    route = yaml.safe_load(route_path.read_text(encoding="utf-8"))["weather_default"]
    document = yaml.safe_load((root / route["config"]).read_text(encoding="utf-8"))
    deployed = next(iter(document.values()))
    protected_dir = deployed.get("report_dir")
    protected_file = deployed.get("report_file")
    targets = [root / cfg.report_dir / name for name in ("preflight.json", "progress.json")]
    targets.append(root / cfg.report_file)
    for path in targets:
        if (
            protected_dir and path.resolve().is_relative_to((root / protected_dir).resolve())
            or protected_file and path.resolve() == (root / protected_file).resolve()
        ):
            raise ValueError("报告输出包含受保护的默认天气运行证据；请使用独立 report_dir/report_file")


def preflight_attention(cfg, root, datasets, split_info):
    require_independent_weather_reports(cfg, root)
    device = select_device(cfg.device)
    weights = ensure_pretrained_weights(cfg, root)
    model = WeatherAttention(cfg, weights).to(device)
    if cfg.train["channels_last"]:
        model.to(memory_format=torch.channels_last)
    ds = datasets["train"]
    indices = np.random.default_rng(cfg.data["seed"]).choice(
        len(ds), min(cfg.train["batch_size"], len(ds)), replace=False
    )
    reference = torch.stack([ds[i][0] for i in indices])
    fast = torch.from_numpy(np.take(ds.images, indices, axis=0, mode="clip")).permute(0, 3, 1, 2)
    slow_runs, quick_runs = [], []
    for _ in range(5):
        started = time.perf_counter()
        reference = torch.stack([ds[i][0] for i in indices])
        slow_runs.append(time.perf_counter() - started)
        started = time.perf_counter()
        fast = torch.from_numpy(np.take(ds.images, indices, axis=0, mode="clip")).permute(
            0, 3, 1, 2
        )
        quick_runs.append(time.perf_counter() - started)
    slow, quick = float(np.median(slow_runs)), float(np.median(quick_runs))
    if not torch.equal(reference, fast):
        raise ValueError("批量读取与逐图读取像素不一致")
    transfer = {}
    for name in ("vector", "selected_stack"):
        times = []
        for _ in range(5):
            if device.type == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            if name == "vector":
                transferred = torch.from_numpy(
                    np.take(ds.images, indices, axis=0, mode="clip")
                ).permute(0, 3, 1, 2)
                if device.type == "cuda":
                    transferred = transferred.pin_memory()
                transferred = transferred.to(device, dtype=torch.float32, non_blocking=True).div_(
                    255
                )
                if cfg.train["channels_last"]:
                    transferred = transferred.contiguous(memory_format=torch.channels_last)
            else:
                transferred = batch_images(ds, indices, device, cfg)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - started)
        transfer[name] = float(np.median(times) * 1000)
    image = batch_images(ds, slice(0, 1), device, cfg)
    models = {
        "attention_before_fusion": {
            "parameters": sum(p.numel() for p in model.parameters()),
            "forward_ms": benchmark_model(model, image, cfg),
        }
    }
    optimized = optimize_attention_inference(model)
    with torch.inference_mode():
        default_precision_error = float((model.eval()(image) - optimized(image)).abs().max())
        # TF32 的卷积舍入在合并前后不同；只在等价性检查中临时使用完整 FP32。
        with torch.backends.cudnn.flags(allow_tf32=False):
            fusion_error = float((model(image) - optimized(image)).abs().max())
    if fusion_error > 1e-5:
        raise ValueError("卷积／BN 合并输出误差超出检查限值")
    models["attention_asl"] = {
        "parameters": sum(p.numel() for p in optimized.parameters()),
        "forward_ms": benchmark_model(optimized, image, cfg),
    }
    old_cfg = replace(load_weather_config(root / cfg.evaluation["old_config"]), device=cfg.device)
    original = WeatherPredictor.from_checkpoint(root / cfg.evaluation["old_checkpoint"], old_cfg)
    old_ds = WeatherDataset(ds.samples[:1], old_cfg, root / cfg.report_dir / "preflight_old.npy")
    pixels, cues, _ = old_ds[0]
    models["original"] = {
        "parameters": sum(p.numel() for p in original.model.parameters()),
        "forward_ms": benchmark_model(
            original.model, pixels.unsqueeze(0), cfg, cues=cues.unsqueeze(0)
        ),
    }
    asl_cfg = load_weather_pyramid_config(root / cfg.evaluation["asl_config"])
    asl = WeatherPyramidPredictor.from_checkpoint(
        root / cfg.evaluation["asl_checkpoint"], asl_cfg, cfg.device
    )
    models["pyramid_asl"] = {
        "parameters": sum(p.numel() for p in asl.model.parameters()),
        "forward_ms": benchmark_model(asl.model, image, cfg),
    }
    model.set_training_epoch(cfg.backbone["freeze_epochs"] + 1)
    model.train()
    sample = batch_images(ds, indices, device, cfg)
    labels = torch.from_numpy(ds.targets[indices]).to(device)
    mask = torch.from_numpy(ds.mask[indices]).to(device)
    dtype = (
        torch.bfloat16
        if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    regularizer = BackboneL2SP(model, cfg.train["l2_sp"]) if cfg.train.get("l2_sp", 0) else None
    training_probe_times = []
    for _ in range(3):
        model.zero_grad(set_to_none=True)
        training_probe_started = time.perf_counter()
        with torch.autocast(
            device.type, enabled=device.type == "cuda" and cfg.train["amp"], dtype=dtype
        ):
            logits = model(sample)
            data_loss = masked_weather_loss(logits, labels, mask, "asl", cfg)
        loss = data_loss + regularizer() if regularizer is not None else data_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), cfg.train["gradient_clip"], error_if_nonfinite=True
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        training_probe_times.append(time.perf_counter() - training_probe_started)
    training_probe_seconds = float(np.median(training_probe_times[1:]))
    if logits.shape != (len(indices), 4) or not torch.isfinite(loss):
        raise ValueError("真实图片前后向检查失败")
    swad_profile = {}
    if cfg.train.get("swad", {}).get("enabled", False):
        tracker = SWADTracker(model, cfg.train["swad"])
        times = []
        for _ in range(6):
            if device.type == "cuda":
                torch.cuda.synchronize()
            tick = time.perf_counter()
            tracker.update(cfg.train["swad"]["start_epoch"])
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - tick)
        swad_profile = {"swad_dense_update_ms": float(np.median(times[1:]) * 1000)}
    result = {
        "device": str(device),
        "image_size": list(cfg.image_size),
        "models": models,
        "batch_read_speedup": slow / max(quick, 1e-9),
        "batch_transfer_ms": transfer,
        "cpu_threads": cfg.evaluation["cpu_threads"],
        "batch_transfer_speedup": transfer["vector"] / max(transfer["selected_stack"], 1e-9),
        "training_batch_forward_backward_seconds": training_probe_seconds,
        "l2_sp_coefficient": cfg.train.get("l2_sp", 0.0),
        "swad": cfg.train.get("swad"),
        **swad_profile,
        "finite_forward_backward": True,
        "fusion_max_absolute_logit_error": fusion_error,
        "fusion_default_precision_logit_error": default_precision_error,
        "backbone_bn_frozen": all(
            not m.training for m in model.backbone.modules() if isinstance(m, torch.nn.BatchNorm2d)
        ),
        "split_signature": split_info["signature"],
        "model_config": attention_model_config(cfg),
        "batch_size": cfg.train["batch_size"],
        "channels_last": cfg.train["channels_last"],
        "training_started": False,
        "note": "只作效率与梯度检查，没有更新模型参数或开始正式训练。",
    }
    write_json(root / cfg.report_dir / "preflight.json", result)
    progress = {
        "status": "实现与训练前运行检查完成；等待启动",
        "epoch": 0,
        "planned_epochs": cfg.train["epochs"],
        "updated": datetime.now().isoformat(timespec="seconds"),
    }
    write_json(root / cfg.report_dir / "progress.json", progress)
    model_report(cfg, root, progress)
    return result


def train_attention(cfg, root, datasets, split_info, *, fresh=False, stop_controller=None, continue_only=False):
    destination = Path(root).resolve() / cfg.checkpoint_root
    controller = stop_controller or StopController(default_directory=destination)
    controller.watch_directory(destination)
    with controller:
        controller.require_running()
        if continue_only and (fresh or not (destination / "last.pt").is_file()):
            raise ValueError("--continue 需要本训练输出目录已有的配对 last")
        return _train_attention(
            cfg, root, datasets, split_info, fresh=fresh, stop_controller=controller,
            continue_only=continue_only,
        )


def _train_attention(cfg, root, datasets, split_info, *, fresh, stop_controller, continue_only=False):
    root = Path(root).resolve()
    destination = root / cfg.checkpoint_root
    require_independent_weather_output(root, destination)
    require_independent_weather_reports(cfg, root)
    swad_options = cfg.train.get("swad", {})
    use_swad = swad_options.get("enabled", False)
    stopping = cfg.train.get("early_stopping", {})
    use_early_stopping = stopping.get("enabled", False)
    best_path = destination / ("single_best.pt" if use_swad else "best.pt")
    last_path = destination / "last.pt"
    if not fresh and not last_path.exists() and any(
        path.exists() for path in (best_path, destination / "best.pt")
    ):
        raise ValueError("已有 best 但缺少配对 last；保留原件，使用 --fresh 或独立输出目录")
    device = select_device(cfg.device)
    weights = ensure_pretrained_weights(cfg, root)
    if "continuation" in cfg.train and any(
        (root / cfg.train["continuation"][key]).resolve().parent == destination.resolve()
        for key in ("checkpoint", "single_best")
    ):
        raise ValueError("追加训练输出必须使用独立目录，不能覆盖源检查点")
    stop_controller.require_running()
    if fresh:
        existing = [
            p
            for p in dict.fromkeys(
                (
                    best_path,
                    last_path,
                    destination / "best.rollback.pt",
                    destination / "best.pt",
                    *([destination / "swad.pt"] if use_swad else []),
                )
            )
            if p.exists()
        ]
        if existing:
            folder = destination / "history" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            folder.mkdir(parents=True)
            for p in existing:
                shutil.copy2(p, folder / p.name)
            for p in existing:
                p.unlink()
    baseline_hashes = {
        name: hashlib.sha256((root / cfg.evaluation[key]).read_bytes()).hexdigest()
        for name, key in (("original", "old_checkpoint"), ("pyramid_asl", "asl_checkpoint"))
    }
    baseline_hashes.update(
        {
            name: hashlib.sha256((root / ref["checkpoint"]).read_bytes()).hexdigest()
            for name, ref in cfg.evaluation.get("attention_references", {}).items()
        }
    )
    recipe = {
        "model": attention_model_config(cfg),
        "train": cfg.train,
        "backbone_training": {k: cfg.backbone[k] for k in ("freeze_epochs", "fine_tune_from")},
        "loss": cfg.loss,
        "selection": cfg.evaluation,
        "split_signature": split_info["signature"],
        "dataset_signatures": {n: d.signature for n, d in datasets.items()},
        "seed": cfg.data["seed"],
        "pretrained_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
        "baseline_hashes": baseline_hashes,
    }
    cuda_devices = (
        [device.index if device.index is not None else torch.cuda.current_device()]
        if device.type == "cuda" else []
    )
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(cfg.data["seed"])
        model = WeatherAttention(cfg, weights).to(device)
        if cfg.train["channels_last"]:
            model.to(memory_format=torch.channels_last)
        initial_model_sha256 = state_hash(model)
        regularizer = BackboneL2SP(model, cfg.train["l2_sp"]) if cfg.train.get("l2_sp", 0) else None
        tracker = SWADTracker(model, swad_options) if use_swad else None
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": list(model.backbone.parameters()),
                    "base_lr": cfg.train["backbone_learning_rate"],
                },
                {
                    "params": [
                        p for n, p in model.named_parameters() if not n.startswith("backbone.")
                    ],
                    "base_lr": cfg.train["learning_rate"],
                },
            ],
            lr=cfg.train["learning_rate"],
            weight_decay=cfg.train["weight_decay"],
        )
        amp = device.type == "cuda" and cfg.train["amp"]
        dtype = torch.bfloat16 if amp and torch.cuda.is_bf16_supported() else torch.float16
        scaler = torch.amp.GradScaler("cuda", enabled=amp and dtype == torch.float16)
        generator = torch.Generator().manual_seed(cfg.data["seed"])
        epoch, best_epoch, best_score, history, elapsed_before = 0, 0, -1.0, [], 0.0
        stopping_state = {"best_score": None, "wait": 0, "stopped": False}
        own_resume = last_path.exists() and not fresh
        if own_resume or "continuation" in cfg.train:
            if own_resume:
                payload = torch.load(last_path, map_location="cpu", weights_only=True)
            else:
                payload = load_continuation(cfg, root, recipe, best_path)
            if own_resume and payload["recipe"] != recipe:
                raise ValueError("数据、基线或训练配方已变；请用 --fresh 新开实验")
            if (
                not best_path.exists()
                or hashlib.sha256(best_path.read_bytes()).hexdigest() != payload["best_sha256"]
            ):
                rollback = destination / "best.rollback.pt"
                if (
                    not rollback.exists()
                    or hashlib.sha256(rollback.read_bytes()).hexdigest() != payload["best_sha256"]
                ):
                    raise ValueError("last 对应的 best 及备份丢失或已改变")
                shutil.copy2(rollback, best_path)
                logger.info("恢复 last 对应的 best 备份，继续原计划")
            model.load_state_dict(payload["model_state"])
            if tracker is not None:
                if own_resume:
                    tracker.load_state_dict(payload["swad_state"])
                else:
                    tracker.step = payload["swad_state"]["step"]
            optimizer.load_state_dict(payload["optimizer_state"])
            if scaler.is_enabled() and payload["scaler_state"]:
                scaler.load_state_dict(payload["scaler_state"])
            generator.set_state(payload["generator_state"])
            torch.set_rng_state(payload["torch_rng"])
            if device.type == "cuda" and payload["cuda_rng"] is not None:
                torch.cuda.set_rng_state(payload["cuda_rng"], device)
            epoch, best_epoch, best_score, history = (
                payload["epoch"],
                payload["best_epoch"],
                payload["best_score"],
                payload["history"],
            )
            if continue_only and payload.get("planned_epochs") != cfg.train["epochs"]:
                raise ValueError("--continue 不允许改变 last 的原累计训练目标或学习率计划")
            elapsed_before = payload["training_seconds"]
            initial_model_sha256 = payload.get("initial_model_sha256", initial_model_sha256)
            if use_early_stopping:
                if "early_stopping" not in payload:
                    raise ValueError("检查点缺少早停状态，不能按当前早停配方恢复；请新开实验")
                stopping_state = dict(payload["early_stopping"])
        probabilities = None
        start = time.perf_counter()
        train = datasets["train"]
        sampling = attention_sampling_weights(train, cfg)
        logger.info(
            "attention ASL：%s，恢复 %d/%d，AMP=%s，全部轮数均验证",
            device,
            epoch,
            cfg.train["epochs"],
            dtype if amp else "关闭",
        )
        completed_stage = continue_only and own_resume and bool(payload.get("training_complete"))
        remaining = () if stopping_state["stopped"] or completed_stage else range(epoch + 1, cfg.train["epochs"] + 1)
        for current in remaining:
            if stop_controller.requested:
                break
            tick = time.perf_counter()
            model.set_training_epoch(current)
            model.train()
            for group in optimizer.param_groups:
                group["lr"] = group["base_lr"] * learning_rate_factor(current, cfg)
            order = torch.multinomial(
                sampling, len(train), replacement=True, generator=generator
            ).numpy()
            accumulated = torch.zeros((), device=device)
            accumulated_data, accumulated_penalty = accumulated.clone(), accumulated.clone()
            examples, skipped = 0, 0
            for offset in range(0, len(order), cfg.train["batch_size"]):
                indices = order[offset : offset + cfg.train["batch_size"]]
                images = _augment(batch_images(train, indices, device, cfg), cfg.train)
                labels = torch.from_numpy(train.targets[indices]).to(device)
                mask = torch.from_numpy(train.mask[indices]).to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device.type, enabled=amp, dtype=dtype):
                    logits = model(images)
                    data_loss = masked_weather_loss(logits, labels, mask, "asl", cfg)
                penalty = regularizer() if regularizer is not None else data_loss.new_zeros(())
                loss = data_loss + penalty if regularizer is not None else data_loss
                if not torch.isfinite(loss):
                    raise ValueError("训练损失非有限，停止本轮；保留上一个检查点")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    cfg.train["gradient_clip"],
                    error_if_nonfinite=not scaler.is_enabled(),
                )
                valid_update = not (scaler.is_enabled() and not torch.isfinite(norm))
                if not valid_update:
                    skipped += 1
                scaler.step(optimizer)
                scaler.update()
                if tracker is not None and valid_update:
                    tracker.update(current)
                accumulated += loss.detach() * len(indices)
                accumulated_data += data_loss.detach() * len(indices)
                accumulated_penalty += penalty.detach() * len(indices)
                examples += len(indices)
            probabilities = attention_probabilities(model, datasets["validation"], cfg, device)
            _, selection = balanced_thresholds(datasets["validation"], probabilities, cfg)
            metrics = detailed_metrics(
                datasets["validation"], probabilities, cfg, dict.fromkeys(cfg.attributes, 0.5)
            )
            swad_row = {}
            if tracker is not None:
                validation_loss = source_balanced_bce(datasets["validation"], probabilities, cfg)
                tracker.end_epoch(current, validation_loss)
                swad_row = {
                    "swad_validation_bce": validation_loss,
                    "swad": tracker.valley.summary(),
                }
            if selection["score"] > best_score + 1e-12:
                best_score, best_epoch = selection["score"], current
                save_attention_best(
                    {
                        "format_version": 1,
                        "model_config": attention_model_config(cfg),
                        "model_state": {n: p.detach().cpu() for n, p in model.state_dict().items()},
                        "epoch": current,
                        "selection": selection,
                        "validation": metrics,
                        "split_signature": split_info["signature"],
                        "loss_name": "asl",
                        "initial_model_sha256": initial_model_sha256,
                    },
                    best_path,
                )
            if use_early_stopping:
                reference = stopping_state["best_score"]
                if reference is None or selection["score"] > reference + stopping["min_delta"]:
                    stopping_state["best_score"] = selection["score"]
                    stopping_state["wait"] = 0
                else:
                    stopping_state["wait"] += 1
                stopping_state["stopped"] = (
                    stopping_state["wait"] >= stopping["patience"]
                    and current < cfg.train["epochs"]
                )
            training_complete = stopping_state["stopped"] or current == cfg.train["epochs"]
            termination_reason = (
                "early_stopping" if stopping_state["stopped"] else (
                    "budget_completed" if current == cfg.train["epochs"] else None
                )
            )
            row = {
                "epoch": current,
                "loss": float(accumulated / examples),
                "data_loss": float(accumulated_data / examples),
                "l2_sp_penalty": float(accumulated_penalty / examples),
                "sampling_sha256": hashlib.sha256(order.tobytes()).hexdigest(),
                "selection": selection,
                "validation": metrics,
                "best_epoch": best_epoch,
                "skipped_updates": skipped,
                "head_lr": optimizer.param_groups[1]["lr"],
                "backbone_lr": optimizer.param_groups[0]["lr"],
                "seconds": time.perf_counter() - tick,
                **swad_row,
            }
            history.append(row)
            elapsed = elapsed_before + time.perf_counter() - start
            _save_checkpoint(
                {
                    "format_version": 1,
                    "recipe": recipe,
                    "model_state": {n: p.detach().cpu() for n, p in model.state_dict().items()},
                    "optimizer_state": optimizer.state_dict(),
                    "scaler_state": scaler.state_dict(),
                    "epoch": current,
                    "planned_epochs": cfg.train["epochs"],
                    "training_complete": training_complete,
                    "termination_reason": termination_reason,
                    **({"early_stopping": dict(stopping_state)} if use_early_stopping else {}),
                    "best_epoch": best_epoch,
                    "best_score": best_score,
                    "history": history,
                    "training_seconds": elapsed,
                    "generator_state": generator.get_state(),
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
                    "best_sha256": hashlib.sha256(best_path.read_bytes()).hexdigest(),
                    "initial_model_sha256": initial_model_sha256,
                    **({"swad_state": tracker.state_dict()} if tracker is not None else {}),
                },
                last_path,
            )
            progress = {
                "status": (
                    "已早停，正在校准和对照" if stopping_state["stopped"] else (
                        "训练预算完成，正在校准和对照" if training_complete else "训练中"
                    )
                ),
                "epoch": current,
                "planned_epochs": cfg.train["epochs"],
                "best_epoch": best_epoch,
                "best_validation_score": best_score,
                "training_seconds": elapsed,
                "last_epoch_seconds": row["seconds"],
                "termination_reason": termination_reason,
                **({"early_stopping": dict(stopping_state)} if use_early_stopping else {}),
                "updated": datetime.now().isoformat(timespec="seconds"),
            }
            write_json(root / cfg.report_dir / "progress.json", progress)
            write_json(root / cfg.report_dir / "history.json", history)
            model_report(cfg, root, progress)
            logger.info(
                "attention %d/%d loss=%.6e ASL=%.6e L2-SP=%.6e score=%.4f rain-AP=%.4f fog-AP=%.4f best=%d %.1fs",
                current,
                cfg.train["epochs"],
                row["loss"],
                row["data_loss"],
                row["l2_sp_penalty"],
                selection["score"],
                selection["per_attribute"]["rain"]["source_balanced_ap"],
                selection["per_attribute"]["fog"]["source_balanced_ap"],
                best_epoch,
                row["seconds"],
            )
            if stop_controller.requested:
                break
            if stopping_state["stopped"]:
                logger.info("验证评分连续 %d 轮未达改善门槛，提前停止于 epoch %d", stopping_state["wait"], current)
                break
        def pause_if_requested():
            if not stop_controller.requested:
                return None
            if not last_path.is_file():
                stop_controller.require_running()
            last = torch.load(last_path, map_location="cpu", weights_only=True)
            last.update(termination_reason="manual_stop", manual_stop=True)
            _save_checkpoint(last, last_path)
            progress = {
                "status": "手动停止", "epoch": last["epoch"],
                "planned_epochs": cfg.train["epochs"], "best_epoch": last["best_epoch"],
                "termination_reason": "manual_stop", "manual_stop": True,
                "training_complete": last["training_complete"],
            }
            write_json(root / cfg.report_dir / "progress.json", progress)
            model_report(cfg, root, progress)
            return model.eval(), {
                **progress, "checkpoint": str(best_path),
                "trained_through_epoch": last["epoch"],
                "training_seconds": last["training_seconds"],
                "baseline_hashes": baseline_hashes,
                "split_signature": split_info["signature"],
                "initial_model_sha256": initial_model_sha256,
                "calibrated": False,
            }

        paused = pause_if_requested()
        if paused is not None:
            return paused
        payload = torch.load(best_path, map_location="cpu", weights_only=True)
        model.load_state_dict(payload["model_state"])
        export = {}
        if tracker is not None:
            try:
                inference, payload, export = finish_swad_checkpoints(
                    tracker, model, payload, cfg, root, datasets, device,
                    stop_controller=stop_controller,
                )
            except SystemExit as exc:
                if exc.code != STOP_EXIT_CODE:
                    raise
                paused = pause_if_requested()
                if paused is None:
                    raise
                return paused
            thresholds = payload["decision_thresholds"]
        else:
            inference = optimize_attention_inference(model)
            cal_values = attention_probabilities(inference, datasets["calibration"], cfg, device)
            paused = pause_if_requested()
            if paused is not None:
                return paused
            thresholds, _ = balanced_thresholds(datasets["calibration"], cal_values, cfg)
            payload.update(
                {
                    "calibration_signature": datasets["calibration"].signature,
                    "decision_thresholds": thresholds,
                    "calibration": detailed_metrics(
                        datasets["calibration"], cal_values, cfg, thresholds
                    ),
                }
            )
            save_attention_best(payload, best_path)
        last = torch.load(last_path, map_location="cpu", weights_only=True)
        last["best_sha256"] = hashlib.sha256(best_path.read_bytes()).hexdigest()
        if last["training_complete"]:
            last.update(manual_stop=False, termination_reason=(
                "budget_completed" if last["epoch"] == cfg.train["epochs"] else "early_stopping"
            ))
        if tracker is not None:
            last["export_best_sha256"] = hashlib.sha256(
                (destination / "best.pt").read_bytes()
            ).hexdigest()
        _save_checkpoint(last, last_path)
        paused = pause_if_requested()
        if paused is not None:
            return paused
    return inference, {
        "checkpoint": str(destination / "best.pt"),
        "best_epoch": last["best_epoch"],
        "selected_epoch": payload["epoch"],
        "trained_through_epoch": last["epoch"],
        "training_complete": last["training_complete"],
        "manual_stop": False,
        "termination_reason": last.get("termination_reason", "budget_completed"),
        **({"early_stopping": last["early_stopping"]} if use_early_stopping else {}),
        "thresholds": thresholds,
        "validation": payload["validation"],
        "selection": payload["selection"],
        "calibration": payload["calibration"],
        "training_seconds": last["training_seconds"],
        "baseline_hashes": baseline_hashes,
        "split_signature": split_info["signature"],
        "initial_model_sha256": initial_model_sha256,
        **export,
    }


def compare_attention(cfg, root, datasets, split_info, model, trained, *, stop_controller=None):
    stop_controller = stop_controller or StopController(default_directory=root / cfg.checkpoint_root)
    stop_controller.require_running()
    device = select_device(cfg.device)
    selected = trained.get("selected_kind", "attention_asl")
    results = {selected: trained}
    values = {selected: attention_probabilities(model, datasets["test"], cfg, device)}
    stop_controller.require_running()
    for name, candidate in trained.get("candidates", {}).items():
        if name == selected:
            continue
        stop_controller.require_running()
        predictor = WeatherAttentionPredictor.from_checkpoint(
            candidate["checkpoint"], cfg, cfg.device
        )
        results[name] = dict(candidate)
        values[name] = attention_probabilities(predictor.model, datasets["test"], cfg, device)
        del predictor
    for name, reference in cfg.evaluation.get("attention_references", {}).items():
        stop_controller.require_running()
        reference_cfg = load_weather_attention_config(root / reference["config"])
        predictor = WeatherAttentionPredictor.from_checkpoint(
            root / reference["checkpoint"], reference_cfg, cfg.device
        )
        if reference_cfg.image_size != cfg.image_size or reference_cfg.attributes != cfg.attributes:
            raise ValueError("注意力对照输入尺寸或属性不一致")
        payload = torch.load(root / reference["checkpoint"], map_location="cpu", weights_only=True)
        results[name] = {
            "checkpoint": str(root / reference["checkpoint"]),
            "thresholds": predictor.thresholds,
            "best_epoch": payload["epoch"],
            "selection": payload["selection"],
        }
        values[name] = attention_probabilities(predictor.model, datasets["test"], cfg, device)
        del predictor
    stop_controller.require_running()
    asl_cfg = load_weather_pyramid_config(root / cfg.evaluation["asl_config"])
    asl = WeatherPyramidPredictor.from_checkpoint(
        root / cfg.evaluation["asl_checkpoint"], asl_cfg, cfg.device
    )
    if asl_cfg.image_size != cfg.image_size:
        raise ValueError("ASL 对照输入尺寸与缓存不一致")
    stop_controller.require_running()
    values["pyramid_asl"] = attention_probabilities(asl.model, datasets["test"], cfg, device)
    stop_controller.require_running()
    asl_calibration = attention_probabilities(asl.model, datasets["calibration"], cfg, device)
    stop_controller.require_running()
    shared, _ = balanced_thresholds(datasets["calibration"], asl_calibration, cfg)
    results["pyramid_asl"] = {
        "checkpoint": str(root / cfg.evaluation["asl_checkpoint"]),
        "thresholds": asl.thresholds,
        "shared_calibration_thresholds": shared,
    }
    stop_controller.require_running()
    old_cfg = replace(load_weather_config(root / cfg.evaluation["old_config"]), device=cfg.device)
    old = WeatherPredictor.from_checkpoint(root / cfg.evaluation["old_checkpoint"], old_cfg)
    old_sets = {
        name: WeatherDataset(
            datasets[name].samples, old_cfg, root / cfg.data["cache_dir"] / f"old_{name}.npy"
        )
        for name in ("calibration", "test")
    }
    stop_controller.require_running()
    values["original"] = _old_probabilities(old, old_sets["test"], cfg.evaluation["batch_size"])
    stop_controller.require_running()
    old_calibration = _old_probabilities(old, old_sets["calibration"], cfg.evaluation["batch_size"])
    stop_controller.require_running()
    shared, _ = balanced_thresholds(datasets["calibration"], old_calibration, cfg)
    results["original"] = {
        "checkpoint": str(root / cfg.evaluation["old_checkpoint"]),
        "thresholds": old_cfg.decision_thresholds,
        "shared_calibration_thresholds": shared,
    }
    for name, probability in values.items():
        results[name]["test"] = detailed_metrics(
            datasets["test"], probability, cfg, results[name]["thresholds"]
        )
        if "shared_calibration_thresholds" in results[name]:
            results[name]["test_shared_calibration"] = detailed_metrics(
                datasets["test"], probability, cfg, results[name]["shared_calibration_thresholds"]
            )
        if name in trained["baseline_hashes"]:
            current_hash = hashlib.sha256(
                Path(results[name]["checkpoint"]).read_bytes()
            ).hexdigest()
            if current_hash != trained["baseline_hashes"][name]:
                raise ValueError("训练期间对照权重被外部改变；不生成混合版本成绩")
            results[name]["sha256"] = current_hash
    stop_controller.require_running()
    output = {
        "status": "完成",
        "trained_through_epoch": trained.get("trained_through_epoch", cfg.train["epochs"]),
        "planned_epochs": cfg.train["epochs"],
        "termination_reason": trained.get("termination_reason", "budget_completed"),
        **({"early_stopping": trained["early_stopping"]} if "early_stopping" in trained else {}),
        "protocol": "project_fourfold_development_regression",
        "official_test_protocol": False,
        "split_signature": split_info["signature"],
        "split_manifest": split_info["manifest"],
        "seed": cfg.data["seed"],
        "splits": {
            name: {k: v for k, v in row.items() if k != "records"}
            for name, row in split_info["splits"].items()
        },
        "default_pipeline_promoted": False,
        "models": results,
        **(
            {"selected_kind": trained["selected_kind"], "swad": trained["swad"]}
            if "selected_kind" in trained
            else {}
        ),
    }
    write_json(root / cfg.report_dir / "comparison.json", output)
    path = root / cfg.report_dir / "test_predictions.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "sample",
                "source",
                "group",
                "context",
                "model",
                "attribute",
                "label",
                "known",
                "probability",
                "threshold",
                "prediction",
            ]
        )
        for name, probability in values.items():
            for index, sample in enumerate(datasets["test"].samples):
                for attr_index, attribute in enumerate(cfg.attributes):
                    threshold = results[name]["thresholds"][attribute]
                    writer.writerow(
                        [
                            sample_identifier(sample),
                            weather_source(sample),
                            sample.group,
                            sample.stratum,
                            name,
                            attribute,
                            sample.labels[attr_index],
                            int(datasets["test"].mask[index, attr_index]),
                            float(probability[index, attr_index]),
                            threshold,
                            int(probability[index, attr_index] >= threshold),
                        ]
                    )
    progress = {
        "status": "完成",
        "epoch": trained.get("trained_through_epoch", cfg.train["epochs"]),
        "best_epoch": trained["best_epoch"],
        "selected_kind": trained.get("selected_kind", "attention_asl"),
        "selected_epoch": trained.get("selected_epoch", trained["best_epoch"]),
        "termination_reason": trained.get("termination_reason", "budget_completed"),
        **({"early_stopping": trained["early_stopping"]} if "early_stopping" in trained else {}),
        "updated": datetime.now().isoformat(timespec="seconds"),
    }
    write_json(root / cfg.report_dir / "progress.json", progress)
    model_report(cfg, root, progress, output)
    if "summary_file" in cfg.evaluation:
        summary = (root / cfg.evaluation["summary_file"]).resolve()
        if not summary.is_relative_to(root.resolve()):
            raise ValueError("汇总报告须留在项目内")
        text = summary.read_text(encoding="utf-8") if summary.exists() else ""
        tag = cfg.evaluation.get("summary_tag", "WEATHER_SWAD")
        begin, end = f"<!-- {tag}_BEGIN -->", f"<!-- {tag}_END -->"
        block = begin + "\n" + (root / cfg.report_file).read_text(encoding="utf-8") + end + "\n"
        if begin in text and end in text:
            before, rest = text.split(begin, 1)
            _, after = rest.split(end, 1)
            text = before + block + after
        else:
            text += "\n\n---\n\n" + block
        temporary = summary.with_suffix(".tmp.md")
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(summary)
    return output
