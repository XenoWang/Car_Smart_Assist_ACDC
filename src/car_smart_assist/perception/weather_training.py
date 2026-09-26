"""ACDC 官方 train/val 四类条件小模型训练；保留 test 集用于最终评估。"""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from car_smart_assist.config.weather import WeatherConfig
from car_smart_assist.perception.visibility.dataset import read_rgb_image
from car_smart_assist.perception.weather import WeatherClassifier, image_tensors, select_device


def list_condition_images(
    acdc_root: Path, split: str, classes: tuple[str, ...]
) -> list[tuple[Path, int]]:
    """只取官方恶劣条件图片，排除 paired *_ref 图片和 test。"""
    if split not in ("train", "val"):
        raise ValueError("训练和选优只允许官方 train/val 划分")
    root = acdc_root / "rgb_anon"
    records: list[tuple[Path, int]] = []
    for label, condition in enumerate(classes):
        folder = root / condition / split
        records.extend((p, label) for p in sorted(folder.rglob("*_rgb_anon.png")) if p.is_file())
    return records


def split_by_sequence(
    records: list[tuple[Path, int]],
    validation_fraction: float,
    seed: int,
) -> tuple[list[tuple[Path, int]], list[tuple[Path, int]]]:
    """按天气类别整段留出视频序列，避免相邻帧跨训练与验证集合。"""
    grouped: dict[int, dict[str, list[tuple[Path, int]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for record in records:
        path, label = record
        grouped[label][path.parent.name].append(record)

    train_records: list[tuple[Path, int]] = []
    val_records: list[tuple[Path, int]] = []
    for label, sequences in sorted(grouped.items()):
        groups = sorted(sequences.items())
        if len(groups) < 2:
            raise ValueError(f"天气类别 {label} 少于两个独立视频序列，无法做无泄漏验证")
        random.Random(seed + label).shuffle(groups)
        total = sum(len(items) for _, items in groups)
        target = min(total - 1, max(1, round(total * validation_fraction)))

        reachable: dict[int, tuple[str, ...]] = {0: ()}
        for sequence, items in groups:
            for count, selected in list(reachable.items()):
                new_count = count + len(items)
                if new_count < total and new_count not in reachable:
                    reachable[new_count] = (*selected, sequence)
        val_count = min(
            (count for count in reachable if 0 < count < total),
            key=lambda count: abs(count - target),
        )
        val_sequences = set(reachable[val_count])
        for sequence, items in groups:
            (val_records if sequence in val_sequences else train_records).extend(items)

    return sorted(train_records), sorted(val_records)


def dataset_signature(records: list[tuple[Path, int]], size: tuple[int, int]) -> str:
    digest = hashlib.sha256(repr(size).encode())
    for path, label in records:
        stat = path.stat()
        digest.update(f"{path.resolve()}|{label}|{stat.st_size}|{stat.st_mtime_ns}\n".encode())
    return digest.hexdigest()


class ACDCWeatherDataset(Dataset):
    """首次缩放并缓存 uint8 图像；以后从 memmap 读取，避免每轮解码大 PNG。"""

    def __init__(
        self, records: list[tuple[Path, int]], cfg: WeatherConfig, cache_file: Path
    ) -> None:
        if not records:
            raise FileNotFoundError("未找到 ACDC 天气图，请先放置官方 rgb_anon 数据")
        self.records = records
        self.cfg = cfg
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        manifest = cache_file.with_suffix(".json")
        signature = dataset_signature(records, cfg.image_size)
        expected_shape = (len(records), *cfg.image_size, 3)
        reuse = False
        if cache_file.exists() and manifest.exists():
            try:
                info = json.loads(manifest.read_text(encoding="utf-8"))
                reuse = info.get("signature") == signature
                if reuse:
                    cache = np.load(cache_file, mmap_mode="r", allow_pickle=False)
                    reuse = cache.shape == expected_shape and cache.dtype == np.uint8
            except (OSError, ValueError, json.JSONDecodeError):
                reuse = False
        if not reuse:
            if "cache" in locals():
                del cache
            temporary = cache_file.with_name(cache_file.stem + ".partial.npy")
            array = np.lib.format.open_memmap(
                temporary, mode="w+", dtype=np.uint8, shape=expected_shape
            )
            try:
                for i, (path, _) in enumerate(records):
                    array[i] = read_rgb_image(path, cfg.image_size)
                array.flush()
            finally:
                del array
            temporary.replace(cache_file)
            manifest.write_text(json.dumps({"signature": signature}), encoding="utf-8")
            cache = np.load(cache_file, mmap_mode="r", allow_pickle=False)
        self.images = cache
        self.signature = signature

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        pixels, cues = image_tensors(self.images[index], self.cfg)
        return pixels, cues, self.records[index][1]


def checkpoint_model_config(cfg: WeatherConfig) -> dict[str, Any]:
    return {
        "classes": list(cfg.classes),
        "image_size": list(cfg.image_size),
        "channels": list(cfg.channels),
        "features": dict(cfg.features),
    }


def _save_checkpoint(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


@dataclass(frozen=True)
class WeatherTrainResult:
    best_epoch: int
    best_val_loss: float
    val_accuracy: float
    val_macro_recall: float
    checkpoint: Path
    device: str
    train_count: int
    val_count: int


def train_weather(
    cfg: WeatherConfig,
    project_root: str | Path,
    *,
    resume: bool = True,
) -> WeatherTrainResult:
    """验证损失选 best，早停；续训仅接收相同模型配置和数据签名。"""
    root = Path(project_root)
    train_cfg = cfg.train
    try:
        batch_size = int(train_cfg["batch_size"])
        epochs = int(train_cfg["epochs"])
        patience = int(train_cfg["patience"])
        seed = int(train_cfg["seed"])
        validation_fraction = float(train_cfg.get("validation_fraction", 0.2))
        workers = int(train_cfg["num_workers"])
        learning_rate = float(train_cfg["learning_rate"])
        weight_decay = float(train_cfg["weight_decay"])
        smoothing = float(train_cfg["label_smoothing"])
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"天气训练配置无效：{exc}") from exc
    if min(batch_size, epochs, patience) <= 0:
        raise ValueError("batch_size/epochs/patience 必须为正整数")
    if not np.isfinite(validation_fraction) or not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction 必须在 0 和 1 之间")
    if workers < 0 or not np.isfinite([learning_rate, weight_decay, smoothing]).all():
        raise ValueError("天气训练参数必须为非负有限数")
    if learning_rate <= 0 or weight_decay < 0 or not 0 <= smoothing < 1:
        raise ValueError("天气训练学习率、权重衰减或标签平滑范围无效")

    acdc_root = root / train_cfg["acdc_root"]
    official_train = list_condition_images(acdc_root, "train", cfg.classes)
    official_val = list_condition_images(acdc_root, "val", cfg.classes)
    if not official_train or not official_val:
        raise FileNotFoundError("ACDC 官方 train/val 图片不完整，无法训练或选优")
    train_records, val_records = split_by_sequence(
        [*official_train, *official_val], validation_fraction, seed
    )
    for label, condition in enumerate(cfg.classes):
        if not any(y == label for _, y in train_records) or not any(
            y == label for _, y in val_records
        ):
            raise ValueError(f"{condition} 在序列分组后的 train 或 val 划分中无图片")
    train_sequences = {p.parent.name for p, _ in train_records}
    val_sequences = {p.parent.name for p, _ in val_records}
    if train_sequences & val_sequences:
        raise ValueError("按序列切分后仍发现 train/val 视频重叠，停止训练以避免泄漏")

    device = select_device(cfg.device)
    cache_dir = root / train_cfg["cache_dir"]
    train_set = ACDCWeatherDataset(train_records, cfg, cache_dir / "train.npy")
    val_set = ACDCWeatherDataset(val_records, cfg, cache_dir / "val.npy")

    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )
    with torch.random.fork_rng(
        devices=[device.index or torch.cuda.current_device()] if device.type == "cuda" else []
    ):
        torch.manual_seed(seed)
        model = WeatherClassifier(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    loss_fn = nn.CrossEntropyLoss(label_smoothing=smoothing)
    best_path = root / cfg.checkpoint
    last_path = best_path.with_name("last.pt")
    signatures = {"train": train_set.signature, "val": val_set.signature}
    model_cfg = checkpoint_model_config(cfg)
    start_epoch, best_epoch, best_loss, wait = 0, 0, float("inf"), 0
    if resume and last_path.exists():
        previous = torch.load(last_path, map_location="cpu", weights_only=True)
        if previous.get("format_version") != 1 or previous.get("model_config") != model_cfg:
            raise ValueError("天气检查点模型配置不兼容；如需重训请使用 --fresh")
        if previous.get("dataset_signatures") != signatures:
            raise ValueError("天气训练数据变化，不能沿用旧优化器状态；请使用 --fresh")
        model.load_state_dict(previous["model_state"])
        optimizer.load_state_dict(previous["optimizer_state"])
        start_epoch = int(previous["epoch"])
        best_epoch = int(previous["best_epoch"])
        best_loss = float(previous["best_val_loss"])
        wait = int(previous["wait"])
        generator.set_state(previous["shuffle_state"])

    best_accuracy, best_macro_recall = 0.0, 0.0
    if best_epoch > 0 and best_path.exists():
        best = torch.load(best_path, map_location="cpu", weights_only=True)
        if best.get("model_config") != model_cfg or int(best.get("epoch", -1)) != best_epoch:
            raise ValueError("天气 best/last 检查点不匹配；请使用 --fresh")
        best_accuracy = float(best["val_accuracy"])
        best_macro_recall = float(best["val_macro_recall"])
    elif best_epoch > 0:
        raise ValueError("last 检查点引用的 best 权重不存在；请使用 --fresh")
    for epoch in range(start_epoch + 1, epochs + 1):
        model.train()
        for images, cues, labels in train_loader:
            images, cues, labels = images.to(device), cues.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(images, cues), labels)
            loss.backward()
            optimizer.step()
        model.eval()
        total_loss, total_correct, total = 0.0, 0, 0
        class_correct = np.zeros(len(cfg.classes), dtype=np.int64)
        class_total = np.zeros(len(cfg.classes), dtype=np.int64)
        with torch.inference_mode():
            for images, cues, labels in val_loader:
                images, cues, labels = images.to(device), cues.to(device), labels.to(device)
                logits = model(images, cues)
                count = labels.numel()
                total_loss += float(loss_fn(logits, labels).item()) * count
                predicted = logits.argmax(dim=1)
                total_correct += int((predicted == labels).sum().item())
                total += count
                for i in range(len(cfg.classes)):
                    mask = labels == i
                    class_total[i] += int(mask.sum().item())
                    class_correct[i] += int(((predicted == i) & mask).sum().item())
        val_loss = total_loss / total
        accuracy = total_correct / total
        macro_recall = float((class_correct / class_total).mean())
        if val_loss < best_loss:
            best_epoch, best_loss, wait = epoch, val_loss, 0
            best_accuracy, best_macro_recall = accuracy, macro_recall
            _save_checkpoint(
                {
                    "format_version": 1,
                    "model_config": model_cfg,
                    "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "val_accuracy": accuracy,
                    "val_macro_recall": macro_recall,
                },
                best_path,
            )
        else:
            wait += 1
        _save_checkpoint(
            {
                "format_version": 1,
                "model_config": model_cfg,
                "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch,
                "best_epoch": best_epoch,
                "best_val_loss": best_loss,
                "wait": wait,
                "shuffle_state": generator.get_state(),
                "dataset_signatures": signatures,
            },
            last_path,
        )
        if wait >= patience:
            break
    if best_epoch == 0:
        raise ValueError("未找到可用的天气 best 检查点；请检查 epochs 配置")
    return WeatherTrainResult(
        best_epoch,
        best_loss,
        best_accuracy,
        best_macro_recall,
        best_path,
        str(device),
        len(train_set),
        len(val_set),
    )
