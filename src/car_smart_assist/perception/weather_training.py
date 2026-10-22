"""训练天气/光照属性模型，避免场景泄漏。"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import random
import re
from collections import defaultdict
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import numpy as np
import torch
import yaml
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from car_smart_assist.config.weather import WeatherConfig, load_weather_config
from car_smart_assist.data.detection import recording_group
from car_smart_assist.data.preprocessing import load_invalid_entries
from car_smart_assist.perception.visibility.dataset import read_rgb_image
from car_smart_assist.perception.weather import (
    FEATURE_NAMES,
    WeatherClassifier,
    prepare_image,
    select_device,
    visual_cues,
)


@dataclass(frozen=True)
class WeatherSample:
    labels: tuple[float, ...]
    group: str
    stratum: str
    path: Path | None = None
    archive_path: Path | None = None
    member: str | None = None


def _acdc_split(
    acdc_root: Path,
    split: str,
    attributes: tuple[str, ...],
    invalid: set[str] | None = None,
) -> list[WeatherSample]:
    if split not in ("train", "val", "test"):
        raise ValueError("ACDC 天气图片 split 必须是 train/val/test")
    root = acdc_root / "rgb_anon"
    samples: list[WeatherSample] = []
    for condition in attributes:
        folder = root / condition / split
        for path in sorted(folder.rglob("*_rgb_anon.png")) if folder.is_dir() else ():
            if not path.is_file() or str(path.resolve()) in (invalid or ()):
                continue
            labels = tuple(float(attribute == condition) for attribute in attributes)
            samples.append(
                WeatherSample(
                    labels=labels,
                    group=f"acdc:{condition}:{recording_group(path.parent.name, 'recording_family')}",
                    stratum=condition,
                    path=path,
                )
            )
    return samples


def list_condition_images(
    acdc_root: Path,
    split: str,
    attributes: tuple[str, ...],
    invalid: set[str] | None = None,
) -> list[WeatherSample]:
    """列出 ACDC 开发集图像；test 专门留给评估。"""
    if split not in ("train", "val"):
        raise ValueError("训练和选优只允许读取 ACDC 官方 train/val")
    return _acdc_split(acdc_root, split, attributes, invalid)


def split_by_sequence(
    records: list[WeatherSample], validation_fraction: float, seed: int
) -> tuple[list[WeatherSample], list[WeatherSample]]:
    """ACDC 同一天气序列的所有帧放在同一个划分里。"""
    grouped: dict[str, dict[str, list[WeatherSample]]] = defaultdict(lambda: defaultdict(list))
    for sample in records:
        grouped[sample.stratum][sample.group].append(sample)

    train_records: list[WeatherSample] = []
    val_records: list[WeatherSample] = []
    for stratum_index, stratum in enumerate(sorted(grouped)):
        groups = sorted(grouped[stratum].items())
        if len(groups) < 2:
            raise ValueError(f"天气类别 {stratum} 少于两个独立视频序列，无法无泄漏选优")
        random.Random(seed + stratum_index).shuffle(groups)
        total = sum(len(items) for _, items in groups)
        target = min(total - 1, max(1, round(total * validation_fraction)))

        reachable: dict[int, tuple[str, ...]] = {0: ()}
        for group, items in groups:
            for count, selected in list(reachable.items()):
                new_count = count + len(items)
                if new_count < total and new_count not in reachable:
                    reachable[new_count] = (*selected, group)
        val_count = min(
            (count for count in reachable if 0 < count < total),
            key=lambda count: abs(count - target),
        )
        val_groups = set(reachable[val_count])
        for group, items in groups:
            (val_records if group in val_groups else train_records).extend(items)

    return sorted(train_records, key=lambda sample: str(sample.path)), sorted(
        val_records, key=lambda sample: str(sample.path)
    )


def list_pixel_accurate_images(
    archive_path: Path,
    attributes: tuple[str, ...],
    validation_scene: int,
    invalid: set[str] | None = None,
) -> tuple[list[WeatherSample], list[WeatherSample]]:
    """用文件名元数据作多标签目标，并留出一整个场景。"""
    pattern = re.compile(
        r"scene(?P<scene>\d+)_(?P<illumination>day|night)_"
        r"(?P<condition>clear|fog\d+|rain\d+)_(?P<frame>\d+)\.png$",
        re.IGNORECASE,
    )
    train_samples: list[WeatherSample] = []
    val_samples: list[WeatherSample] = []
    with ZipFile(archive_path) as archive:
        for member in archive.namelist():
            if f"{archive_path.resolve()}::{member}" in (invalid or ()):
                continue
            match = pattern.fullmatch(Path(member).name)
            if not match:
                continue
            scene = int(match.group("scene"))
            illumination = match.group("illumination").lower()
            condition = match.group("condition").lower()
            values = {attribute: 0.0 for attribute in attributes}
            if condition.startswith("fog"):
                values["fog"] = 1.0
            elif condition.startswith("rain"):
                values["rain"] = 1.0
            if illumination == "night":
                values["night"] = 1.0
            sample = WeatherSample(
                labels=tuple(values[attribute] for attribute in attributes),
                group=f"pixel-accurate:scene{scene}",
                stratum=f"{illumination}:{condition}",
                archive_path=archive_path,
                member=member,
            )
            (val_samples if scene == validation_scene else train_samples).append(sample)
    if not train_samples or not val_samples:
        raise ValueError(
            f"Pixel Accurate 场景切分无效：训练 {len(train_samples)}，验证 {len(val_samples)}"
        )
    return train_samples, val_samples


def _sample_signature(samples: list[WeatherSample], size: tuple[int, int]) -> str:
    digest = hashlib.sha256(repr(size).encode())
    archive_stats: dict[Path, tuple[int, int]] = {}
    for sample in samples:
        if sample.path is not None:
            stat = sample.path.stat()
            source = f"{sample.path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}"
        else:
            assert sample.archive_path is not None and sample.member is not None
            if sample.archive_path not in archive_stats:
                stat = sample.archive_path.stat()
                archive_stats[sample.archive_path] = (stat.st_size, stat.st_mtime_ns)
            size_bytes, modified = archive_stats[sample.archive_path]
            source = f"{sample.archive_path.resolve()}|{size_bytes}|{modified}|{sample.member}"
        digest.update(f"{source}|{sample.labels}\n".encode())
    return digest.hexdigest()


class WeatherDataset(Dataset):
    """缓存缩放后的 uint8 图像和视觉线索；直接读 ZIP 里的图像，不解压。"""

    def __init__(self, samples: list[WeatherSample], cfg: WeatherConfig, cache_file: Path) -> None:
        if not samples:
            raise FileNotFoundError("没有可用于天气模型的图片")
        self.samples = samples
        self.cfg = cfg
        self.targets = np.asarray([sample.labels for sample in samples], dtype=np.float32)
        self.signature = hashlib.sha256(
            (
                _sample_signature(samples, cfg.image_size)
                + json.dumps(cfg.features, sort_keys=True)
            ).encode()
        ).hexdigest()
        # Windows 下正在读取的 memmap 不能覆盖；不同数据／线索配置使用独立缓存文件。
        cache_file = cache_file.with_name(f"{cache_file.stem}_{self.signature[:16]}.npy")
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cues_file = cache_file.with_name(cache_file.stem + "_cues.npy")
        manifest = cache_file.with_suffix(".json")
        expected_shape = (len(samples), *cfg.image_size, 3)
        expected_cues_shape = (len(samples), len(FEATURE_NAMES))
        reuse = False
        if cache_file.exists() and cues_file.exists() and manifest.exists():
            try:
                info = json.loads(manifest.read_text(encoding="utf-8"))
                cache = np.load(cache_file, mmap_mode="r", allow_pickle=False)
                cached_cues = np.load(cues_file, mmap_mode="r", allow_pickle=False)
                reuse = (
                    info.get("signature") == self.signature
                    and cache.shape == expected_shape
                    and cache.dtype == np.uint8
                    and cached_cues.shape == expected_cues_shape
                    and cached_cues.dtype == np.float32
                )
            except (OSError, ValueError, json.JSONDecodeError):
                reuse = False
        if not reuse:
            logging.getLogger(__name__).info("构建天气缓存：%d 张 -> %s", len(samples), cache_file)
            if "cache" in locals():
                del cache
            if "cached_cues" in locals():
                del cached_cues
            image_temp = cache_file.with_name(cache_file.stem + ".partial.npy")
            cues_temp = cues_file.with_name(cues_file.stem + ".partial.npy")
            image_cache = np.lib.format.open_memmap(
                image_temp, mode="w+", dtype=np.uint8, shape=expected_shape
            )
            cue_cache = np.lib.format.open_memmap(
                cues_temp, mode="w+", dtype=np.float32, shape=expected_cues_shape
            )
            try:
                with ExitStack() as stack:
                    archives = {
                        archive_path: stack.enter_context(ZipFile(archive_path))
                        for archive_path in {sample.archive_path for sample in samples}
                        if archive_path is not None
                    }
                    for index, sample in enumerate(samples):
                        if sample.path is not None:
                            image = read_rgb_image(sample.path, cfg.image_size)
                        else:
                            assert sample.archive_path is not None and sample.member is not None
                            with Image.open(
                                io.BytesIO(archives[sample.archive_path].read(sample.member))
                            ) as source:
                                image = prepare_image(source, cfg.image_size)
                        image_cache[index] = image
                        cues = visual_cues(image, cfg)
                        cue_cache[index] = [cues[name] for name in FEATURE_NAMES]
                        if (index + 1) % 256 == 0:
                            logging.getLogger(__name__).info(
                                "天气缓存进度：%d/%d", index + 1, len(samples)
                            )
                image_cache.flush()
                cue_cache.flush()
            finally:
                del image_cache
                del cue_cache
            image_temp.replace(cache_file)
            cues_temp.replace(cues_file)
            manifest.write_text(json.dumps({"signature": self.signature}), encoding="utf-8")
            cache = np.load(cache_file, mmap_mode="r", allow_pickle=False)
            cached_cues = np.load(cues_file, mmap_mode="r", allow_pickle=False)
        self.images = cache
        self.cues = cached_cues

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pixels = torch.from_numpy(np.array(self.images[index], copy=True)).permute(2, 0, 1)
        pixels = pixels.float().div_(255.0)
        cues = torch.from_numpy(np.array(self.cues[index], copy=True))
        labels = torch.from_numpy(self.targets[index].copy())
        return pixels, cues, labels


def checkpoint_model_config(cfg: WeatherConfig) -> dict[str, Any]:
    result = {
        "attributes": list(cfg.attributes),
        "image_size": list(cfg.image_size),
        "channels": list(cfg.channels),
        "features": dict(cfg.features),
    }
    return result


def _save_checkpoint(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def require_independent_weather_output(project_root: Path, destination: Path) -> None:
    """训练输出与当前默认部署目录分离；路由损坏时拒绝猜测。"""
    root = Path(project_root).resolve()
    route_path = root / "configs/model/weather_default.yaml"
    if route_path.exists():
        try:
            route = yaml.safe_load(route_path.read_text(encoding="utf-8"))["weather_default"]
            if (
                not isinstance(route, dict)
                or set(route) != {"config", "checkpoint"}
                or any(not isinstance(value, str) or not value for value in route.values())
            ):
                raise ValueError("需要 config 和 checkpoint 两条路径")
            checkpoint = root / route["checkpoint"]
        except (KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
            raise ValueError(f"默认天气路由无效，停止训练以保护部署产物：{exc}") from exc
    else:
        fallback = root / "configs/model/weather_classifier.yaml"
        if not fallback.is_file():
            return
        checkpoint = root / load_weather_config(fallback).checkpoint
    destination = Path(destination)
    if not destination.is_absolute():
        destination = root / destination
    if destination.resolve() == checkpoint.resolve().parent:
        raise ValueError(
            f"训练输出目录包含受保护的默认天气权重：{checkpoint.resolve()}。"
            "请将训练配置的 checkpoint_root/checkpoint 指向独立实验目录；"
            "普通恢复和 --fresh 均不能写入部署目录，训练完成后须另行明确发布。"
        )


def _archive_weather_checkpoints(paths: tuple[Path, ...]) -> None:
    existing = {path: path.read_bytes() for path in dict.fromkeys(paths) if path.exists()}
    if not existing:
        return
    folder = paths[0].parent / "history" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    folder.mkdir(parents=True)
    for path, content in existing.items():
        backup = folder / path.name
        with backup.open("xb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        if backup.read_bytes() != content:
            raise OSError(f"旧 CNN 检查点归档回读不一致，保留原件并停止：{backup}")
    if any(path.read_bytes() != content for path, content in existing.items()):
        raise OSError("归档期间旧 CNN 检查点发生变化，保留原件并停止")
    for path in existing:
        path.unlink()


def _load_verified_weather_best(path: Path, reference: dict[str, Any]) -> dict[str, Any]:
    expected = reference.get("best_sha256")
    advice = (
        "请保留原件并使用有匹配 best/last 的原目录；或使用 --fresh 归档后重开，"
        "或选择独立输出并另行明确仅权重初始化方案（当前入口不会自动回退）。"
    )
    if (
        not isinstance(expected, str)
        or len(expected) != 64
        or any(char not in "0123456789abcdef" for char in expected)
    ):
        raise ValueError(f"旧 CNN 历史缺少可核验的 best 身份，拒绝完整恢复。{advice}")
    if not path.is_file():
        raise ValueError(f"旧 CNN 历史 best 缺失：{path}，拒绝完整恢复。{advice}")
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != expected:
        raise ValueError(f"旧 CNN best/last 摘要不配对：{path}，拒绝完整恢复。{advice}")
    best = torch.load(io.BytesIO(content), map_location="cpu", weights_only=True)
    for best_key, reference_key in (
        ("format_version", "format_version"),
        ("model_config", "model_config"),
        ("dataset_signatures", "dataset_signatures"),
        ("epoch", "best_epoch"),
        ("val_loss", "best_val_loss"),
        ("val_metrics", "best_val_metrics"),
    ):
        if (
            best_key not in best
            or reference_key not in reference
            or best[best_key] != reference[reference_key]
        ):
            raise ValueError(f"旧 CNN best/last 的 {best_key} 不配对，拒绝完整恢复。{advice}")
    return best


def attribute_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    attributes: tuple[str, ...],
    thresholds: dict[str, float],
) -> dict[str, Any]:
    predicted = np.stack(
        [probabilities[:, index] >= thresholds[name] for index, name in enumerate(attributes)],
        axis=1,
    )
    actual = targets.astype(bool)
    per_attribute: dict[str, dict[str, Any]] = {}
    f1s: list[float] = []
    tp_all = fp_all = fn_all = 0
    for index, name in enumerate(attributes):
        tp = int(np.logical_and(actual[:, index], predicted[:, index]).sum())
        fp = int(np.logical_and(~actual[:, index], predicted[:, index]).sum())
        fn = int(np.logical_and(actual[:, index], ~predicted[:, index]).sum())
        support = int(actual[:, index].sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_attribute[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
            "threshold": thresholds[name],
        }
        if support:
            f1s.append(f1)
        tp_all += tp
        fp_all += fp
        fn_all += fn
    micro_precision = tp_all / (tp_all + fp_all) if tp_all + fp_all else 0.0
    micro_recall = tp_all / (tp_all + fn_all) if tp_all + fn_all else 0.0
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if micro_precision + micro_recall
        else 0.0
    )
    return {
        "sample_count": int(len(targets)),
        "exact_match_accuracy": float((actual == predicted).all(axis=1).mean()),
        "macro_f1": float(np.mean(f1s)) if f1s else 0.0,
        "micro_f1": micro_f1,
        "per_attribute": per_attribute,
    }


@dataclass(frozen=True)
class WeatherTrainResult:
    best_epoch: int
    best_val_loss: float
    val_metrics: dict[str, Any]
    checkpoint: Path
    device: str
    train_count: int
    val_count: int
    pixel_train_count: int
    pixel_val_count: int


def train_weather(
    cfg: WeatherConfig,
    project_root: str | Path,
    *,
    resume: bool = True,
) -> WeatherTrainResult:
    """用按场景分组的 ACDC 和 Pixel 数据训练天气与光照属性。"""
    root = Path(project_root).resolve()
    best_path = root / cfg.checkpoint
    last_path = best_path.with_name("last.pt")
    require_independent_weather_output(root, best_path.parent)
    if best_path == last_path:
        raise ValueError("旧 CNN 的 checkpoint 不能命名为 last.pt；best 与 last 必须分开")
    if resume and best_path.exists() and not last_path.exists():
        raise ValueError(
            "旧 CNN 只有 best、缺少配对 last，不能完整恢复；请保留原件，"
            "使用 --fresh 归档后重开或选择独立输出并明确仅权重初始化方案。"
        )
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
        pos_weight_cap = float(train_cfg.get("pos_weight_cap", 5.0))
        use_pixel = bool(train_cfg.get("use_pixel_accurate", True))
        validation_scene = int(train_cfg.get("pixel_accurate_validation_scene", 4))
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"天气训练配置无效：{exc}") from exc
    if min(batch_size, epochs, patience) <= 0:
        raise ValueError("batch_size/epochs/patience 必须为正整数")
    if not np.isfinite(validation_fraction) or not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction 必须在 0 和 1 之间")
    if workers < 0 or not np.isfinite([learning_rate, weight_decay, pos_weight_cap]).all():
        raise ValueError("天气训练参数必须为非负有限数")
    if learning_rate <= 0 or weight_decay < 0 or pos_weight_cap <= 0:
        raise ValueError("学习率必须为正数，权重衰减和正样本权重上限必须为非负数")

    acdc_root = root / train_cfg["acdc_root"]
    invalid = load_invalid_entries(
        root / train_cfg.get("acdc_cleaning_manifest", "data/processed/manifests/acdc.json"), root
    )
    official_train = list_condition_images(acdc_root, "train", cfg.attributes, invalid)
    official_val = list_condition_images(acdc_root, "val", cfg.attributes, invalid)
    if not official_train or not official_val:
        raise FileNotFoundError("ACDC 官方 train/val 图片不完整，无法训练或选优")
    acdc_train, acdc_val = split_by_sequence(
        [*official_train, *official_val], validation_fraction, seed
    )
    if {sample.group for sample in acdc_train} & {sample.group for sample in acdc_val}:
        raise ValueError("ACDC 按序列切分后仍有 train/val 重叠，停止训练以避免泄漏")

    pixel_train: list[WeatherSample] = []
    pixel_val: list[WeatherSample] = []
    if use_pixel:
        pixel_archive = root / train_cfg["pixel_accurate_zip"]
        if not pixel_archive.is_file():
            raise FileNotFoundError(f"Pixel Accurate RGB 压缩包不存在：{pixel_archive}")
        pixel_train, pixel_val = list_pixel_accurate_images(
            pixel_archive,
            cfg.attributes,
            validation_scene,
            load_invalid_entries(
                root
                / train_cfg.get(
                    "pixel_cleaning_manifest",
                    "data/processed/manifests/pixel_accurate_benchmark.json",
                ),
                root,
            ),
        )
    train_samples = [*acdc_train, *pixel_train]
    val_samples = [*acdc_val, *pixel_val]
    for index, attribute in enumerate(cfg.attributes):
        train_values = {sample.labels[index] for sample in train_samples}
        val_values = {sample.labels[index] for sample in val_samples}
        if len(train_values) < 2 or len(val_values) < 2:
            raise ValueError(f"属性 {attribute} 在 train 或 val 中缺少正/负样本")

    device = select_device(cfg.device)
    cache_dir = root / train_cfg["cache_dir"]
    train_signature = _sample_signature(train_samples, cfg.image_size)[:16]
    val_signature = _sample_signature(val_samples, cfg.image_size)[:16]
    train_set = WeatherDataset(train_samples, cfg, cache_dir / f"train_{train_signature}.npy")
    val_set = WeatherDataset(val_samples, cfg, cache_dir / f"val_{val_signature}.npy")
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
    positives = train_set.targets.sum(axis=0)
    pos_weight = np.minimum((len(train_set) - positives) / np.maximum(positives, 1), pos_weight_cap)
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.as_tensor(pos_weight, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    signatures = {"train": train_set.signature, "val": val_set.signature}
    model_cfg = checkpoint_model_config(cfg)
    start_epoch, best_epoch, best_loss, wait = 0, 0, float("inf"), 0
    best_metrics: dict[str, Any] = {}
    best_reference: dict[str, Any] = {}
    best_sha256 = None
    if not resume:
        _archive_weather_checkpoints((best_path, last_path))
    if resume and last_path.exists():
        previous = torch.load(last_path, map_location="cpu", weights_only=True)
        previous_model_cfg = dict(previous.get("model_config", {}))
        previous_model_cfg.pop("decision_thresholds", None)
        if previous.get("format_version") != 2 or previous_model_cfg != model_cfg:
            raise ValueError("天气属性模型检查点不兼容；请使用 --fresh")
        if previous.get("dataset_signatures") != signatures:
            raise ValueError("天气训练数据变化，不能沿用旧优化器状态；请使用 --fresh")
        _load_verified_weather_best(best_path, previous)
        best_reference = previous
        best_sha256 = previous["best_sha256"]
        model.load_state_dict(previous["model_state"])
        optimizer.load_state_dict(previous["optimizer_state"])
        start_epoch = int(previous["epoch"])
        best_epoch = int(previous["best_epoch"])
        best_loss = float(previous["best_val_loss"])
        wait = int(previous["wait"])
        best_metrics = previous["best_val_metrics"]
        generator.set_state(previous["shuffle_state"])

    for epoch in range(start_epoch + 1, epochs + 1):
        model.train()
        for images, cues, labels in train_loader:
            images = images.to(device, non_blocking=True)
            cues = cues.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(images, cues), labels)
            loss.backward()
            optimizer.step()

        model.eval()
        total_loss = 0.0
        label_count = 0
        all_targets: list[np.ndarray] = []
        all_probabilities: list[np.ndarray] = []
        with torch.inference_mode():
            for images, cues, labels in val_loader:
                images = images.to(device, non_blocking=True)
                cues = cues.to(device, non_blocking=True)
                labels_device = labels.to(device, non_blocking=True)
                logits = model(images, cues)
                count = labels.numel()
                total_loss += float(loss_fn(logits, labels_device).item()) * count
                label_count += count
                all_targets.append(labels.numpy())
                all_probabilities.append(torch.sigmoid(logits.float()).cpu().numpy())
        val_loss = total_loss / label_count
        metrics = attribute_metrics(
            np.concatenate(all_targets),
            np.concatenate(all_probabilities),
            cfg.attributes,
            cfg.decision_thresholds,
        )
        if val_loss < best_loss:
            best_epoch, best_loss, wait = epoch, val_loss, 0
            best_metrics = metrics
            _save_checkpoint(
                {
                    "format_version": 2,
                    "model_config": model_cfg,
                    "model_state": {
                        key: value.detach().cpu() for key, value in model.state_dict().items()
                    },
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "val_metrics": metrics,
                    "dataset_signatures": signatures,
                },
                best_path,
            )
            best_sha256 = hashlib.sha256(best_path.read_bytes()).hexdigest()
        else:
            wait += 1
        best_reference = {
            "format_version": 2,
            "model_config": model_cfg,
            "model_state": {
                key: value.detach().cpu() for key, value in model.state_dict().items()
            },
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "best_epoch": best_epoch,
            "best_val_loss": best_loss,
            "best_val_metrics": best_metrics,
            "wait": wait,
            "shuffle_state": generator.get_state(),
            "dataset_signatures": signatures,
            "best_sha256": best_sha256,
        }
        _save_checkpoint(best_reference, last_path)
        if wait >= patience:
            break
    if best_epoch == 0:
        raise ValueError("未找到可用的天气属性模型 best 检查点")
    best_checkpoint = _load_verified_weather_best(best_path, best_reference)
    model.load_state_dict(best_checkpoint["model_state"])
    model.eval()
    final_targets: list[np.ndarray] = []
    final_probabilities: list[np.ndarray] = []
    with torch.inference_mode():
        for images, cues, labels in val_loader:
            logits = model(images.to(device, non_blocking=True), cues.to(device, non_blocking=True))
            final_targets.append(labels.numpy())
            final_probabilities.append(torch.sigmoid(logits.float()).cpu().numpy())
    best_metrics = attribute_metrics(
        np.concatenate(final_targets),
        np.concatenate(final_probabilities),
        cfg.attributes,
        cfg.decision_thresholds,
    )
    return WeatherTrainResult(
        best_epoch,
        best_loss,
        best_metrics,
        best_path,
        str(device),
        len(train_set),
        len(val_set),
        len(pixel_train),
        len(pixel_val),
    )
