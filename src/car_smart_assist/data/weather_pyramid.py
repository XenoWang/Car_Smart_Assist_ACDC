"""新天气实验的完整录制组四分和高分辨率缓存。"""

from __future__ import annotations

import hashlib
import io
import itertools
import json
import logging
import os
import random
import tempfile
from collections import Counter, defaultdict
from contextlib import ExitStack
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch
from PIL import Image
from filelock import FileLock
from torch.utils.data import Dataset

from car_smart_assist.config.weather_pyramid import WeatherPyramidConfig
from car_smart_assist.data.detection import recording_group
from car_smart_assist.data.preprocessing import load_invalid_entries
from car_smart_assist.perception.weather_pyramid import prepare_pyramid_image
from car_smart_assist.perception.weather_training import (
    WeatherSample,
    _acdc_split,
    _sample_signature,
    list_pixel_accurate_images,
)

logger = logging.getLogger(__name__)


def weather_source(sample: WeatherSample) -> str:
    return "acdc" if sample.path is not None else "pixel_accurate"


def sample_identifier(sample: WeatherSample) -> str:
    return (
        str(sample.path.resolve())
        if sample.path is not None
        else f"{sample.archive_path.resolve()}::{sample.member}"
    )


def known_mask(samples: list[WeatherSample]) -> np.ndarray:
    # ACDC 原论文规定单一恶劣条件采集；PAB 的天气／昼夜也有明确元数据。
    return np.ones((len(samples), 4), dtype=np.float32)


def collect_pyramid_samples(cfg: WeatherPyramidConfig, root: Path) -> list[WeatherSample]:
    data = cfg.data
    invalid_by_source = {}
    for source, key in (("acdc", "acdc_cleaning_manifest"), ("pixel_accurate", "pixel_cleaning_manifest")):
        configured = data.get(key)
        if configured is None:
            invalid_by_source[source] = set()
            continue
        manifest = root / configured
        if not manifest.is_file():
            dataset = "acdc" if source == "acdc" else "pixel_accurate_benchmark"
            raise FileNotFoundError(
                f"清洗清单不存在：{manifest}。请先进行数据清洗。\n"
                f"在项目根目录（{root}）运行：\n"
                rf".\.venv\Scripts\python.exe -X utf8 scripts\clean_data.py --only {dataset} --corrupt-images-only"
            )
        invalid_by_source[source] = load_invalid_entries(manifest, root)
    invalid = invalid_by_source["acdc"]
    records = [
        s
        for split in data["acdc_splits"]
        for s in _acdc_split(root / data["acdc_root"], split, cfg.attributes, invalid)
    ]
    if data["include_normal_references"]:
        root_images = root / data["acdc_root"] / "rgb_anon"
        for condition in cfg.attributes:
            for split in data["acdc_splits"]:
                folder = root_images / condition / f"{split}_ref"
                for path in sorted(folder.rglob("*_rgb_ref_anon.png")):
                    if str(path.resolve()) in invalid:
                        continue
                    group = (
                        f"acdc:{condition}:{recording_group(path.parent.name, 'recording_family')}"
                    )
                    records.append(
                        WeatherSample((0, 0, 0, 0), group, f"normal:{condition}", path=path)
                    )
    pixel_train, pixel_val = list_pixel_accurate_images(
        root / data["pixel_zip"],
        cfg.attributes,
        4,
        invalid_by_source["pixel_accurate"],
    )
    records.extend((*pixel_train, *pixel_val))
    if len({sample_identifier(s) for s in records}) != len(records):
        raise ValueError("天气图片清单有重复条目")
    return sorted(records, key=sample_identifier)


def _assign_groups(groups: dict[str, list[WeatherSample]], fractions, seed, fixed=None):
    if len(groups) < 4:
        raise ValueError("每个天气来源／主条件至少需要四个完整录制组，才能覆盖四分")
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    fixed = {key: value for key, value in (fixed or {}).items() if key in groups}
    free = [key for key in keys if key not in fixed]
    sizes = [len(groups[k]) for k in keys]
    total = sum(sizes)
    best, best_score = None, float("inf")
    # 本地数据每个主条件只有 4—8 组，穷举比随机逐帧划分更可靠。
    for choices_free in itertools.product(range(4), repeat=len(free)):
        current = {**fixed, **dict(zip(free, choices_free, strict=True))}
        choices = tuple(current[k] for k in keys)
        if len(set(choices)) != 4:
            continue
        counts = [0] * 4
        for choice, size in zip(choices, sizes, strict=True):
            counts[choice] += size
        score = sum(
            (count / total - fraction) ** 2
            for count, fraction in zip(counts, fractions, strict=True)
        )
        if score < best_score - 1e-15:
            best, best_score = choices, score
    if best is None:
        raise ValueError("重复图片绑定后，无法使各天气条件覆盖四个集合")
    return {key: choice for key, choice in zip(keys, best, strict=True)}, best_score


def split_pyramid_samples(
    samples: list[WeatherSample], ratios: dict[str, float], seed: int, duplicate_links=()
):
    names = ("train", "validation", "calibration", "test")
    domains = defaultdict(lambda: defaultdict(list))
    for sample in samples:
        domain = (
            f"acdc:{sample.stratum.split(':')[-1]}" if sample.path is not None else "pixel_accurate"
        )
        domains[domain][sample.group].append(sample)
    acdc_count = sum(sample.path is not None for sample in samples)
    if acdc_count == 0:
        raise ValueError(
            f"天气四分需要 ACDC 样本；ACDC=0，pixel_accurate={len(samples)}，total={len(samples)}"
        )
    fractions = [ratios[n] for n in names]
    assignments = {}
    pixel_groups = domains.pop("pixel_accurate")
    assignments.update(_assign_groups(pixel_groups, fractions, seed)[0])
    pixel_counts = [
        sum(len(items) for group, items in pixel_groups.items() if assignments[group] == i)
        for i in range(4)
    ]
    remaining = len(samples) - sum(pixel_counts)
    adjusted = [max(0, len(samples) * fractions[i] - pixel_counts[i]) / remaining for i in range(4)]
    adjusted = [v / sum(adjusted) for v in adjusted]
    components = []
    for link in duplicate_links:
        joined = set(link)
        overlapping = [c for c in components if c & joined]
        for component in overlapping:
            joined.update(component)
            components.remove(component)
        components.append(joined)
    best_assignment, best_score = None, float("inf")
    for choices in itertools.product(range(4), repeat=len(components)):
        fixed = {
            group: choice
            for component, choice in zip(components, choices, strict=True)
            for group in component
        }
        candidate, score = {}, 0.0
        try:
            for index, domain in enumerate(sorted(domains)):
                allocation, domain_score = _assign_groups(
                    domains[domain], adjusted, seed + index + 1, fixed
                )
                candidate.update(allocation)
                score += domain_score
        except ValueError:
            continue
        if score < best_score - 1e-15:
            best_assignment, best_score = candidate, score
    if best_assignment is None:
        raise ValueError("无法在保留录制组和重复图片的情况下完成四分")
    assignments.update(best_assignment)
    splits = {name: [] for name in names}
    for sample in samples:
        splits[names[assignments[sample.group]]].append(sample)
    groups = [set(s.group for s in splits[n]) for n in names]
    if any(groups[i] & groups[j] for i in range(4) for j in range(i)):
        raise ValueError("天气四分存在重复录制组")
    if sum(map(len, splits.values())) != len(samples):
        raise ValueError("天气划分没有完整保留输入图片")
    if any(len({assignments[group] for group in link}) > 1 for link in duplicate_links):
        raise ValueError("相同图片跨越了不同集合")
    return splits


def _duplicate_recording_links(samples, cache_dir):
    records = [s for s in samples if s.path is not None]
    signature = _sample_signature(records, (0, 0))
    output = cache_dir / f"file_hashes_{signature[:16]}.json"
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        cached = json.loads(output.read_text(encoding="utf-8"))
    except FileNotFoundError:
        cached = None
    except (json.JSONDecodeError, UnicodeDecodeError):
        cached = None
    valid_groups = {sample.group for sample in records}
    if (
        isinstance(cached, dict) and cached.get("signature") == signature
        and isinstance(cached.get("cross_recording_links"), list)
        and all(
            isinstance(link, list) and len(link) >= 2
            and all(type(group) is str and group and group in valid_groups for group in link)
            and len(set(link)) == len(link)
            for link in cached["cross_recording_links"]
        )
    ):
        return cached["cross_recording_links"]
    groups = defaultdict(set)
    for index, sample in enumerate(records):
        digest = hashlib.sha256()
        with sample.path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        groups[digest.hexdigest()].add(sample.group)
        if (index + 1) % 1000 == 0:
            logger.info("天气文件重复检查 %d/%d", index + 1, len(records))
    links = sorted(sorted(group) for group in groups.values() if len(group) > 1)
    content = json.dumps({"signature": signature, "cross_recording_links": links}).encode("utf-8")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix=output.name + ".", suffix=".tmp", dir=cache_dir, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        if temporary.read_bytes() != content:
            raise OSError(f"天气重复检查缓存临时文件回读不一致：{temporary}")
        temporary.replace(output)
        temporary = None
        if output.read_bytes() != content:
            raise OSError(f"天气重复检查缓存最终回读不一致：{output}")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return links


def prepare_pyramid_splits(cfg: WeatherPyramidConfig, root: Path):
    samples = collect_pyramid_samples(cfg, root)
    duplicate_links = _duplicate_recording_links(samples, root / cfg.data["cache_dir"])
    splits = split_pyramid_samples(
        samples, cfg.data["split_ratios"], cfg.data["seed"], duplicate_links
    )
    signature = hashlib.sha256(
        (
            _sample_signature(samples, (0, 0))
            + json.dumps(
                {
                    "seed": cfg.data["seed"],
                    "ratios": cfg.data["split_ratios"],
                    "acdc_splits": cfg.data["acdc_splits"],
                    "algorithm": "whole-group-fourfold-v2",
                    "acdc_label_policy": "single_adversity_protocol_2021",
                    "include_normal_references": cfg.data["include_normal_references"],
                    "duplicate_links": duplicate_links,
                },
                sort_keys=True,
            )
        ).encode()
    ).hexdigest()
    info = {
        "signature": signature,
        "seed": cfg.data["seed"],
        "target_ratios": cfg.data["split_ratios"],
        "acdc_original_splits": cfg.data["acdc_splits"],
        "total_samples": len(samples),
        "protocol": "project_fourfold",
        "group_overlap": 0,
        "acdc_label_policy": "single_adversity_protocol_2021",
        "normal_references": sum(s.stratum.startswith("normal:") for s in samples),
        "cross_recording_duplicate_links": duplicate_links,
        "splits": {},
    }
    for name, records in splits.items():
        labels, mask = np.asarray([s.labels for s in records]), known_mask(records)
        info["splits"][name] = {
            "samples": len(records),
            "fraction": len(records) / len(samples),
            "sources": dict(Counter(weather_source(s) for s in records)),
            "normal_references": sum(s.stratum.startswith("normal:") for s in records),
            "groups": sorted({s.group for s in records}),
            "known_positive": dict(
                zip(cfg.attributes, (labels * mask).sum(axis=0).astype(int).tolist(), strict=True)
            ),
            "known_negative": dict(
                zip(
                    cfg.attributes,
                    ((1 - labels) * mask).sum(axis=0).astype(int).tolist(),
                    strict=True,
                )
            ),
            "records": [
                {
                    "id": sample_identifier(s),
                    "group": s.group,
                    "source": weather_source(s),
                    "context": s.stratum,
                    "labels": list(s.labels),
                    "mask": m.tolist(),
                }
                for s, m in zip(records, mask, strict=True)
            ],
        }
    output = root / cfg.data["cache_dir"] / f"splits_{signature[:16]}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(info, ensure_ascii=False, indent=2)
    with FileLock(str(output.resolve()) + ".lock"):
        if not output.exists() or output.read_text(encoding="utf-8") != serialized:
            _write_cache_metadata(output, info)
    info["manifest"] = str(output)
    return splits, info


def _write_cache_metadata(path: Path, value):
    content = json.dumps(value, ensure_ascii=False, indent=2).encode("utf-8")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=path.name + ".", suffix=".tmp", dir=path.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        if temporary.read_bytes() != content:
            raise OSError(f"天气缓存记录临时文件回读不一致：{temporary}")
        temporary.replace(path)
        temporary = None
        if path.read_bytes() != content:
            raise OSError(f"天气缓存记录最终回读不一致：{path}")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _load_pyramid_cache(samples, cfg, output, signature):
    """调用方持有缓存路径锁；读取者只会打开完整发布的数组。"""
    manifest = output.with_suffix(".json")
    expected_shape = (len(samples), *cfg.image_size, 3)
    cached = None
    if output.exists() and manifest.exists():
        try:
            info = json.loads(manifest.read_text(encoding="utf-8"))
            cached = np.load(output, mmap_mode="r", allow_pickle=False)
            if (
                info["signature"] == signature
                and cached.shape == expected_shape
                and cached.dtype == np.uint8
            ):
                return cached
        except (OSError, ValueError, KeyError, TypeError, EOFError):
            pass
        if cached is not None:
            cached._mmap.close()
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=output.stem + ".", suffix=".partial.npy", dir=output.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
        array = np.lib.format.open_memmap(
            temporary, mode="w+", dtype=np.uint8, shape=expected_shape
        )
        try:
            with ExitStack() as stack:
                archives = {
                    p: stack.enter_context(ZipFile(p))
                    for p in {s.archive_path for s in samples}
                    if p is not None
                }
                for index, sample in enumerate(samples):
                    source = (
                        sample.path
                        if sample.path is not None
                        else io.BytesIO(archives[sample.archive_path].read(sample.member))
                    )
                    with Image.open(source) as image:
                        array[index] = prepare_pyramid_image(image, cfg.image_size)
                    if (index + 1) % 256 == 0:
                        logger.info("新天气缓存 %d/%d", index + 1, len(samples))
            array.flush()
        finally:
            array._mmap.close()
        with temporary.open("r+b") as stream:
            os.fsync(stream.fileno())
        candidate = np.load(temporary, mmap_mode="r", allow_pickle=False)
        try:
            if candidate.shape != expected_shape or candidate.dtype != np.uint8:
                raise ValueError(f"天气缓存候选形状或类型无效：{temporary}")
        finally:
            candidate._mmap.close()
        temporary.replace(output)
        temporary = None
        _write_cache_metadata(manifest, {"signature": signature})
        return np.load(output, mmap_mode="r", allow_pickle=False)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class PyramidDataset(Dataset):
    def __init__(self, samples: list[WeatherSample], cfg: WeatherPyramidConfig, cache_dir: Path):
        self.samples = samples
        self.targets = np.asarray([s.labels for s in samples], dtype=np.float32)
        self.mask = known_mask(samples)
        self.signature = hashlib.sha256(
            (_sample_signature(samples, cfg.image_size) + "rgb-letterbox-bilinear-v1").encode()
        ).hexdigest()
        output = cache_dir / f"images_{self.signature[:16]}.npy"
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            with FileLock(str(output.resolve()) + ".lock"):
                self.images = _load_pyramid_cache(samples, cfg, output, self.signature)
        except PermissionError as exc:
            raise ValueError(
                f"无法读取或发布天气图像缓存：{output}。缓存可能仍被其他训练进程或 "
                "DataLoader 映射，也可能没有写权限；请结束占用进程、核对目录权限，"
                "或设置独立的 data.cache_dir 后重试。"
            ) from exc

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        pixels = torch.from_numpy(np.array(self.images[index], copy=True)).permute(2, 0, 1)
        return (
            pixels,
            torch.from_numpy(self.targets[index].copy()),
            torch.from_numpy(self.mask[index].copy()),
        )


def pyramid_sampling_weights(dataset: PyramidDataset):
    keys = [(weather_source(s), s.labels) for s in dataset.samples]
    counts = Counter(keys)
    combinations = Counter(source for source, _ in counts)
    return torch.tensor(
        [1 / counts[key] / combinations[key[0]] for key in keys], dtype=torch.float64
    )
