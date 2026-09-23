"""能见度门控的数据集。

职责:
    - 收集训练/评分用的图像路径（ACDC 正常天气参考图 / 恶劣天气图）
    - 统一缩放到固定尺寸并归一化，供自编码器消费
    - 用磁盘 npy 缓存规避「每个 epoch 重新解码 4006 张 1920×1080 PNG」的开销

为什么需要磁盘缓存:
    ACDC 原图 1920×1080，解码 + 缩放单张约 40~60ms。
    4006 张一轮就是 3~4 分钟，训 50 轮要三个小时，而这些像素每轮完全一样。
    首次读入后压成 uint8 存成 npy（约 710 MB），之后用 memmap 直接读，
    单轮降到秒级。这不是过早优化，是没有它这个模型根本训不动。

    缓存放在 data/processed/ 下，已被 .gitignore 覆盖。

被谁调用: scripts/train_visibility.py, scorer.VisibilityScorer
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# ACDC 目录约定（见 docs/dataset.md 4.1）
#   rgb_anon/{condition}/{split}/{sequence}/{sequence}_frame_{n:06d}_rgb_anon.png
#   rgb_anon/{condition}/{split}_ref/.../{sequence}_frame_{n:06d}_rgb_ref_anon.png
ADVERSE_SUFFIX = "_rgb_anon.png"
REF_SUFFIX = "_rgb_ref_anon.png"


def list_ref_images(acdc_root: str | Path) -> list[Path]:
    """列出全部正常天气参考图。

    这是 AE 的训练集 —— 它们是「清晰可见」的无标注样本，
    用它们定义 normal 分布，才让重建误差具备判别力。
    """
    root = Path(acdc_root) / "rgb_anon"
    return sorted(p for p in root.rglob("*.png") if p.name.endswith(REF_SUFFIX))


def list_adverse_images(
    acdc_root: str | Path, conditions: Sequence[str] | None = None
) -> list[Path]:
    """列出恶劣天气图。conditions 为 None 时返回全部四类。"""
    root = Path(acdc_root) / "rgb_anon"
    paths = [p for p in root.rglob("*.png") if p.name.endswith(ADVERSE_SUFFIX)]
    if conditions:
        want = set(conditions)
        paths = [p for p in paths if p.relative_to(root).parts[0] in want]
    return sorted(paths)


def _build_cache(paths: Sequence[Path], size: tuple[int, int], cache: Path) -> np.ndarray:
    """把图像解码、缩放、存成 npy。返回 uint8 数组 (N, H, W, 3)。"""
    h, w = size
    cache.parent.mkdir(parents=True, exist_ok=True)
    arr = np.lib.format.open_memmap(
        cache, mode="w+", dtype=np.uint8, shape=(len(paths), h, w, 3)
    )
    failed: list[tuple[str, str]] = []
    for i, p in enumerate(paths):
        try:
            with Image.open(p) as im:
                # BILINEAR 而非 LANCZOS：缩放是预处理，不是关键路径；
                # 且 LANCZOS 的锐化会人为抬高高频能量，干扰「糊没糊」的判据
                arr[i] = np.asarray(im.convert("RGB").resize((w, h), Image.BILINEAR))
        except Exception as exc:  # noqa: BLE001
            failed.append((str(p), f"{type(exc).__name__}: {exc}"))
            arr[i] = 0
    arr.flush()
    if failed:
        logger.warning("%d 张图解码失败，已置零，请先跑 scripts/clean_data.py：%s", len(failed), failed[:3])
    return np.asarray(arr)


def sequence_of(path: str | Path) -> str:
    """从 ACDC 路径中取出序列名（文件名上一级目录）。

    路径形如 .../rgb_anon/{condition}/{split}/{sequence}/{seq}_frame_{n}_rgb_ref_anon.png
    """
    return Path(path).parent.name


def split_ref_indices(
    n: int,
    train_ratio: float = 0.80,
    calib_ratio: float = 0.10,
    test_ratio: float = 0.10,
    seed: int = 42,
    mode: str = "fixed",
    groups: Sequence[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """把参考图划成 train / calib / test 三份。

    为什么是三分而不是两分（train / calib）:
        反复用同一套 calib 迭代（挑轮数、挑阈值、挑超参），calib 会被间接拟合 ——
        这就是验证集泄漏。跑上十次之后，calib 上的指标已经不代表泛化能力了。
        所以必须留一份**从头到尾不参与任何决策**的 test：
        它只在最终报告里出现，你迭代一百次也污染不到它。
        这不是洁癖，是「报告的指标要有意义」的最低条件。

为什么默认 fixed 而不是每轮重随机:
        重随机会让两次运行的指标测在不同数据上，于是
        「第二次比第一次好」无法判断是模型变好了，还是这次抽到的集合更容易。
        迭代优化需要一个固定基准；对安全门控来说，阈值标定也必须稳定。
        想达到「每张图都被测到」的效果，正确做法是 k 折交叉验证，
        而不是每次换一套划分 —— 后者连可比性都丢了。

        mode='random' 仍然提供（每次用不同的种子），但只建议在
        「不在乎跨运行可比性、只想让模型见过更多数据」时用，
        且**不要**在这种模式下标定阈值。

    Args:
        mode: 'fixed' 用固定 seed（可复现、跨运行可比）；'random' 每次换种子。
        groups: 每张图所属的组（通常是序列名）。**强烈建议传入。**
            ACDC 的图像来自视频序列，相邻帧近乎重复。按图随机划分会把
            某帧放进训练集、把它的邻居放进校准集 —— 等于校准集里混进了
            训练样本的复制品，测出来的泛化能力是虚高的。
            传入 groups 后按**整组**划分，同一序列不会跨集合。
            None 表示按图划分（仅用于没有序列信息的数据）。

    Returns:
        (train_idx, calib_idx, test_idx)，均为**已排序**的索引数组。
        排序是刻意的：缓存是按顺序 memmap 的，排序后的索引访问局部性更好。
    """
    if mode not in ("fixed", "random"):
        raise ValueError(f"未知 split_mode {mode!r}，可选: fixed | random")

    total = train_ratio + calib_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"train/calib/test 比例之和应为 1.0，收到 {total:.4f}"
            f"（{train_ratio} + {calib_ratio} + {test_ratio}）"
        )

    if mode == "random":
        seed = int(np.random.SeedSequence().entropy % (2**31))
        logger.warning(
            "split_mode=random：本次划分种子为 %d，与历史运行**不可比**。"
            "该模式下不要标定阈值 —— 阈值必须基于固定的 calib 集。", seed,
        )

    rng = np.random.default_rng(seed)

    if groups is None:
        idx = rng.permutation(n)
        n_train = int(n * train_ratio)
        n_calib = int(n * calib_ratio)
        return (
            np.sort(idx[:n_train]),
            np.sort(idx[n_train : n_train + n_calib]),
            np.sort(idx[n_train + n_calib :]),
        )

    # --- 按组划分 ---
    if len(groups) != n:
        raise ValueError(f"groups 长度 {len(groups)} 与样本数 {n} 不一致")

    # 把每个组映射成索引列表
    by_group: dict[str, list[int]] = {}
    for i, g in enumerate(groups):
        by_group.setdefault(str(g), []).append(i)

    names = np.array(sorted(by_group))
    perm = rng.permutation(len(names))

    # 目标：按**图数**尽量贴近给定比例（组大小不一，按组数切会偏离很多）
    target_train = n * train_ratio
    target_calib = n * calib_ratio
    train_g: list[str] = []
    calib_g: list[str] = []
    acc = 0
    for k, gi in enumerate(perm):
        gname = names[gi]
        size = len(by_group[gname])
        if acc < target_train:
            train_g.append(gname)
            acc += size
        elif acc < target_train + target_calib:
            calib_g.append(gname)
            acc += size
        else:
            break
    used = set(train_g) | set(calib_g)
    test_g = [str(x) for x in names if str(x) not in used]

    def collect(gs: Sequence[str]) -> np.ndarray:
        out = [i for g in gs for i in by_group[g]]
        return np.sort(np.asarray(out, dtype=np.int64))

    tr, ca, te = collect(train_g), collect(calib_g), collect(test_g)
    logger.info(
        "按序列划分：%d 个序列 -> 训练 %d 组(%d 张) / 校准 %d 组(%d 张) / 测试 %d 组(%d 张)",
        len(names), len(train_g), len(tr), len(calib_g), len(ca), len(test_g), len(te),
    )
    # 组大小不均时比例会有偏差，实测出来而不是假装精确
    actual = np.array([len(tr), len(ca), len(te)]) / max(n, 1)
    if np.abs(actual - np.array([train_ratio, calib_ratio, test_ratio])).max() > 0.05:
        logger.warning(
            "实际划分比例 %s 与目标 %s 偏差较大（序列大小不均所致）",
            np.round(actual, 3).tolist(),
            [train_ratio, calib_ratio, test_ratio],
        )
    return tr, ca, te


def ensure_cache(
    paths: Sequence[Path], size: tuple[int, int], cache_path: str | Path | None
) -> Path | None:
    """确保磁盘缓存存在，返回它的路径（不返回数组）。cache_path 为 None 时返回 None。

    单独抽出来是为了让**全量**参考图的缓存只构建一次，
    再由 train / calibration 两个子集通过索引共享 —— 见 VisibilityImageDataset 的
    indices 参数。

    返回路径而不是数组，是因为 DataLoader 的 worker 需要 pickle 数据集：
    把 memmap 对象挂在数据集上会让 pickle 试图序列化整个 443MB 数组，
    在 Windows 的 spawn 模式下直接抛 `pickle data was truncated`。
    只传路径、由每个 worker 自己惰性打开，才是正确做法。

    早期版本还有第二个 bug：给每个子集都传 cache_path，靠「缓存条目数 == 路径数」
    判断能否复用；子集长度与全量不等，于是每个子集都重建一次缓存并互相覆盖，
    既浪费几分钟、又让最后一次的子集变成唯一有效内容。
    """
    if cache_path is None:
        return None
    cp = Path(cache_path)
    if not cp.exists():
        logger.info("构建图像缓存（%d 张 -> %s）...", len(paths), cp)
        _build_cache(paths, size, cp)
    return cp


class VisibilityImageDataset(Dataset):
    """能见度门控的图像数据集。

    Args:
        paths: 图像路径列表（全量，与缓存的行一一对应）
        input_size: (H, W)，需与 AE 的 input_size 一致
        cache_path: npy 缓存**路径**（不是数组）。传 None 则实时解码。
            每个 worker 进程在首次访问时惰性打开自己的 memmap，
            因此数据集本身可以安全地被 pickle（只带一个字符串）。
        indices: 本数据集实际使用 paths 中的哪些行。None 表示全部。
            用于从同一份全量缓存切出 train / calibration 子集。
        augment: 训练时是否做轻微增强。
            ⚠️ 只允许光度抖动，**不允许模糊、降对比度、加雾** ——
            把退化当增强喂给 AE，等于教它「退化也是正常的」，判别力会被自己毁掉。

    Note:
        惰性打开 memmap 是必需的，不能改成在 __init__ 里打开后存 `self._cache`：
        DataLoader 用 spawn 启动 worker 时要 pickle 数据集，
        memmap 的 pickle 会内联整个数组数据，443MB × 4 worker 直接截断报错。
    """

    def __init__(
        self,
        paths: Sequence[Path],
        input_size: tuple[int, int] = (144, 256),
        cache_path: str | Path | None = None,
        indices: Sequence[int] | None = None,
        augment: bool = False,
    ) -> None:
        self.paths = [Path(p) for p in paths]
        self.input_size = tuple(input_size)
        self.augment = augment
        self.indices = list(indices) if indices is not None else None
        self.cache_path = Path(cache_path) if cache_path is not None else None
        # 注意：这里是**每个进程各自打开**的，不是共享对象
        self._cache: np.ndarray | None = None

    def _cache_arr(self) -> np.ndarray | None:
        """惰性打开缓存。首次调用发生在 worker 进程内。"""
        if self._cache is None and self.cache_path is not None:
            if not self.cache_path.exists():
                raise FileNotFoundError(
                    f"图像缓存不存在: {self.cache_path}。"
                    "请先运行 scripts/train_visibility.py 让 ensure_cache 构建它"
                )
            self._cache = np.load(self.cache_path, mmap_mode="r")
        return self._cache

    def __len__(self) -> int:
        return len(self.indices) if self.indices is not None else len(self.paths)

    def __getitem__(self, i: int) -> torch.Tensor:
        idx = self.indices[i] if self.indices is not None else i
        cache = self._cache_arr()
        if cache is not None:
            img = np.asarray(cache[idx])                             # (H, W, 3) uint8
        else:
            with Image.open(self.paths[idx]) as im:
                img = np.asarray(
                    im.convert("RGB").resize(
                        (self.input_size[1], self.input_size[0]), Image.BILINEAR
                    )
                )

        # 显式拷贝而非 ascontiguousarray：
        # memmap 是只读的，torch.from_numpy 会共享内存并发出
        # 「non-writable tensor」警告；后续 div_ 又是原地操作，语义上不该写回缓存。
        x = torch.from_numpy(np.array(img, dtype=np.float32)).div_(255.0)   # -> [0, 1]

        if self.augment:
            # 仅光度：轻微亮度/对比度抖动，模拟曝光差异，不改变「看得清」这一属性
            x = (x - 0.5) * float(torch.empty(1).uniform_(0.9, 1.1)) + 0.5
            x = x + float(torch.empty(1).uniform_(-0.03, 0.03))
            x = x.clamp_(0.0, 1.0)

        return x.permute(2, 0, 1).contiguous()                        # -> (3, H, W)

    def path_of(self, i: int) -> Path:
        idx = self.indices[i] if self.indices is not None else i
        return self.paths[idx]
