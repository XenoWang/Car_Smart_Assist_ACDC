"""能见度打分：重建误差 + 信息量特征。

职责:
    - 用训练好的 AE 算重建误差，并以「正常天气参考图」的误差分布做零校准（z-score）
    - 同时算一组确定性的信息量特征（对比度 / 熵 / 边缘密度 / 高频能量占比）
    - 输出 VisibilityScore，供 gate.py 做判定

为什么必须同时给两类信号（本模块的核心设计）:
    只用重建误差会在**低信息量帧**上失效，而且方向是反的：

      纯黑帧      结构极简、几乎是常数图  -> AE 容易重建 -> 误差低 -> 被判正常 ✗
      白茫茫浓雾  接近均匀灰度、低秩      -> AE 容易重建 -> 误差低 -> 被判正常 ✗

    这两类恰恰是「最该报警」的。原因是重建误差衡量的是「像不像训练集里的样子」，
    而「看不清」的本质是「画面里还剩多少信息」—— 两者不是一回事。

    所以本模块把两路信号正交地分开：
      重建误差 z 分数  -> 「这张图像不像一个正常可见的驾驶场景」
      信息量分数        -> 「这张图里还有没有东西」
    任何一路单独都不足以判定，见 gate.py 的组合规则。

    这个失效模式是实测发现的，不是推测 —— 见
    artifacts/reports/visibility/ 下的分数分布报告。
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch

from car_smart_assist.config.visibility import MODEL_DEFAULTS, SCORING_DEFAULTS
from car_smart_assist.perception.visibility.autoencoder import (
    ConvAutoencoder,
    build_autoencoder,
    reconstruction_error,
    reconstruction_mean,
)
from car_smart_assist.perception.visibility.dataset import read_rgb_image

logger = logging.getLogger(__name__)


@lru_cache(maxsize=16)
def _high_frequency_mask(height: int, width: int, cutoff: float) -> np.ndarray:
    """缓存频域高频掩膜，避免每帧重新创建坐标网格与开方数组。"""
    fy = (2.0 * np.fft.fftfreq(height))[:, None]
    fx = (2.0 * np.fft.fftfreq(width))[None, :]
    mask = np.hypot(fy, fx) > cutoff
    mask.setflags(write=False)
    return mask


# =============================================================================
# 信息量特征（确定性、无参考、无训练）
# =============================================================================


@dataclass
class InformationFeatures:
    """一帧的确定性信息量特征。

    全部无参考（不需要真值），且每一维都能单独画分布做验证 ——
    这是刻意选择的：当门控判错时，我们要能立刻指出是哪一维出了问题，
    而不是面对一个 256 维黑箱潜向量无从下手。
    """

    contrast: float        # 灰度标准差，直接量对比度
    entropy: float         # 灰度直方图熵，量灰度分布的丰富度
    edge_density: float    # 梯度幅值超阈值的像素占比，量结构多寡
    hf_ratio: float        # 高频能量占比（傅里叶），雾/模糊/黑暗都会压低它

    def to_dict(self) -> dict[str, float]:
        return {
            "contrast": self.contrast,
            "entropy": self.entropy,
            "edge_density": self.edge_density,
            "hf_ratio": self.hf_ratio,
        }


def compute_information_features(
    img: np.ndarray, scoring_cfg: dict[str, Any] | None = None
) -> InformationFeatures:
    """从 (H, W, 3) uint8 或 float[0,1] 图像算信息量特征。

    实现在 numpy 上而非 torch：这些特征要为每一帧单独解释，
    用 numpy 便于直接对照数值排查，也不需要 GPU。
    """
    a = img.astype(np.float32)
    if a.max() > 1.0:
        a /= 255.0
    gray = a @ np.array([0.299, 0.587, 0.114], dtype=np.float32)

    # --- 对比度：灰度标准差 ---
    contrast = float(gray.std())

    # --- 熵：直方图熵，与灰度分布是否丰富有关 ---
    hist, _ = np.histogram(gray, bins=256, range=(0.0, 1.0))
    p = hist.astype(np.float64) / max(1, hist.sum())
    p = p[p > 0]
    entropy = float(-(p * np.log2(p)).sum())

    # --- 边缘密度：梯度幅值超过阈值的像素占比 ---
    # 用 np.gradient（中心差分）而非 Sobel：Sobel 的 3×3 核自带平滑，
    # 会把高频噪声也一并抹掉，而这里要的恰恰是「还剩多少高频结构」。
    # 阈值来自 scoring_cfg；默认 0.04 是 [0,1] 灰度量纲下的值（≈ 10/255，与 preprocessing.py 的
    # 0-255 量纲阈值 10.0 等价 —— 两边量纲不同但物理含义一致）。
    gy, gx = np.gradient(gray)
    mag = np.hypot(gx, gy)
    feature_cfg = (scoring_cfg or {}).get("features", {})
    edge_threshold = float(feature_cfg.get(
        "edge_gradient_threshold", SCORING_DEFAULTS["features"]["edge_gradient_threshold"]
    ))
    edge_density = float((mag > edge_threshold).mean())

    # --- 高频能量占比 ---
    # 用径向掩膜把频谱分成低频(内 50% 半径)与高频(外 50%)两部分。
    # 雾、雨、模糊、黑暗的共同效应就是高频塌陷 —— 这是四类退化里最一致的信号。
    f = np.fft.fft2(gray - gray.mean())
    power = np.abs(f) ** 2
    h, w = gray.shape
    total = power.sum()
    hf_cutoff = float(feature_cfg.get("hf_cutoff_ratio", SCORING_DEFAULTS["features"]["hf_cutoff_ratio"]))
    if total > 0:
        high_frequency = np.sum(
            power, where=_high_frequency_mask(h, w, hf_cutoff)
        )
        hf_ratio = float(high_frequency / total)
    else:
        hf_ratio = 0.0

    return InformationFeatures(
        contrast=contrast,
        entropy=entropy,
        edge_density=edge_density,
        hf_ratio=hf_ratio,
    )


# =============================================================================
# 打分结果
# =============================================================================


@dataclass
class VisibilityScore:
    """一帧的完整能见度打分。"""

    path: str
    # 重建误差（原始值）与其相对正常天气分布的 z 分数
    recon_mean: float
    recon_p90_block: float
    recon_z: float
    # 信息量特征与其归一化后的综合分 [0, 1]
    features: InformationFeatures
    information: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "recon_mean": self.recon_mean,
            "recon_p90_block": self.recon_p90_block,
            "recon_z": self.recon_z,
            "information": self.information,
            **self.features.to_dict(),
        }


# =============================================================================
# 归一化：把各特征映射到可比的 [0,1]
# =============================================================================

# 参考尺度。取值来自对 ACDC 全量的实测分位数（见 configs/model/visibility.yaml
# 的标定说明）。兼容默认值集中在 config/visibility.py，运行参数在 YAML 调整。
#
# ⚠️ 单位约定（这里踩过一次坑，务必看清）：
#   compute_information_features 会先把图像归一化到 [0, 1] 再算特征，
#   因此 **contrast 的量纲是 [0,1] 而不是 [0,255]**。
#   ACDC 实测灰度标准差中位数约 61/255 ≈ 0.24，所以尺度取 0.25。
#   早期版本误按 0-255 量纲填了 64.0，导致 contrast 归一化后恒为 ~0.004，
#   几何平均被整体拽到接近 0 —— 所有图的信息量分数一起塌陷、阈值全线失效，
#   而表面上不会报任何错。tests/unit/test_visibility.py 里有针对这一点的测试。
_FEATURE_SCALES: dict[str, float] = dict(SCORING_DEFAULTS["feature_scales"])


def normalize_features(
    f: InformationFeatures, scales: dict[str, float] | None = None
) -> dict[str, float]:
    """把各特征除以参考尺度并截断到 [0,1]。"""
    scales = scales or _FEATURE_SCALES
    return {
        "contrast": min(1.0, f.contrast / scales["contrast"]),
        "entropy": min(1.0, f.entropy / scales["entropy"]),
        "edge_density": min(1.0, f.edge_density / scales["edge_density"]),
        "hf_ratio": min(1.0, f.hf_ratio / scales["hf_ratio"]),
    }


# 信息量分数的默认聚合参数。改这里要同步重跑 scripts/evaluate_visibility.py
# 重新标定 gate 阈值 —— 分数尺度变了，旧阈值全部失效。
#
# 权重分配的依据（逐特征实测，见 information_score 的 docstring）：
#   contrast      0.40  雾 / 黑暗 / 遮挡 的主判据
#   entropy       0.25  遮挡 / 黑暗
#   edge_density  0.25  模糊的主判据（且分级良好：0.759→0.135）
#   hf_ratio      0.10  仅作辅助。它**在轻度模糊时就饱和**（0.25~1.0 强度下
#                       恒为 0.024），不具备分级能力，因此不给高权重
_INFO_WEIGHTS: dict[str, float] = dict(SCORING_DEFAULTS["aggregation"]["weights"])
# 广义平均的阶数：p<0 时趋近最小值
_INFO_P: float = SCORING_DEFAULTS["aggregation"]["power"]
# 单维下限。防止某一维取到接近 0 时在 p<0 的幂运算中绝对支配总分 ——
# 那会让「一个饱和的噪声维度」把分数钉死，反而制造误报。
# 取 0.10 的含义：任何一维最多只能把总分压到约 1/(0.1·w) 的量级。
_INFO_EPS: float = SCORING_DEFAULTS["aggregation"]["feature_floor"]


def information_score(
    f: InformationFeatures,
    weights: dict[str, float] | None = None,
    p: float | None = None,
    eps: float | None = None,
    scoring_cfg: dict[str, Any] | None = None,
) -> float:
    """把归一化后的特征合成单一信息量分数（加权广义平均）。

    ⚠️ 这里用**广义平均且 p<0（趋近最小值）**，而不是算术或几何平均。
    这是实测逼出来的修正，不是理论上更优雅的选择。

    M_p(x) = ( Σ w_i · x_i^p / Σ w_i )^(1/p)

        p → +∞  趋近最大值      —— 完全不适合本任务
        p = 1   算术平均        —— 一维塌陷会被其他维度稀释
        p → 0   几何平均        —— 比算术敏感，但稀释仍然严重
        p = -1  调和式平均      —— 强烈偏向低值，但保留一点鲁棒性
        p → -∞  趋近最小值      —— 最敏感，但一维噪声就会误报

    为什么必须偏向最小值（逐特征实测，severity=1.0）：

        退化         contrast  entropy  edge   hf_ratio   几何平均下的 info
        clean         0.989    0.932   0.989    0.252        0.645
        blur          0.904    0.929   0.135    0.024        0.228  ← 漏报
        occlusion     0.148    0.033   0.070    0.989        0.170  ← 漏报

      模糊只杀高频、不动大尺度明暗，所以 contrast 纹丝不动，
      几何平均被它撑住；遮挡则相反，覆盖区的**边界**是强边缘，
      hf_ratio 反而涨到 0.989，把对比度与熵的塌陷抵消掉。

      每一种退化都至少有一维崩了，但加权平均把它们全部稀释掉了。
      而「看不清」的定义就是「**只要**有一路信号说画面空了，就该报警」——
      这是「或」的逻辑，不是「与」的逻辑。广义平均 p<0 正是这个语义的平滑实现。

    ⚠️ eps 下限不可省（默认 0.10）：
      p<0 的幂运算里，一维取到 0.024 会贡献 41.7，取到 0.001 会贡献 1000 ——
      单维就能把总分完全钉死。实测中 hf_ratio 在**轻度模糊**时就已经饱和到 0.024，
      不加下限会导致「轻微模糊」与「完全看不见」得到同样的分数，
      于是轻微模糊也被判 BLIND —— 把误报造了出来。
      eps 的作用是限制任何单维的最大支配权，让分数保持分级能力。

    权重默认见 _INFO_WEIGHTS（contrast/entropy/edge_density 主导，hf_ratio 仅辅助）。
    """
    cfg = scoring_cfg or {}
    aggregate_cfg = cfg.get("aggregation", {})
    n = normalize_features(f, cfg.get("feature_scales"))
    w = weights or aggregate_cfg.get("weights") or _INFO_WEIGHTS
    p = float(aggregate_cfg.get("power", _INFO_P) if p is None else p)
    eps = float(aggregate_cfg.get("feature_floor", _INFO_EPS) if eps is None else eps)
    w_sum = sum(w.values())
    acc = 0.0
    for k, weight in w.items():
        acc += weight * (max(n[k], eps) ** p)
    return float((acc / max(w_sum, eps)) ** (1.0 / p))


# =============================================================================
# 打分器
# =============================================================================


@dataclass
class CalibrationStats:
    """正常天气参考图上的重建误差分布，用于零校准。"""

    mean: float
    std: float
    p95: float
    p99: float
    n: int

    @classmethod
    def from_errors(cls, errors: Sequence[float] | np.ndarray) -> CalibrationStats:
        """训练与手动校准共用统计口径，拒绝空数据及非有限误差。"""
        values = np.asarray(errors, dtype=np.float64)
        if values.ndim != 1 or values.size == 0:
            raise ValueError("校准需要非空的一维重建误差序列")
        if not np.isfinite(values).all():
            raise ValueError("校准重建误差包含 NaN 或无穷大")
        p95, p99 = np.percentile(values, [95, 99])
        return cls(
            mean=float(values.mean()), std=float(values.std()),
            p95=float(p95), p99=float(p99), n=int(values.size),
        )

    def to_dict(self) -> dict[str, float | int]:
        return {"mean": self.mean, "std": self.std, "p95": self.p95, "p99": self.p99, "n": self.n}


class VisibilityScorer:
    """加载训练好的 AE，对图像算能见度分数。

    Args:
        model: 已训练的 ConvAutoencoder
        calibration: 正常天气参考图上的误差分布。为 None 时 z 分数不可用，
            只能看原始误差 —— 那种情况下不应该做判定，因此 gate 会拒绝工作。
        device: 推理设备
        input_size: 需与模型一致
        scoring_cfg: 特征尺度、聚合权重及分块误差参数；随训练检查点固定
    """

    def __init__(
        self,
        model: ConvAutoencoder,
        calibration: CalibrationStats | None = None,
        device: str | torch.device = "cpu",
        input_size: tuple[int, int] = MODEL_DEFAULTS["input_size"],
        scoring_cfg: dict[str, Any] | None = None,
    ) -> None:
        self.model = model.to(device).eval()
        self.calibration = calibration
        self.device = torch.device(device)
        self.input_size = tuple(input_size)
        self.scoring_cfg = scoring_cfg or {}

    # --- 构建 ---

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        device: str | torch.device = "cpu",
        cfg: dict[str, Any] | None = None,
        scoring_cfg: dict[str, Any] | None = None,
    ) -> VisibilityScorer:
        """从训练 checkpoint 恢复。checkpoint 内嵌模型配置与校准统计。"""
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model_cfg = ckpt.get("model_cfg", cfg or {})
        model = build_autoencoder(model_cfg)
        model.load_state_dict(ckpt["model_state"])
        cal = ckpt.get("calibration")
        calibration = CalibrationStats(**cal) if cal else None
        # 调用方显式提供的配置代表当前实验/部署策略，应优先于 checkpoint 快照。
        # 未提供时再回退到训练时保存的值，保证独立加载仍可复现。
        effective_scoring_cfg = (
            scoring_cfg if scoring_cfg is not None else ckpt.get("scoring_cfg", {})
        )
        if calibration is None:
            logger.warning(
                "checkpoint 中没有校准统计，z 分数不可用。"
                "判定需要以正常天气分布为零点，请用 scripts/train_visibility.py 重新训练。"
            )
        return cls(
            model,
            calibration,
            device=device,
            input_size=tuple(model_cfg.get("input_size", MODEL_DEFAULTS["input_size"])),
            scoring_cfg=effective_scoring_cfg,
        )

    # --- 打分 ---

    def _to_tensor_batch(self, images: Sequence[np.ndarray]) -> torch.Tensor:
        """一次完成 batch 堆叠、通道转换、类型转换和设备传输。"""
        # np.stack 会生成可写的 uint8 连续数组；随后在一次 dtype/内存格式
        # 转换中直接得到 NCHW float32，避免逐图分配张量再 torch.cat。
        batch = np.stack(images, axis=0)
        x = torch.from_numpy(batch).permute(0, 3, 1, 2).to(
            dtype=torch.float32, memory_format=torch.contiguous_format
        )
        x.div_(255.0)
        return x.to(self.device)

    @torch.no_grad()
    def score_tensors(self, x: torch.Tensor) -> list[tuple[float, float]]:
        """对一批张量算 (recon_mean, recon_p90_block)。"""
        block_grid = tuple(
            self.scoring_cfg.get("reconstruction_error", {}).get(
                "block_grid", SCORING_DEFAULTS["reconstruction_error"]["block_grid"]
            )
        )
        errs = reconstruction_error(self.model, x.to(self.device), block_grid=block_grid)
        return [(e.mean, e.p90_block) for e in errs]

    def score_arrays(self, images: Sequence[np.ndarray], paths: Sequence[str] | None = None):
        """对一组已解码的 (H,W,3) uint8 数组打分。"""
        if not images:
            return []
        batch = self._to_tensor_batch(images)
        recon = self.score_tensors(batch)

        cal = self.calibration
        out: list[VisibilityScore] = []
        for i, img in enumerate(images):
            feat = compute_information_features(img, self.scoring_cfg)
            mean, p90 = recon[i]
            z = (mean - cal.mean) / cal.std if cal and cal.std > 0 else float("nan")
            out.append(
                VisibilityScore(
                    path=str(paths[i]) if paths else f"<array:{i}>",
                    recon_mean=mean,
                    recon_p90_block=p90,
                    recon_z=z,
                    features=feat,
                    information=information_score(feat, scoring_cfg=self.scoring_cfg),
                )
            )
        return out

    def score_paths(self, paths: Sequence[str | Path]) -> list[VisibilityScore]:
        """对文件路径逐个解码并打分。批量小、只用于调试或小规模评估。"""
        images: list[np.ndarray] = []
        ok_paths: list[str] = []
        for p in paths:
            try:
                images.append(read_rgb_image(p, self.input_size))
                ok_paths.append(str(p))
            except Exception as exc:  # noqa: BLE001
                logger.error("无法读取 %s: %s", p, exc)
        return self.score_arrays(images, ok_paths)

    # --- 校准 ---

    def calibrate(self, images: Sequence[np.ndarray]) -> CalibrationStats:
        """用一组正常天气图重新计算校准统计。"""
        if not images:
            raise ValueError("校准至少需要一张正常天气参考图")
        means = reconstruction_mean(self.model, self._to_tensor_batch(images))
        stats = CalibrationStats.from_errors(means.cpu().tolist())
        self.calibration = stats
        return stats
