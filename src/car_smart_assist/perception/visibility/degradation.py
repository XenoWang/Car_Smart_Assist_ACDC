"""合成退化：为能见度门控生成「确定无疑看不见」的验证样本。

职责:
    - 把清晰图像人为退化成雾 / 黑暗 / 遮挡 / 模糊，用于度量门控的召回率
    - 提供可控的退化强度，用于验证分数随退化程度的单调性

为什么这是必需的（而不是可选项）:
    「看不清」没有现成标签，无法直接算召回率。而一个不知道召回率的
    安全门控等于没有 —— 我们无法回答「真的看不见时它会不会报警」。
    合成退化的价值在于：**退化的强度是我们自己设的，
    所以「这一张是看不见的」是我们确知的真值。**

    于是可以定量回答：
      · 正确率   —— 合成的 1000 张「看不见」里，门控抓到几张
      · 单调性   —— 退化从轻到重时，分数是否单调上升
                   （不单调说明模型在某个区间行为反常，比单纯的低召回更值得查）
      · 误报率   —— 未退化的清晰图有多少被误判（应当接近 0）

局限（必须写清楚，不要当成真值）:
    合成退化与真实恶劣天气不完全一致 —— 真实的雾有非均匀的浓度分布、
    真实夜间的噪声模型也不同于 gamma 拉伸。因此这里的召回率是
    「在上界意义上」的估计，不能替代真实场景验证。这一点在报告里要写明。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class DegradationSpec:
    """一次退化的描述。"""

    kind: str
    severity: float          # 0~1，越大越严重
    params: dict[str, float]


def _rng(seed: int | None) -> np.random.Generator:
    return np.random.default_rng(seed)


def add_fog(img: np.ndarray, severity: float, rng: np.random.Generator,
            atmospheric_light: float = 1.0) -> tuple[np.ndarray, DegradationSpec]:
    """雾模拟：I = J * t + A * (1 - t)。

    这是大气散射模型的标准形式，也是暗通道去雾的基础。
    severity 越大透射率 t 越小，图像越接近纯大气光 —— severity=1 时完全是白板。
    """
    # t 从 1.0（无雾）线性降到约 0.03（几乎全白）
    t = float(np.clip(1.0 - 0.97 * severity, 0.03, 1.0))
    out = img.astype(np.float32) / 255.0 * t + atmospheric_light * (1.0 - t)
    return (np.clip(out, 0, 1) * 255).astype(np.uint8), DegradationSpec("fog", severity, {"transmission": t})


def add_darkness(img: np.ndarray, severity: float, rng: np.random.Generator,
                 gamma_max: float = 6.0) -> tuple[np.ndarray, DegradationSpec]:
    """黑暗模拟：gamma 拉伸压暗中间调 + 线性缩放整体亮度。

    两者叠加而不是只用其一：单纯缩放会让亮部仍可见，
    单纯 gamma 又压不黑高光区，合起来才接近「完全无照明」。
    """
    gamma = 1.0 + (gamma_max - 1.0) * severity
    scale = float(np.clip(1.0 - 0.98 * severity, 0.02, 1.0))
    out = (img.astype(np.float32) / 255.0) ** gamma * scale
    return (np.clip(out, 0, 1) * 255).astype(np.uint8), DegradationSpec(
        "darkness", severity, {"gamma": gamma, "scale": scale}
    )


def add_occlusion(img: np.ndarray, severity: float, rng: np.random.Generator,
                  coverage_max: float = 0.98) -> tuple[np.ndarray, DegradationSpec]:
    """遮挡模拟：模拟镜头被泥污/雪覆盖。

    用低频噪声生成一块不规则的覆盖区域，而不是简单贴矩形 ——
    矩形遮挡会让「边缘密度」这类特征突然升高（遮挡边界本身是强边缘），
    给模型一个虚假的信号。低频噪声生成的覆盖没有这种人为边缘。
    """
    h, w = img.shape[:2]
    coverage = float(np.clip(coverage_max * severity, 0.0, 1.0))
    # 低分辨率噪声放大成平滑的覆盖掩膜
    small = rng.random((max(2, h // 32), max(2, w // 32))).astype(np.float32)
    mask = np.array(
        _bicubic_resize(small, (h, w))
    )
    thr = np.quantile(mask, 1.0 - coverage) if coverage > 0 else 1.1
    m = (mask >= thr).astype(np.float32)
    # 覆盖物取图像自身的中位亮度附近的灰白，避免引入训练集外的奇怪颜色
    fill = float(np.median(img)) * 0.8 + 40.0
    out = img.astype(np.float32) * (1 - m[..., None]) + fill * m[..., None]
    return (np.clip(out, 0, 255)).astype(np.uint8), DegradationSpec(
        "occlusion", severity, {"coverage": coverage}
    )


def add_blur(img: np.ndarray, severity: float, rng: np.random.Generator,
             kernel_max: int = 41) -> tuple[np.ndarray, DegradationSpec]:
    """模糊模拟：高斯模糊。用可分离卷积实现，避免引入 scipy 依赖。"""
    sigma = 0.5 + (kernel_max / 6.0 - 0.5) * severity
    radius = int(max(1, round(sigma * 3)))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    k = np.exp(-(x**2) / (2 * sigma**2))
    k /= k.sum()

    a = img.astype(np.float32)
    # 先横向卷积再纵向卷积
    a = np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), axis=1, arr=a)
    a = np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), axis=0, arr=a)
    return np.clip(a, 0, 255).astype(np.uint8), DegradationSpec(
        "blur", severity, {"sigma": sigma}
    )


def _bicubic_resize(arr: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """双线性放大。手写而非调 cv2/PIL：这里只需要一个平滑的低频掩膜，
    不值得为它引入额外的图像库调用路径，也便于单测。"""
    h, w = shape
    sh, sw = arr.shape
    yi = np.linspace(0, sh - 1, h)
    xi = np.linspace(0, sw - 1, w)
    y0 = np.floor(yi).astype(int)
    x0 = np.floor(xi).astype(int)
    y1 = np.minimum(y0 + 1, sh - 1)
    x1 = np.minimum(x0 + 1, sw - 1)
    wy = (yi - y0)[:, None]
    wx = (xi - x0)[None, :]
    top = arr[y0][:, x0] * (1 - wx) + arr[y0][:, x1] * wx
    bot = arr[y1][:, x0] * (1 - wx) + arr[y1][:, x1] * wx
    return top * (1 - wy) + bot * wy


_KINDS = {
    "fog": add_fog,
    "darkness": add_darkness,
    "occlusion": add_occlusion,
    "blur": add_blur,
}


def degrade(
    img: np.ndarray, kind: str, severity: float, seed: int | None = None
) -> tuple[np.ndarray, DegradationSpec]:
    """对 (H, W, 3) uint8 图像施加指定退化。

    Args:
        kind: fog | darkness | occlusion | blur
        severity: 0~1
    """
    if kind not in _KINDS:
        raise ValueError(f"未知退化类型 {kind!r}，可选: {sorted(_KINDS)}")
    if not 0.0 <= severity <= 1.0:
        raise ValueError(f"severity 需在 [0,1]，收到 {severity}")
    return _KINDS[kind](img, severity, _rng(seed))


def degrade_all_kinds(
    img: np.ndarray, severity: float, seed: int | None = None
) -> dict[str, tuple[np.ndarray, DegradationSpec]]:
    """对同一张图施加全部四种退化，便于横向对比。"""
    return {k: degrade(img, k, severity, seed) for k in _KINDS}


def severity_sweep(
    img: np.ndarray, kind: str, severities: list[float], seed: int | None = None
) -> list[tuple[float, np.ndarray, DegradationSpec]]:
    """对同一张图做一系列强度的退化，用于验证分数的单调性。"""
    return [(s, *degrade(img, kind, s, seed)) for s in severities]


def kinds(cfg: dict[str, Any] | None = None) -> list[str]:
    """可用的退化类型。cfg 传入时只返回配置中启用的那些。"""
    if not cfg:
        return sorted(_KINDS)
    return [k for k in sorted(_KINDS) if k in cfg]
