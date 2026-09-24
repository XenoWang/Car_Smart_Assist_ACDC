"""能见度门控的自编码器与重建误差计算。

职责:
    - MultiScaleBlock：并行 3×3 / 5×5 / 7×7 卷积提取多尺度特征，通道融合
    - ConvAutoencoder：多尺度编码器 + 镜像解码器，学「清晰可见的驾驶场景」的流形
    - ReconstructionError：多种重建误差度量（全局 / 分块 / 分位数）
    - 只做模型与前向计算，不含训练循环（trainer.py）与判定逻辑（gate.py）

为什么编码器用多尺度并行卷积核（本模块的核心设计）:
    不同类型的退化破坏的是**不同尺度**的结构信息：

        雾      -> 高频对比度被整体压缩，细纹理最先消失        -> 3×3 主导
        模糊    -> 边缘锐度丢失，中尺度轮廓变糊               -> 3×3 / 5×5
        黑暗    -> 全局亮度塌陷 + 传感器噪声                    -> 7×7（大结构）与 3×3（噪声）
        遮挡    -> 大片连续区域被覆盖，破坏场景整体布局         -> 7×7

    单尺度（比如全用 3×3）要靠堆叠深度去间接"发现"这些不同尺度的线索，
    而且浅层的 3×3 感受野太小，看不到全局亮度这类信息；
    等到深层感受野够了，细纹理的信号又已经在多次下采样中丢掉。
    并行多尺度把三个尺度在**同一层**同时摆出来，再由 1×1 卷积做通道融合
    学习「哪个尺度在这个场景下更重要」—— 对能见度判别来说，
    这个尺度选择能力本身就是有用的信号。

    这也是 Inception 系列的核心思想，只是这里用在一个无监督的重建任务上。

设计要点:
    1. **输入分辨率压到 256×144 而非原图。**
       能见度是全局属性，不需要 1920×1080 的细节；
       降采样后 12GB 显存能开大 batch，训练几分钟收敛。
       保持 16:9 避免拉伸 —— 拉伸会改变图像的空间频率分布，
       而高频能量正是判「糊没糊」的关键量之一。

    2. **各分支先用 1×1 降维再卷积（bottleneck）。**
       5×5 和 7×7 的参数量是 3×3 的 2.8 倍与 5.4 倍；
       若每条分支都保持完整通道数，一个 block 就要 1.4M 参数。
       先把通道压到 out/3 再各自卷积，参数量降到约 1/8，
       而多尺度的表达能力基本保留。可用 use_bottleneck 关闭做消融。

    3. **瓶颈压到 256 维。**
       瓶颈太宽会让 AE 退化成恒等映射（重建误差恒为 0，完全失去判别力）。
       256 维对应输出 3×144×256 ≈ 110k 维，压缩比约 430:1，
       足以迫使 AE 只保留场景的结构性信息。

    4. **误差按图归一化后再聚合。**
       直接用 sum 会让大图/小图不可比；用 mean 则会被大面积平坦区稀释。
       这里用逐像素 MSE 的均值作为主分数，并额外提供分块分数的分位数，
       后者对「大部分糊住 + 小部分清晰」这类局部退化更敏感。

    5. **不做任何 BN。**
       无监督异常检测里，BN 会把 batch 内的统计量泄漏到每条样本上 ——
       同一条样本和不同样本凑一批会得到不同的分数，评测结果不可复现。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from car_smart_assist.config.visibility import MODEL_DEFAULTS, SCORING_DEFAULTS

DEFAULT_KERNEL_SIZES: tuple[int, ...] = MODEL_DEFAULTS["kernel_sizes"]


class MultiScaleBlock(nn.Module):
    """多尺度并行卷积块：3×3 / 5×5 / 7×7 分别提取，通道维融合。

    结构:
        输入 (B, C_in, H, W)
          │
          ├─ 可选 1×1 降维 -> C_b
          │
          ├── Conv k=3, stride=s, pad=1 ─┐
          ├── Conv k=5, stride=s, pad=2 ─┼─ 各自输出 (B, C_b, H/s, W/s)
          └── Conv k=7, stride=s, pad=3 ─┘
                       │
                   通道拼接 -> (B, len(kernels)*C_b, H/s, W/s)
                       │
                   1×1 卷积融合 -> (B, C_out, H/s, W/s)
                       │
                     GELU

    padding 取 k//2 保证各分支输出空间尺寸一致，否则无法在通道维拼接。
    步长只在分支卷积上施加，三个分支同步下采样，拼接时尺寸天然对齐。

    Args:
        in_channels: 输入通道
        out_channels: 输出通道
        kernel_sizes: 并行核尺寸，默认 (3, 5, 7)
        dilations: 各分支的空洞率，默认全 1（普通卷积）。
            设为 (1, 2, 3) 时有效感受野为 3 / 9 / 15，参数量仍按实际核尺寸算 ——
            这是用空洞卷积替代大核的做法，能在大感受野下显著省参数。
            置 None 等价于全 1。
        stride: 下采样步长
        use_bottleneck: 是否先用 1×1 降维。关闭则各分支保持完整通道，
            参数量约 8 倍，仅在消融实验里用。

    Note:
        padding 取 d*(k-1)//2。这个取值让**任意 (k, d) 组合的输出尺寸都相同**：
            out = floor((in + d(k-1) - d(k-1) - 1)/s) + 1 = ceil(in/s)
        与 k、d 无关，因此三条分支的尺寸天然对齐，可以直接在通道维拼接。
        若改用常见的 k//2（不考虑空洞），dilation>1 时各分支尺寸会各不相同，
        拼接时直接报错。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: Sequence[int] = DEFAULT_KERNEL_SIZES,
        dilations: Sequence[int] | None = None,
        stride: int = 2,
        use_bottleneck: bool = MODEL_DEFAULTS["use_bottleneck"],
    ) -> None:
        super().__init__()
        self.kernel_sizes = tuple(kernel_sizes)
        self.dilations = tuple(dilations) if dilations is not None else (1,) * len(self.kernel_sizes)
        if len(self.dilations) != len(self.kernel_sizes):
            raise ValueError(
                f"dilations 长度 {len(self.dilations)} 与 kernel_sizes 长度 "
                f"{len(self.kernel_sizes)} 不一致：kernel_sizes={self.kernel_sizes}, "
                f"dilations={self.dilations}。"
                "两者一一对应，改其中一个必须同步改另一个 —— "
                "例如只把 kernel_sizes 从 [3,5,7] 改成 [3,5] 时，"
                "dilations 也要从 [1,1,1] 改成 [1,1]。"
                "（不自动补齐长度是刻意的：静默补齐会掩盖配置错误）"
            )
        if any(d < 1 for d in self.dilations):
            raise ValueError(f"空洞率必须 >= 1，收到 {self.dilations}")

        n_branch = len(self.kernel_sizes)
        # 各分支通道数：有 bottleneck 时压到 out/n_branch，拼接后正好是 out
        branch_ch = max(1, out_channels // n_branch) if use_bottleneck else out_channels

        self.reduce: nn.Module
        if use_bottleneck:
            self.reduce = nn.Sequential(
                nn.Conv2d(in_channels, branch_ch, 1, bias=False),
                nn.GELU(),
            )
            conv_in = branch_ch
        else:
            self.reduce = nn.Identity()
            conv_in = in_channels

        self.branches = nn.ModuleList(
            nn.Conv2d(
                conv_in,
                branch_ch,
                k,
                stride=stride,
                padding=d * (k - 1) // 2,
                dilation=d,
                bias=False,
            )
            for k, d in zip(self.kernel_sizes, self.dilations, strict=True)
        )

        fused_in = branch_ch * n_branch
        # 1×1 融合：学「这个场景下哪个尺度更该被信任」
        self.fuse = nn.Sequential(
            nn.Conv2d(fused_in, out_channels, 1, bias=False),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.reduce(x)
        feats = [conv(h) for conv in self.branches]
        return self.fuse(torch.cat(feats, dim=1))


class ConvAutoencoder(nn.Module):
    """多尺度卷积自编码器。

    结构: 4 个 MultiScaleBlock 编码 -> 全连接瓶颈 -> 镜像反卷积解码。
    没有跳连（不是 U-Net）—— 跳连会把输入的高频细节直接抄到输出，
    使重建误差对模糊/低对比度不敏感，恰好抹掉我们要检测的信号。

    Args:
        in_channels: 输入通道数，RGB 为 3
        base_channels: 第一层通道数，后续层逐级翻倍
        latent_dim: 瓶颈维度
        input_size: (H, W)，用于校验形状并在解码端反推空间尺寸
        encoder_type: multiscale | plain。plain 用单尺度 4×4 卷积，
            保留它是为了做「多尺度 vs 单尺度」的消融对比，不是默认路径。
        kernel_sizes: 多尺度分支的核尺寸
        dilations: 各分支空洞率，默认 None（等价全 1）。
            设 (1,2,3) 配合 kernel_sizes (3,3,3) 可得到 3/9/15 的有效感受野，
            参数量只有大核方案的三分之一。
        use_bottleneck: 多尺度块内是否用 1×1 降维
        pre_latent_channels: 展平进瓶颈**之前**先用 1×1 卷积压到的通道数。0 表示不压。
            这一项直接决定参数量与过拟合程度：
              不压（128 通道）: feat_dim = 128×9×16 = 18432
                                to_latent + from_latent = 2×18432×256 ≈ 9.4M 参数，
                                占整个模型约 90% —— 而这部分就是过拟合的来源。
                                实测（3605 张训练图）：留出集误差比训练集高 **45.6%**。
              压到 32 通道   : feat_dim = 32×9×16 = 4608
                                两个全连接降到 2.36M，模型从 10.5M 降到约 3.4M。
            在 3605 张图上训 10M 参数本来就不合理，这不是「防患于未然」，
            是实测出来的过拟合，见 artifacts/reports/visibility/ 的对比报告。
    """

    def __init__(
        self,
        in_channels: int = MODEL_DEFAULTS["in_channels"],
        base_channels: int = MODEL_DEFAULTS["base_channels"],
        latent_dim: int = MODEL_DEFAULTS["latent_dim"],
        input_size: tuple[int, int] = MODEL_DEFAULTS["input_size"],
        encoder_type: str = MODEL_DEFAULTS["encoder_type"],
        kernel_sizes: Sequence[int] = DEFAULT_KERNEL_SIZES,
        dilations: Sequence[int] | None = None,
        use_bottleneck: bool = MODEL_DEFAULTS["use_bottleneck"],
        pre_latent_channels: int = MODEL_DEFAULTS["pre_latent_channels"],
    ) -> None:
        super().__init__()
        self.input_size = tuple(input_size)
        self.encoder_type = encoder_type
        self.dilations = tuple(dilations) if dilations is not None else None
        c = base_channels
        chans = [c, c * 2, c * 4, c * 4]

        if encoder_type == "multiscale":
            self.encoder = nn.Sequential(
                MultiScaleBlock(in_channels, chans[0], kernel_sizes, dilations, 2, use_bottleneck),
                MultiScaleBlock(chans[0], chans[1], kernel_sizes, dilations, 2, use_bottleneck),
                MultiScaleBlock(chans[1], chans[2], kernel_sizes, dilations, 2, use_bottleneck),
                MultiScaleBlock(chans[2], chans[3], kernel_sizes, dilations, 2, use_bottleneck),
            )
        elif encoder_type == "plain":
            # 单尺度对照：全 4×4 卷积，结构与本模块早期版本一致，
            # 仅用于消融，验证多尺度确实带来增益
            self.encoder = nn.Sequential(
                nn.Conv2d(in_channels, chans[0], 4, stride=2, padding=1), nn.GELU(),
                nn.Conv2d(chans[0], chans[1], 4, stride=2, padding=1), nn.GELU(),
                nn.Conv2d(chans[1], chans[2], 4, stride=2, padding=1), nn.GELU(),
                nn.Conv2d(chans[2], chans[3], 4, stride=2, padding=1), nn.GELU(),
            )
        else:
            raise ValueError(
                f"未知 encoder_type {encoder_type!r}，可选: multiscale | plain"
            )

        h, w = self.input_size
        # 4 次减半；输入尺寸不是 16 的整数倍时用上取整，解码端再对齐回来
        self.feat_hw = ((h + 15) // 16, (w + 15) // 16)

        # 把编码器输出强行对齐到 feat_hw。
        #
        # 为什么需要这一步：4 次 stride-2 卷积的实际输出尺寸取决于
        # 每层 padding 与输入奇偶性（奇数尺寸会向上取整），
        # 当传入尺寸与配置的 input_size 不同时，flatten 出来的维度
        # 会和 feat_dim 对不上，to_latent 的矩阵乘直接崩。
        # 早期版本只依赖「最后插值回去」，但崩的位置在插值之前，兜不住。
        #
        # 用自适应平均池化把空间尺寸钉死在 feat_hw：
        #   · 输入尺寸 == 配置尺寸时是恒等操作，不改变任何既有行为、无额外开销
        #   · 尺寸不同时退化为平滑下采样，模型仍能工作而不是崩掉
        self.pool_to_grid = nn.AdaptiveAvgPool2d(self.feat_hw)

        # 展平前的 1×1 投影：把通道压下来，直接决定两个全连接层的规模。
        # 不做这一步时它们占整个模型约 90% 的参数，是过拟合的主因。
        if pre_latent_channels and pre_latent_channels < chans[3]:
            self.pre_latent: nn.Module = nn.Sequential(
                nn.Conv2d(chans[3], pre_latent_channels, 1, bias=False),
                nn.GELU(),
            )
            latent_spatial_ch = pre_latent_channels
        else:
            self.pre_latent = nn.Identity()
            latent_spatial_ch = chans[3]

        self.feat_dim = latent_spatial_ch * self.feat_hw[0] * self.feat_hw[1]

        self.to_latent = nn.Linear(self.feat_dim, latent_dim)
        self.from_latent = nn.Linear(latent_dim, self.feat_dim)
        self.latent_spatial_ch = latent_spatial_ch
        # 解码前把通道数还原回编码器的输出宽度，与 pre_latent 对称
        if latent_spatial_ch != chans[3]:
            self.post_latent: nn.Module = nn.Sequential(
                nn.Conv2d(latent_spatial_ch, chans[3], 1, bias=False),
                nn.GELU(),
            )
        else:
            self.post_latent = nn.Identity()

        # --- 解码器：镜像回去。保持单尺度反卷积。
        # 解码只是重建，不承担判别职责；判别力来自编码器的多尺度表示。
        # 在这里也上多尺度会显著增加计算，对重建质量的增益却很有限。
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(chans[3], chans[3], 4, stride=2, padding=1), nn.GELU(),
            nn.ConvTranspose2d(chans[3], chans[2], 4, stride=2, padding=1), nn.GELU(),
            nn.ConvTranspose2d(chans[2], chans[1], 4, stride=2, padding=1), nn.GELU(),
            nn.ConvTranspose2d(chans[1], in_channels, 4, stride=2, padding=1),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """图像 -> 瓶颈向量。"""
        f = self.pool_to_grid(self.encoder(x))
        f = self.pre_latent(f)
        return self.to_latent(f.flatten(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """自重建。输出与输入同尺寸（用 interpolate 兜底非 16 倍数的输入）。"""
        f = self.pool_to_grid(self.encoder(x))
        f = self.pre_latent(f)
        z = self.to_latent(f.flatten(1))
        y = self.from_latent(z).view(x.shape[0], self.latent_spatial_ch, *self.feat_hw)
        y = self.post_latent(y)
        y = self.decoder(y)
        if y.shape[-2:] != x.shape[-2:]:
            # 输入尺寸不是 16 的整数倍时会有 1~15 像素的偏差，插值对齐
            y = F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return y


@dataclass
class ReconstructionError:
    """一张图的重建误差统计。

    同时给出多个度量，因为它们在失效场景上互补：
        mean      —— 主分数，整体偏差
        p90_block —— 分块误差的 90 分位，对局部退化敏感
        max_block —— 最坏的一块，用于「大部分糊住但有小片清晰」的情形
        std_block —— 块间方差，反映退化是否均匀（雾是均匀的，遮挡不是）
    """

    mean: float
    p90_block: float
    max_block: float
    std_block: float

    def to_dict(self) -> dict[str, float]:
        return {
            "mean": self.mean,
            "p90_block": self.p90_block,
            "max_block": self.max_block,
            "std_block": self.std_block,
        }


@torch.no_grad()
def reconstruction_mean(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """返回 batch 每张图的逐像素重建 MSE，不计算分块分位数等附加统计。"""
    model.eval()
    recon = model(x)
    return F.mse_loss(recon, x, reduction="none").mean(dim=(1, 2, 3))


@torch.no_grad()
def reconstruction_error(
    model: ConvAutoencoder,
    x: torch.Tensor,
    block_grid: tuple[int, int] = SCORING_DEFAULTS["reconstruction_error"]["block_grid"],
) -> list[ReconstructionError]:
    """计算一批图像的重建误差。

    Args:
        model: 已训练的 AE（应为 eval 模式）
        x: (B, 3, H, W) 归一化后的输入
        block_grid: 把图切成几行几列来算分块误差。(3,3) 在 256×144 下每块约 85×48

    Returns:
        与 batch 等长的 ReconstructionError 列表。
        逐像素误差取通道维平均，得到单通道误差图，再做分块统计。
    """
    model.eval()
    recon = model(x)
    # (B, 1, H, W)：先对通道取平均，保证误差量纲与颜色无关
    err = F.mse_loss(recon, x, reduction="none").mean(dim=1, keepdim=True)
    if err.shape[0] == 0:
        return []

    gh, gw = block_grid
    # 整个 batch 一次池化和统计。逐张转 float 会在 GPU 上触发多次同步；
    # 向量化后只把四列统计量一次性拷回 CPU。
    blocks = F.adaptive_avg_pool2d(err, (gh, gw)).flatten(1)
    means = err.mean(dim=(1, 2, 3))
    p90 = torch.quantile(blocks, 0.9, dim=1)
    maxima = blocks.max(dim=1).values
    stds = blocks.std(dim=1) if blocks.shape[1] > 1 else torch.zeros_like(means)
    rows = torch.stack((means, p90, maxima, stds), dim=1).cpu().tolist()
    return [ReconstructionError(*(float(value) for value in row)) for row in rows]


def effective_kernel_size(kernel_sizes: Sequence[int], dilations: Sequence[int]) -> tuple[int, ...]:
    """各分支的有效感受野尺寸：k_eff = d*(k-1) + 1。

    用于在报告与日志里说明「这一组配置实际看多大范围」，
    避免只看 kernel_sizes 而误判感受野。
    """
    return tuple(
        d * (k - 1) + 1 for k, d in zip(kernel_sizes, dilations, strict=True)
    )


def build_autoencoder(cfg: dict[str, Any]) -> ConvAutoencoder:
    """按配置构建 AE。配置项见 configs/model/visibility.yaml。"""
    cfg = {**MODEL_DEFAULTS, **cfg}
    return ConvAutoencoder(
        in_channels=int(cfg["in_channels"]),
        base_channels=int(cfg["base_channels"]),
        latent_dim=int(cfg["latent_dim"]),
        input_size=tuple(cfg["input_size"]),
        encoder_type=str(cfg["encoder_type"]),
        kernel_sizes=tuple(cfg["kernel_sizes"]),
        dilations=tuple(cfg["dilations"]) if cfg.get("dilations") is not None else None,
        use_bottleneck=bool(cfg["use_bottleneck"]),
        pre_latent_channels=int(cfg["pre_latent_channels"]),
    )
