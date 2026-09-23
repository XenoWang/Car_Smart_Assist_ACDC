"""能见度门控：无监督判断「这一帧是否已经看不清」。

定位:
    这是感知链路最前面的一道闸门，也是「接管边界」最保守的那一档 ——
    不是「路况复杂建议接管」，而是「我根本看不见，必须立刻接管」。
    判定为不可用时，下游的分割/检测/距离全部不执行，直接给出接管请求。
    理由：在完全看不见的帧上硬跑感知，输出的框和距离都是噪声，
    把它当成有效结果喂给决策层，比直接说「看不见」危险得多。

方法（无监督，不需要任何人工标注）:
    卷积自编码器在 ACDC 的**正常天气参考图**上做自重建训练，
    学到「清晰可见的驾驶场景长什么样」；推理时以重建误差（相对正常天气
    分布的 z 分数）+ 一组确定性信息量特征，共同判定能见度。

    训练集只用正常天气参考图，是这套方法成立的前提：
    若把恶劣天气图也放进训练集，AE 会连浓雾、黑夜一起学会，
    重建误差对所有人一样低，判别力直接归零。

⚠️ 已知失效模式与应对（重要，改这个模块前必读）:
    朴素的图像域重建误差**对「低信息量帧」会失效，且方向是反的**：
    纯黑帧、白茫茫的浓雾帧结构极简、几乎是常数图，AE 反而容易重建，
    误差偏低 —— 会把最该报警的帧漏掉。

    因此本模块不单独依赖重建误差，而是让它与**确定性信息量特征**
    （对比度、熵、边缘密度、高频能量占比）共同参与判定，见 gate.py。
    两路信号是正交的：重建误差答「像不像正常场景」，信息量答「画面里还有没有东西」。
    缺任何一路都会在某一类失效场景上翻车。

    该失效模式是实测发现的，不是预防性设计 ——
    见 artifacts/reports/visibility/ 下的评估报告。

⚠️ 判定哲学（与接管决策的「宁可保守」不同）:
    高信息量的帧**永远不判 BLIND**。信息量够 = 画面里有东西 = 至少还能做判断；
    此时即便异常（隧道、施工区这类训练集里没见过的新奇场景）也只降到 DEGRADED。
    因为把可用帧误判为 BLIND 会让系统在能工作时拒绝工作，用户很快学会忽略它，
    真的看不见时也不再被相信。普通异常检测把「新奇」当「危险」，在驾驶场景里是错的。

主要入口:
    VisibilityTrainer      无监督训练（scripts/train_visibility.py）
    VisibilityScorer       打分：重建误差 z 分数 + 信息量
    VisibilityGate         判定：三档能见度 + 触发原因
    degradation            合成退化，用于定量验证召回率
"""

from car_smart_assist.perception.visibility.autoencoder import (
    DEFAULT_KERNEL_SIZES,
    ConvAutoencoder,
    MultiScaleBlock,
    ReconstructionError,
    build_autoencoder,
    effective_kernel_size,
    reconstruction_error,
)
from car_smart_assist.perception.visibility.degradation import degrade, severity_sweep
from car_smart_assist.perception.visibility.gate import (
    GateThresholds,
    VisibilityGate,
    VisibilityLevel,
    VisibilityVerdict,
)
from car_smart_assist.perception.visibility.scorer import (
    CalibrationStats,
    InformationFeatures,
    VisibilityScore,
    VisibilityScorer,
    compute_information_features,
    information_score,
)
from car_smart_assist.perception.visibility.trainer import (
    TrainHistory,
    VisibilityTrainer,
    resolve_device,
)

__all__ = [
    "DEFAULT_KERNEL_SIZES",
    "CalibrationStats",
    "ConvAutoencoder",
    "GateThresholds",
    "InformationFeatures",
    "MultiScaleBlock",
    "ReconstructionError",
    "TrainHistory",
    "VisibilityGate",
    "VisibilityLevel",
    "VisibilityScore",
    "VisibilityScorer",
    "VisibilityTrainer",
    "VisibilityVerdict",
    "build_autoencoder",
    "compute_information_features",
    "degrade",
    "effective_kernel_size",
    "information_score",
    "reconstruction_error",
    "resolve_device",
    "severity_sweep",
]
