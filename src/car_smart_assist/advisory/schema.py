"""Stage 1 -> Stage 2 之间传递的结构化数据契约。

职责:
    - 定义 PerceptionResult：路况、接管边界与置信度、各目标的方向/距离/不确定度
    - 定义 AdvisoryResult：是否接管、风险等级、给司机的文本、以及触发它的证据
    - 两个 Stage 只通过这个 contract 耦合；改字段要同步改两侧和评测脚本

为什么这一层必须存在（而不是让 Stage 2 直接读 Stage 1 的内部结构）:
    1. **可解释性的落点**。每条给司机的建议都能追溯到具体证据 ——
       哪一档接管边界、哪个目标、多少米、由哪条规则触发。
       出问题时能回答「为什么这么提示」，而不是面对一段无法归因的文本。
    2. **可替换性**。Stage 1 换骨干网、Stage 2 从模板换成 VLM，
       只要契约不变，双方互不影响。
    3. **可评测性**。有了结构化字段才能算「接管边界等级 2 的召回率」这类
       真实的安全指标；只有自然语言文本时只能算 BLEU，而 BLEU 在这个任务上
       与可用性几乎无关。

设计约束:
    - 所有字段**必须可序列化**（json）—— 评测、日志、报告都依赖这一点
    - 距离一律用米、一律用「到自车」口径（见 docs/label_spec.md 第 5 节）
    - 不确定的字段用 None 而不是 0 或 -1：0 米表示「贴脸」，与「不知道」是两回事，
      混用会让下游把「没测出来」当成「很近」，恰好造成最危险的误判
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class RiskLevel(str, Enum):
    """风险等级。决定提示强度。"""

    NONE = "none"          # 一切正常，不提示
    NOTICE = "notice"      # 轻提示：注意即可
    WARNING = "warning"    # 需要动作：减速 / 提高注意
    CRITICAL = "critical"  # 必须立即接管


class TargetDirection(str, Enum):
    """目标方向。由序列时序推导，见 docs/label_spec.md 4.1。"""

    LEADING = "leading"    # 同向（前车）
    ONCOMING = "oncoming"  # 对向（来车）
    UNKNOWN = "unknown"    # 推导置信度不足 —— 宁可不说，也不要给错


@dataclass
class TargetObject:
    """一个被检出的交通参与者。

    方向与距离是**独立**的两个属性：前车近该减速、来车近该注意会车，
    两者的驾驶建议完全相反，因此方向未知时必须显式标注，
    不能默认成「前车」。
    """

    category: str
    direction: TargetDirection = TargetDirection.UNKNOWN
    # 到**自车**的距离（米）。None 表示未测出 —— 不要用 0 表示未知。
    distance_m: float | None = None
    # 距离估计的不确定度（米，1σ）。供下游做保守决策：不确定度大就该更保守。
    distance_uncertainty_m: float | None = None
    bbox: tuple[float, float, float, float] | None = None
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["direction"] = self.direction.value
        return d


@dataclass
class PerceptionResult:
    """Stage 1 的输出。门控结果一并带上，因为下游需要知道这一帧的可信度。"""

    # --- 门控（第一级判断：这一帧到底能不能用）---
    # 取值来自 perception.visibility.gate.VisibilityLevel
    visibility_level: str = "unknown"
    # 能见度门控提供的置信度乘子：默认 BLIND=0，DEGRADED=0.6，VISIBLE=1.0。
    # 下游所有置信度都应乘上它，而不是各自重复判断能见度好不好。
    visibility_confidence_multiplier: float = 1.0
    # 触发门控判定的原因，便于追溯
    visibility_reasons: list[str] = field(default_factory=list)

    # --- 路况 ---
    road_condition: str | None = None
    road_condition_confidence: float = 0.0

    # --- 接管边界（0=可继续 1=接近边界 2=应立即接管）---
    handover_level: int | None = None
    handover_confidence: float = 0.0

    # --- 目标 ---
    objects: list[TargetObject] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "visibility_level": self.visibility_level,
            "visibility_confidence_multiplier": self.visibility_confidence_multiplier,
            "visibility_reasons": list(self.visibility_reasons),
            "road_condition": self.road_condition,
            "road_condition_confidence": self.road_condition_confidence,
            "handover_level": self.handover_level,
            "handover_confidence": self.handover_confidence,
            "objects": [o.to_dict() for o in self.objects],
        }

    def effective_confidence(self, raw: float) -> float:
        """把门控降级乘到某个原始置信度上。

        统一在这里做，而不是让每个下游模块自己乘 ——
        否则总有一处会忘记，而忘记的那处恰好是最需要保守的地方。
        """
        return float(raw) * self.visibility_confidence_multiplier

    @property
    def nearest_leading(self) -> TargetObject | None:
        """最近的前车。None 表示没有或距离未知。"""
        return self._nearest_in_direction(TargetDirection.LEADING)

    @property
    def nearest_oncoming(self) -> TargetObject | None:
        """最近的来车。"""
        return self._nearest_in_direction(TargetDirection.ONCOMING)

    def _nearest_in_direction(self, direction: TargetDirection) -> TargetObject | None:
        """单次遍历查找最近目标，不创建候选列表。"""
        return min(
            (o for o in self.objects if o.direction is direction and o.distance_m is not None),
            key=lambda o: o.distance_m,
            default=None,
        )


@dataclass
class AdvisoryResult:
    """Stage 2 的输出：给司机的最终提示 + 可追溯的证据。"""

    should_takeover: bool = False
    risk_level: RiskLevel = RiskLevel.NONE
    # 给司机的文本。**任何情况下都不能为空** ——
    # 失败时降级到模板文案，绝不允许「本帧无输出」。
    text: str = ""
    # 触发这条建议的证据链。每条都应能被独立验证。
    evidence: list[str] = field(default_factory=list)
    # 这条结果是哪个环节产生的：visibility_gate | policy | template | llm
    # 用它区分「规则判定的」与「模型生成的」，出问题时能快速定位。
    source: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["risk_level"] = self.risk_level.value
        return d

    def __post_init__(self) -> None:
        if not self.text:
            raise ValueError(
                "AdvisoryResult.text 不能为空："
                "宁可回退到固定的保守文案，也不允许静默失声"
            )
