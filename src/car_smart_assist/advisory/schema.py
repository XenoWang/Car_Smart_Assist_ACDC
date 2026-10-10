"""Stage 1 传给 Stage 2 的结构化数据。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class RiskLevel(str, Enum):
    """风险等级，决定提示强度。"""

    UNKNOWN = "unknown"    # 识别信息不足，无法评估场景风险
    NONE = "none"          # 当前识别证据没触发风险规则，不代表道路安全
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
    """一个被检出的交通目标或道路杂物候选。

    方向和距离是两个独立的属性：前车近该减速，来车近该注意会车，
    两者的驾驶建议完全相反，所以方向未知时必须显式标注，
    不能默认成「前车」。
    """

    category: str
    direction: TargetDirection = TargetDirection.UNKNOWN
    # 到自车的距离（米）。None 表示未测出 —— 不要用 0 表示未知。
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
    """Stage 1 的输出。门控结果也带上，因为下游需要知道这一帧的可信度。"""

    # --- 门控（第一级判断：这一帧到底能不能用）---
    # 取值来自 perception.visibility.gate.VisibilityLevel
    visibility_level: str = "unknown"
    # 能见度门控提供的置信度乘子：默认 BLIND=0，DEGRADED=0.6，VISIBLE=1.0。
    # 下游所有置信度都要乘上它，不要各自重复判断能见度好不好。
    visibility_confidence_multiplier: float = 1.0
    # 触发门控判定的原因，便于追溯
    visibility_reasons: list[str] = field(default_factory=list)

    # --- 路况 ---
    road_condition: str | None = None
    road_condition_confidence: float = 0.0
    # 独立天气模型的多标签输出。用于提醒；只有和其他信号组合时才参与接管判断。
    weather_attributes: tuple[str, ...] = ()
    weather_probabilities: dict[str, float] = field(default_factory=dict)

    # --- 接管边界（0=可继续 1=接近边界 2=应立即接管）---
    handover_level: int | None = None
    handover_confidence: float = 0.0

    # --- 目标 ---
    objects: list[TargetObject] = field(default_factory=list)
    # 只有检测器实际成功运行并返回本帧完整结果时才设为 True。
    # True + [] 表示未检出目标；False 表示检测结果不可用，两者不能混同。
    object_detection_available: bool = False
    # 检测器支持的类别；成功返回空框也不能排除这些类别之外的道路障碍。
    object_detection_classes: tuple[str, ...] = ()
    # 本帧是否成功运行支持道路障碍候选的检测器；不代表覆盖所有未知物体。
    road_obstacle_detection_available: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "visibility_level": self.visibility_level,
            "visibility_confidence_multiplier": self.visibility_confidence_multiplier,
            "visibility_reasons": list(self.visibility_reasons),
            "road_condition": self.road_condition,
            "road_condition_confidence": self.road_condition_confidence,
            "weather_attributes": list(self.weather_attributes),
            "weather_probabilities": dict(self.weather_probabilities),
            "handover_level": self.handover_level,
            "handover_confidence": self.handover_confidence,
            "objects": [o.to_dict() for o in self.objects],
            "object_detection_available": self.object_detection_available,
            "object_detection_classes": list(self.object_detection_classes),
            "road_obstacle_detection_available": self.road_obstacle_detection_available,
        }

    def effective_confidence(self, raw: float) -> float:
        """把门控降级乘到某个原始置信度上。

        统一在这里做，不让每个下游模块自己乘 ——
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
    """Stage 2 的输出：给司机的最终提示和可追溯的证据。"""

    should_takeover: bool = False
    risk_level: RiskLevel = RiskLevel.NONE
    # 给司机的文本。任何情况下都不能为空 ——
    # 失败时降级到模板文案，绝不允许「本帧无输出」。
    text: str = ""
    # 触发这条建议的证据链。每条都应该能被独立验证。
    evidence: list[str] = field(default_factory=list)
    # 这条结果由哪个环节产生：visibility_gate | policy | template | llm
    # 用它区分「规则判定的」和「模型生成的」，出问题时能快速定位。
    source: str = "unknown"
    # 感知策略的分级依据、未知项和接管决定；门控/兜底路径可为 None。
    policy_details: dict[str, Any] | None = None

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
