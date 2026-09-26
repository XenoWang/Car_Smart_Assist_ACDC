"""根据本帧结构化识别结果评估场景风险，不从静态图片臆测速度或碰撞时间。"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from numbers import Real
from typing import Any

from car_smart_assist.advisory.schema import (
    PerceptionResult,
    RiskLevel,
    TargetDirection,
    TargetObject,
)
from car_smart_assist.config.policy import PolicyConfig


@dataclass
class RiskAssessment:
    """risk_level 是已观察到的风险；reliable 表示是否足以形成完整判断。"""

    risk_level: RiskLevel = RiskLevel.UNKNOWN
    reasons: list[str] = field(default_factory=list)
    diagnostics: list[str] = field(default_factory=list)
    unavailable_reasons: list[str] = field(default_factory=list)
    triggered: list[str] = field(default_factory=list)
    primary_target_index: int | None = None

    @property
    def reliable(self) -> bool:
        return not self.unavailable_reasons and self.risk_level is not RiskLevel.UNKNOWN

    def to_dict(self) -> dict[str, Any]:
        return {
            "risk_level": self.risk_level.value,
            "reliable": self.reliable,
            "reasons": list(self.reasons),
            "diagnostics": list(self.diagnostics),
            "unavailable_reasons": list(self.unavailable_reasons),
            "triggered": list(self.triggered),
            "primary_target_index": self.primary_target_index,
        }


def finite_number(value: Any) -> bool:
    """排除 bool、NaN、无穷大和无法转换的超大数。"""
    if isinstance(value, bool) or not isinstance(value, Real):
        return False
    try:
        return isfinite(float(value))
    except (OverflowError, ValueError):
        return False


def valid_confidence(value: Any) -> bool:
    return finite_number(value) and 0 <= value <= 1


def assess_risk(perception: PerceptionResult | None, config: PolicyConfig) -> RiskAssessment:
    """纯规则评估；全部阈值从 config 获取，输入对象保持不变。"""
    result = RiskAssessment()

    def unknown(code: str, reason: str) -> None:
        result.triggered.append(code)
        result.unavailable_reasons.append(reason)

    if not isinstance(perception, PerceptionResult):
        unknown("perception_missing", "未获得本帧的结构化感知结果")
        return result
    visibility = perception.visibility_level
    if visibility not in ("visible", "degraded"):
        reason = (
            "图像能见度不足，无法可靠识别场景" if visibility == "blind" else "图像能见度状态未知"
        )
        unknown("visibility_unavailable", reason)
        return result
    multiplier = perception.visibility_confidence_multiplier
    if not valid_confidence(multiplier) or multiplier == 0:
        unknown("visibility_confidence_invalid", "能见度置信度乘子无效，无法采信识别结果")
        return result

    ranking = {RiskLevel.NONE: 0, RiskLevel.NOTICE: 1, RiskLevel.WARNING: 2, RiskLevel.CRITICAL: 3}
    result.risk_level = RiskLevel.NONE
    primary_distance = float("inf")

    def evidence(level: RiskLevel, code: str, reason: str) -> None:
        result.triggered.append(code)
        result.reasons.append(reason)
        if ranking[level] > ranking[result.risk_level]:
            result.risk_level = level

    if visibility == "degraded":
        evidence(RiskLevel.NOTICE, "visibility_degraded", "图片能见度下降")

    weather = perception.road_condition
    weather_multiplier = 1.0
    if not isinstance(weather, str) or weather not in config.weather_risk_levels:
        result.diagnostics.append("天气类别未能确认；不单独触发接管，也不应用恶劣天气倍率")
    elif not valid_confidence(perception.road_condition_confidence) or (
        perception.effective_confidence(perception.road_condition_confidence)
        < config.min_confidence_for_clear
    ):
        result.diagnostics.append("天气分类置信度不足；不单独触发接管，也不应用恶劣天气倍率")
    else:
        weather_multiplier = config.weather_risk_multiplier[weather]
        evidence(
            config.weather_risk_levels[weather], "weather_recognized", f"图片识别路况：{weather}"
        )

    if perception.object_detection_available is not True:
        unknown("detection_unavailable", "未确认本帧目标检测已成功完成，空列表不能视作无目标")
    objects = perception.objects
    if not isinstance(objects, (list, tuple)):
        unknown("objects_invalid", "目标检测结果格式无效")
        objects = []
    elif not objects and perception.object_detection_available is True:
        result.reasons.append("本帧检测已完成，未检出交通目标")

    for index, target in enumerate(objects):
        prefix = f"目标[{index}]"
        if not isinstance(target, TargetObject):
            unknown("target_invalid", f"{prefix}结构无效")
            continue
        if not isinstance(target.category, str) or not target.category.strip():
            unknown("category_unknown", f"{prefix}类别未知")
        if not valid_confidence(target.confidence) or (
            perception.effective_confidence(target.confidence) < config.min_confidence_for_clear
        ):
            unknown("target_confidence_low", f"{prefix}识别有效置信度不足")
            continue

        try:
            direction = TargetDirection(target.direction)
        except (ValueError, TypeError):
            direction = TargetDirection.UNKNOWN
        if direction is TargetDirection.UNKNOWN:
            unknown("direction_unknown", f"{prefix}方向未确认，不能当作前车或来车")
        # 未确认方向时仅依据已知距离分级，并由 unknown 原因触发接管；不假定它是来车。
        direction_factor = (
            config.oncoming_stricter_factor if direction is TargetDirection.ONCOMING else 1.0
        )
        distance = target.distance_m
        if not finite_number(distance) or distance < 0:
            unknown("distance_unknown", f"{prefix}距离缺失或无效")
            continue
        uncertainty = target.distance_uncertainty_m
        margin = 0.0
        if uncertainty is None:
            if config.require_distance_uncertainty:
                unknown("distance_uncertainty_missing", f"{prefix}缺少距离不确定度")
                continue
        elif not finite_number(uncertainty) or uncertainty < 0:
            unknown("distance_uncertainty_invalid", f"{prefix}距离不确定度无效")
            continue
        else:
            if uncertainty > config.max_distance_uncertainty_ratio * distance:
                unknown("distance_uncertainty_high", f"{prefix}距离估计不确定度过大")
                continue
            margin = config.uncertainty_sigma * uncertainty
            if not finite_number(margin):
                unknown("distance_margin_invalid", f"{prefix}距离不确定度计算溢出")
                continue

        conservative_distance = max(0.0, distance - margin)
        adjusted_distance = conservative_distance / weather_multiplier / direction_factor
        level = RiskLevel.NONE
        for name in ("critical", "warning", "notice"):
            if adjusted_distance < config.distance_thresholds[name]:
                level = RiskLevel(name)
                break
        if level is not RiskLevel.NONE and (
            ranking[level] > ranking[result.risk_level]
            or (
                ranking[level] == ranking[result.risk_level]
                and adjusted_distance < primary_distance
            )
        ):
            result.primary_target_index = index
            primary_distance = adjusted_distance
        direction_text = {
            TargetDirection.LEADING: "前方同向目标",
            TargetDirection.ONCOMING: "对向目标",
            TargetDirection.UNKNOWN: "方向未确认目标",
        }[direction]
        evidence(
            level,
            f"target_distance_{level.value}",
            f"{prefix}{direction_text}（{target.category}）距离约 {distance:.1f} 米，"
            f"保守距离 {conservative_distance:.1f} 米，天气倍率 {weather_multiplier:g}、"
            f"方向倍率 {direction_factor:g}，距离规则等级 {level.value}",
        )

    if result.unavailable_reasons and result.risk_level is RiskLevel.NONE:
        result.risk_level = RiskLevel.UNKNOWN
    return result
