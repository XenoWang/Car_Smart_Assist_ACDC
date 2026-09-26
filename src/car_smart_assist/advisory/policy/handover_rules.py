"""根据判断可靠性与已识别风险生成接管请求，不执行车辆控制权切换。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from car_smart_assist.advisory.policy.risk import RiskAssessment, assess_risk, valid_confidence
from car_smart_assist.advisory.schema import PerceptionResult, RiskLevel
from car_smart_assist.config.policy import PolicyConfig


@dataclass
class HandoverDecision:
    should_takeover: bool
    unable_to_judge: bool
    action: str
    reasons: list[str]
    risk: RiskAssessment

    def to_dict(self) -> dict[str, Any]:
        return {
            "should_takeover": self.should_takeover,
            "unable_to_judge": self.unable_to_judge,
            "action": self.action,
            "reasons": list(self.reasons),
            "risk": self.risk.to_dict(),
        }


def evaluate_handover(
    perception: PerceptionResult | None,
    config: PolicyConfig,
) -> HandoverDecision:
    """无法可靠判断就请求接管；明确高风险和可靠的接管头 2 级同样请求接管。"""
    risk = assess_risk(perception, config)
    reasons = [*risk.unavailable_reasons, *risk.diagnostics]
    unable_to_judge = not risk.reliable
    head_action = "no_action"
    if isinstance(perception, PerceptionResult):
        level = perception.handover_level
        if level is None:
            if config.require_handover_head:
                unable_to_judge = True
                reasons.append("接管边界分类头未提供本帧结果")
        elif type(level) is not int or level not in config.handover_level_to_action:
            unable_to_judge = True
            reasons.append("接管边界等级无效")
        elif (
            not valid_confidence(perception.handover_confidence)
            or not valid_confidence(perception.visibility_confidence_multiplier)
            or perception.effective_confidence(perception.handover_confidence)
            < config.min_confidence_for_clear
        ):
            unable_to_judge = True
            reasons.append("接管边界识别的有效置信度不足")
        else:
            head_action = config.handover_level_to_action[level]
            if head_action != "no_action":
                reasons.append(f"接管边界分类头识别为 {level} 级，动作 {head_action}")

    if risk.risk_level is RiskLevel.CRITICAL:
        reasons.append("图片识别结果触发 critical 风险规则")
    should_takeover = (
        unable_to_judge
        or risk.risk_level is RiskLevel.CRITICAL
        or head_action == "request_takeover"
    )
    action = (
        "request_takeover"
        if should_takeover
        else (
            "light_notice"
            if risk.risk_level in (RiskLevel.NOTICE, RiskLevel.WARNING)
            or head_action == "light_notice"
            else "no_action"
        )
    )
    if not reasons:
        reasons.append("当前识别信息满足规则判断条件，未触发接管规则")
    return HandoverDecision(should_takeover, unable_to_judge, action, reasons, risk)
