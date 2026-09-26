"""Stage 2 编排：结构化感知结果 -> 给司机的最终提示。

职责:
    - 串起 policy（决策）-> prompt（渲染）-> llm（生成）-> postprocess（清洗）
    - 保证降级路径：LLM 不可用时返回模板文案，绝不返回空
    - 保证安全兜底：LLM 生成内容与规则决策冲突时，以规则的保守结论为准
    - 是 Stage 2 对外的唯一入口

当前实现状态（重要，不要误以为全链路已完成）:
    已实现 —— **门控驱动的接管路径**（from_visibility）。
        这是整个系统里最关键、也最先需要能跑的一条路：
        门控判 BLIND 时直接输出接管请求，不需要感知、不需要 LLM。
        它必须在任何其他模块缺失时都能工作，因为「看不见却不出声」
        是这套系统最不能接受的失败模式。

    已实现 —— 感知驱动的规则路径（from_perception）。
        接收 PerceptionResult，经风险规则与判断可靠性检查后生成固定提示。
        Stage 1 模型仍需单独接入；缺失字段会请求接管，不构造识别结果。

设计原则:
    1. **决策与表达分离**。是否接管由规则/门控决定，语言模型只负责措辞。
    2. **绝不静默失声**。任何失败路径都必须返回保守文案。
    3. **evidence 必须可追溯**。每条提示都要能说清是哪个信号触发的。
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any

from car_smart_assist.advisory.policy.handover_rules import evaluate_handover
from car_smart_assist.advisory.prompt.templates import (
    FALLBACK_TEXT,
    render_policy,
    render_visibility,
)
from car_smart_assist.advisory.schema import (
    AdvisoryResult,
    PerceptionResult,
    RiskLevel,
)
from car_smart_assist.config.policy import PolicyConfig, load_policy_config
from car_smart_assist.perception.visibility.gate import (
    VisibilityLevel,
    VisibilityVerdict,
)

logger = logging.getLogger(__name__)


class AdvisoryGenerator:
    """把结构化输入转成给司机的提示。

    Args:
        cfg: configs/model/advisory_llm.yaml 的 ``advisory`` 段。
            policy 段覆盖集中配置；LLM 后端尚未接入。
        llm_backend: 语言模型后端（尚未实现）。为 None 时全部走模板 ——
            模板路径不是「临时方案」，它是 LLM 失效时的固定兜底，要长期保留。
    """

    def __init__(self, cfg: dict[str, Any] | None = None, llm_backend: Any | None = None) -> None:
        self.cfg = cfg or {}
        self.llm_backend = llm_backend
        self._policy_config: PolicyConfig | None = None

    # --- 门控驱动的路径（已实现，安全关键）---

    def from_visibility(self, verdict: VisibilityVerdict) -> AdvisoryResult:
        """门控结果 -> 司机提示。不需要感知、不需要 LLM。

        这是 pipeline 在 BLIND 时唯一会走的路径，因此它不依赖任何
        尚未实现的模块 —— 少一个依赖就少一个失效点。
        """
        text, risk = render_visibility(verdict.level)

        evidence = [
            f"能见度判定: {verdict.level.value}",
            f"信息量分数: {verdict.information:.3f}",
        ]
        if verdict.recon_z == verdict.recon_z:  # 非 NaN
            evidence.append(f"重建 z 分数: {verdict.recon_z:.2f}")
        if verdict.triggered:
            evidence.append("触发规则: " + ", ".join(verdict.triggered))

        return AdvisoryResult(
            should_takeover=verdict.level is VisibilityLevel.BLIND,
            risk_level=risk,
            text=text,
            evidence=evidence,
            source="visibility_gate",
        )

    def from_unavailable(self, reason: str) -> AdvisoryResult:
        """系统无法形成可靠判断时的失效安全输出。"""
        return AdvisoryResult(
            should_takeover=True,
            risk_level=RiskLevel.UNKNOWN,
            text=FALLBACK_TEXT,
            evidence=["无法形成可靠判断", reason],
            source="fallback",
        )

    # --- 感知驱动的规则路径 ---

    def from_perception(self, perception: PerceptionResult) -> AdvisoryResult:
        """只依据传入的识别结果给出风险与接管请求，不调用语言模型决策。"""
        if self._policy_config is None:
            self._policy_config = load_policy_config(self.cfg.get("policy"))
        decision = evaluate_handover(perception, self._policy_config)
        return AdvisoryResult(
            should_takeover=decision.should_takeover,
            risk_level=decision.risk.risk_level,
            text=render_policy(decision),
            evidence=[*decision.risk.reasons, *decision.reasons],
            source="policy",
            policy_details=decision.to_dict(),
        )

    # --- 统一入口 ---

    def generate(
        self,
        visibility: VisibilityVerdict | None = None,
        perception: PerceptionResult | None = None,
    ) -> AdvisoryResult:
        """按可用信息选择路径。

        优先级：BLIND 直接出接管提示（不浪费算力跑感知）；
        否则若有感知结果则走感知路径，没有则请求接管。
        """
        if visibility is not None and visibility.level is VisibilityLevel.BLIND:
            return self.from_visibility(visibility)

        if perception is not None:
            try:
                if visibility is not None:
                    perception = replace(
                        perception, visibility_level=visibility.level.value,
                        visibility_confidence_multiplier=visibility.confidence_multiplier,
                        visibility_reasons=list(visibility.triggered),
                    )
                return self.from_perception(perception)
            except (ValueError, TypeError, OSError) as exc:
                logger.exception("感知策略无法完成判断")
                return self.from_unavailable(f"感知策略不可用: {exc}")

        if visibility is not None:
            return self.from_unavailable("只有能见度结果，缺少本帧路况和目标识别结果")

        # 既没有门控也没有感知：无法判断时必须明确请求接管。
        logger.error("既无门控结果也无感知结果，输出保守兜底文案")
        return self.from_unavailable("无可用门控或感知输入")
