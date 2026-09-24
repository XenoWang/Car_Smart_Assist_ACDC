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

    未实现 —— 感知驱动的常规路径（from_perception）。
        Stage 1 的多任务模型尚未训练，其输出契约（PerceptionResult）
        虽已定义但还没有真实填充者。这里留出接口并给出明确的未实现提示，
        而不是写一个看起来能跑、实际返回假数据的占位实现 ——
        在安全相关的链路上，假数据比缺实现危险得多。

设计原则:
    1. **决策与表达分离**。是否接管由规则/门控决定，语言模型只负责措辞。
    2. **绝不静默失声**。任何失败路径都必须返回保守文案。
    3. **evidence 必须可追溯**。每条提示都要能说清是哪个信号触发的。
"""

from __future__ import annotations

import logging
from typing import Any

from car_smart_assist.advisory.prompt.templates import FALLBACK_TEXT, render_visibility
from car_smart_assist.advisory.schema import (
    AdvisoryResult,
    PerceptionResult,
    RiskLevel,
)
from car_smart_assist.perception.visibility.gate import (
    VisibilityLevel,
    VisibilityVerdict,
)

logger = logging.getLogger(__name__)


class AdvisoryGenerator:
    """把结构化输入转成给司机的提示。

    Args:
        cfg: configs/model/advisory_llm.yaml 的 ``advisory`` 段。
            当前只用到是否需要接管文案；LLM 后端接入后从这里读模型配置。
        llm_backend: 语言模型后端（尚未实现）。为 None 时全部走模板 ——
            模板路径不是「临时方案」，它是 LLM 失效时的固定兜底，要长期保留。
    """

    def __init__(self, cfg: dict[str, Any] | None = None, llm_backend: Any | None = None) -> None:
        self.cfg = cfg or {}
        self.llm_backend = llm_backend

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
            risk_level=RiskLevel.CRITICAL,
            text=FALLBACK_TEXT,
            evidence=["无法形成可靠判断", reason],
            source="fallback",
        )

    # --- 感知驱动的路径（尚未实现）---

    def from_perception(self, perception: PerceptionResult) -> AdvisoryResult:
        """感知结果 -> 司机提示。

        **尚未实现。** 需要先完成：
            1. Stage 1 多任务模型的训练（perception/models/multitask.py）
            2. 决策规则引擎（advisory/policy/handover_rules.py、risk.py）
            3. 提示词渲染（advisory/prompt/templates.py）

        这里不返回假数据，而是明确抛错 —— 在安全相关的链路上，
        一个「看起来能跑但输出是编的」的占位实现，比缺实现危险得多：
        它会被当成真的用起来。

        过渡期请走 from_visibility：门控路径是完整可用的。
        """
        raise NotImplementedError(
            "感知驱动的建议生成尚未实现（依赖 Stage 1 训练与策略引擎）。"
            "当前可用的完整路径是 from_visibility —— 门控驱动的接管提示。"
        )

    # --- 统一入口 ---

    def generate(
        self,
        visibility: VisibilityVerdict | None = None,
        perception: PerceptionResult | None = None,
    ) -> AdvisoryResult:
        """按可用信息选择路径。

        优先级：BLIND 直接出接管提示（不浪费算力跑感知）；
        否则若有感知结果则走感知路径，没有则退回门控文案。
        """
        if visibility is not None and visibility.level is VisibilityLevel.BLIND:
            return self.from_visibility(visibility)

        if perception is not None:
            try:
                return self.from_perception(perception)
            except NotImplementedError as exc:
                # 有感知结果但策略链未就绪，不能把能见度判断冒充接管判断。
                logger.warning("感知决策路径不可用，输出接管兜底: %s", exc)
                return self.from_unavailable("感知决策路径尚未实现")

        if visibility is not None:
            return self.from_visibility(visibility)

        # 既没有门控也没有感知：无法判断时必须明确请求接管。
        logger.error("既无门控结果也无感知结果，输出保守兜底文案")
        return self.from_unavailable("无可用门控或感知输入")
