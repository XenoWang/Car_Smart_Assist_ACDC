"""提示词模板与结构化输入的渲染。

职责:
    - 把 PerceptionResult 渲染成紧凑、无歧义的自然语言上下文
    - 维护给司机的固定文案（模板路径，LLM 失效时的兜底）
    - 保证模板与 build_instruction_dataset.py 一致

为什么模板路径要长期保留，而不是等 LLM 上线后删掉:
    这是**降级兜底**，不是「临时方案」。语言模型可能加载失败、超时、
    输出为空，而驾驶场景不允许「本帧无输出」。模板文案永远可用、
    永远确定、永远保守，是整条链路的最后一道防线。

文案措辞约束（安全相关，改之前先读）:
    · 必须包含明确的**动作**，不能只有描述（「能见度低」不合格，
      「能见度不足，请立即接管」合格）
    · 不承诺系统做不到的事 —— 不用「安全」「放心」「无需注意」这类词
    · 不使用可能引起恐慌的措辞，但也不能弱化到被忽略
    · 长度受控：驾驶场景下来不及读长句，建议 ≤ 30 字
"""

from __future__ import annotations

from car_smart_assist.advisory.schema import RiskLevel
from car_smart_assist.perception.visibility.gate import VisibilityLevel

# --- 能见度三档的固定文案 ---
VISIBILITY_TEXT: dict[VisibilityLevel, str] = {
    VisibilityLevel.BLIND: "能见度不足，无法判断路况，请立即接管车辆",
    VisibilityLevel.DEGRADED: "能见度下降，请提高注意，随时准备接管",
    # 门控只判断图像是否有足够信息，不负责判断道路是否安全。
    VisibilityLevel.VISIBLE: "图像能见度良好，请继续观察路况",
}

# --- 能见度三档对应的风险等级 ---
# DEGRADED 给 NOTICE 而不是 WARNING：它只要求「提高注意」，
# 还没有到「必须做某个动作」的程度。区分这一点是为了让提示强度
# 与真实风险匹配 —— 总是喊狼来了，真的危险时司机就不信了。
VISIBILITY_RISK: dict[VisibilityLevel, RiskLevel] = {
    VisibilityLevel.BLIND: RiskLevel.CRITICAL,
    VisibilityLevel.DEGRADED: RiskLevel.NOTICE,
    VisibilityLevel.VISIBLE: RiskLevel.NONE,
}

# --- 兜底文案：门控与感知都不可用时 ---
FALLBACK_TEXT = "系统无法判断路况，请立即接管车辆"

# 文案长度上限（字符）。超过这个长度的提示在驾驶场景里读不完。
MAX_TEXT_CHARS = 30


def render_visibility(level: VisibilityLevel) -> tuple[str, RiskLevel]:
    """能见度等级 -> (文案, 风险等级)。

    未知等级退回保守文案，而不是返回空字符串 —— 空字符串会让下游
    「成功返回但内容为空」，比明确的兜底更危险。
    """
    text = VISIBILITY_TEXT.get(level)
    risk = VISIBILITY_RISK.get(level)
    if text is None or risk is None:
        return "路况未知，请注意观察", RiskLevel.NOTICE
    return text, risk


def check_wording(text: str) -> list[str]:
    """检查一条文案是否违反措辞约束。返回违规说明列表。

    供测试与评测使用 —— 报告里「提示文案是否越权承诺」这一项
    需要的是可编程的判据，而不是人工肉眼过一遍。
    """
    problems: list[str] = []
    if len(text) > MAX_TEXT_CHARS:
        problems.append(f"长度 {len(text)} 超过上限 {MAX_TEXT_CHARS}")
    for banned in ("安全", "放心", "无需注意", "绝对", "自动驾驶"):
        if banned in text:
            problems.append(f"含越权措辞: {banned}")
    return problems
