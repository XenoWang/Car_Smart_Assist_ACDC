"""Stage 2 决策与语言子包：策略 + 提示 + 生成。

对外契约:
    PerceptionResult   Stage 1 -> Stage 2 的结构化输入
    AdvisoryResult     Stage 2 的输出（给司机的文本 + 可追溯证据）
    AdvisoryGenerator  Stage 2 的唯一入口
"""

from car_smart_assist.advisory.generator import AdvisoryGenerator
from car_smart_assist.advisory.schema import (
    AdvisoryResult,
    PerceptionResult,
    RiskLevel,
    TargetDirection,
    TargetObject,
)

__all__ = [
    "AdvisoryGenerator",
    "AdvisoryResult",
    "PerceptionResult",
    "RiskLevel",
    "TargetDirection",
    "TargetObject",
]
