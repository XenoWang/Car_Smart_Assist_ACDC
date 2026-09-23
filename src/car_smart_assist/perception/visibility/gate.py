"""能见度门控的判定规则。

职责:
    - 把 scorer 给出的两路信号（重建误差 z 分数、信息量分数）合成三档判定
    - 判定为 BLIND 时，下游感知应当直接跳过并给出接管请求
    - 记录**触发原因**，让每一个判定都能被追溯和复现

判定哲学（与 docs/handover_policy.md 的「宁可保守」一致，但有重要区别）:
    「宁可保守」在接管决策上意味着倾向接管；但在**门控**上不能这么用 ——
    把可用的帧误判为 BLIND 会让系统在能工作时拒绝工作，
    用户很快就会学会忽略它，真的看不见时也不再相信。
    所以这里的原则是：

        **高信息量的帧永远不判 BLIND。**
        信息量够 = 画面里有东西 = 至少还能做点判断，
        此时再异常（比如隧道、施工区这类训练集里没见过的新奇场景）
        也只降到 DEGRADED，把「要不要接管」交回给常规的接管边界逻辑。

        **只有信息量塌陷才判 BLIND。**
        因为「看不清」的定义就是「画面里没有可用信息」，
        这与场景是否熟悉无关。

    这个不对称是刻意的，也是本模块与普通异常检测最大的区别 ——
    普通异常检测会把「新奇」当成「危险」，在驾驶场景里那是错的。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from car_smart_assist.perception.visibility.scorer import VisibilityScore

logger = logging.getLogger(__name__)


class VisibilityLevel(str, Enum):
    """能见度三档。"""

    VISIBLE = "visible"      # 可用，正常走感知链路
    DEGRADED = "degraded"    # 能见度下降，可用但需降低置信度、倾向提示
    BLIND = "blind"          # 看不清，跳过感知，直接请求接管


@dataclass
class GateThresholds:
    """判定阈值。默认值需用真实数据标定，见 configs/model/visibility.yaml。"""

    # 信息量分数低于此值 -> 直接判 BLIND（画面里几乎没有可用信息）
    info_blind: float = 0.34
    # 信息量分数低于此值 -> 至少判 DEGRADED
    info_degraded: float = 0.50
    # 重建 z 分数高于此值 -> 至少判 DEGRADED
    z_degraded: float = 3.0
    # 重建 z 分数高于此值**且**信息量低于 info_degraded -> 判 BLIND
    z_blind: float = 8.0

    # ⚠️ 是否用重建误差参与判定。**默认关闭**，这是实测结论，不是保守设置。
    #
    # 实测（artifacts/reports/visibility/，150 张/组）：
    #   清晰图          recon_z = -0.57
    #   合成 fog        recon_z = -1.80      ← 比清晰图**低**
    #   合成 darkness   recon_z = -1.80
    #   合成 occlusion  recon_z = -1.60
    #   合成 blur       recon_z = -1.36
    #   真实 fog        recon_z = -1.58      ← 比合成 blur 还低
    #
    # 也就是说重建误差与退化程度**系统性反相关**：雾和黑暗抹掉了高频细节，
    # 图像变得低秩、更容易重建，AE 反而重建得更好。
    #
    # 后果是两条路都走不通：
    #   · `z > z_blind` 永远不对退化图触发 —— 死规则
    #   · 反过来用「z 低即退化」会命中真实浓雾（-1.58），
    #     而浓雾是 ACDC 四个合法子集之一，把它判成 BLIND 等于废掉整个子集
    #
    # 所以现在判定只依赖信息量特征，重建误差仍然计算并写进报告供人工观察，
    # 但不参与决策。要重新启用需先证明它在新模型上具备判别力 ——
    # 见 scripts/evaluate_visibility.py 输出的逐组 recon_z 分布。
    use_recon_z: bool = False

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "GateThresholds":
        return cls(
            info_blind=float(cfg.get("info_blind", 0.34)),
            info_degraded=float(cfg.get("info_degraded", 0.50)),
            z_degraded=float(cfg.get("z_degraded", 3.0)),
            z_blind=float(cfg.get("z_blind", 8.0)),
            use_recon_z=bool(cfg.get("use_recon_z", False)),
        )


@dataclass
class VisibilityVerdict:
    """一次门控判定。"""

    level: VisibilityLevel
    reason: str
    triggered: list[str] = field(default_factory=list)
    information: float = float("nan")
    recon_z: float = float("nan")
    # 供上游日志与人工复核：完整的打分明细
    score: VisibilityScore | None = None

    @property
    def allows_perception(self) -> bool:
        """是否允许继续执行下游感知。"""
        return self.level is not VisibilityLevel.BLIND

    @property
    def confidence_multiplier(self) -> float:
        """给下游感知结果的置信度乘子。

        DEGRADED 时下调下游置信度，让决策层更倾向提示与接管；
        BLIND 时没有下游结果，返回 0。
        """
        return {VisibilityLevel.VISIBLE: 1.0, VisibilityLevel.DEGRADED: 0.6}.get(
            self.level, 0.0
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level.value,
            "reason": self.reason,
            "triggered": self.triggered,
            "information": self.information,
            "recon_z": self.recon_z,
            "allows_perception": self.allows_perception,
            "confidence_multiplier": self.confidence_multiplier,
        }


class VisibilityGate:
    """能见度门控：VisibilityScore -> VisibilityVerdict。

    Args:
        thresholds: 判定阈值
        require_calibration: 为 True 时，缺少零校准统计就拒绝给出 BLIND 判定
            （只有信息量这一路信号时，误判代价由信息量单点承担，风险偏高）。
    """

    def __init__(
        self, thresholds: GateThresholds | None = None, require_calibration: bool = True
    ) -> None:
        self.thresholds = thresholds or GateThresholds()
        self.require_calibration = require_calibration

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "VisibilityGate":
        return cls(
            thresholds=GateThresholds.from_config(cfg.get("thresholds", {})),
            require_calibration=bool(cfg.get("require_calibration", True)),
        )

    def judge(self, score: VisibilityScore) -> VisibilityVerdict:
        """对单帧打分做判定。"""
        t = self.thresholds
        info = score.information
        z = score.recon_z
        has_cal = z == z  # NaN 检查：无校准时 z 为 NaN

        triggered: list[str] = []

        # --- BLIND 规则 1：信息量塌陷。这条与场景是否新奇无关 ---
        if info < t.info_blind:
            triggered.append("info_low")

        # --- BLIND 规则 2：极度异常 **且** 信息量已经偏低（双条件，缺一不可）---
        # 加 info 条件的理由见模块头：高信息量的新奇场景不能被判看不见
        #
        # 默认不启用（t.use_recon_z=False），因为实测重建误差与退化反相关，
        # 这条规则在真实数据上永远不触发 —— 详见 GateThresholds.use_recon_z 的说明
        if t.use_recon_z and has_cal and z > t.z_blind and info < t.info_degraded:
            triggered.append("recon_extreme_low_info")

        if triggered:
            if not has_cal and self.require_calibration:
                # 只有信息量信号可用，不足以判 BLIND —— 降级为 DEGRADED 并说明原因
                return self._make(
                    VisibilityLevel.DEGRADED,
                    "信息量极低，但缺少零校准统计，保守降级为 DEGRADED；"
                    "请用 scripts/train_visibility.py 训练以获得完整判定能力",
                    ["info_low", "no_calibration"],
                    score,
                )
            return self._make(
                VisibilityLevel.BLIND,
                f"信息量 {info:.3f} < {t.info_blind}"
                if "info_low" in triggered
                else f"重建 z 分数 {z:.1f} > {t.z_blind} 且信息量 {info:.3f} 偏低",
                triggered,
                score,
            )

        # --- DEGRADED ---
        if info < t.info_degraded:
            triggered.append("info_reduced")
        if t.use_recon_z and has_cal and z > t.z_degraded:
            triggered.append("recon_anomalous")
        if triggered:
            return self._make(
                VisibilityLevel.DEGRADED,
                f"能见度下降：信息量 {info:.3f}，重建 z 分数 "
                f"{z:.1f}" if has_cal else f"能见度下降：信息量 {info:.3f}",
                triggered,
                score,
            )

        return self._make(VisibilityLevel.VISIBLE, "信息量与重建误差均在正常范围", [], score)

    @staticmethod
    def _make(
        level: VisibilityLevel,
        reason: str,
        triggered: list[str],
        score: VisibilityScore,
    ) -> VisibilityVerdict:
        return VisibilityVerdict(
            level=level,
            reason=reason,
            triggered=list(triggered),
            information=score.information,
            recon_z=score.recon_z,
            score=score,
        )

    def judge_many(self, scores: list[VisibilityScore]) -> list[VisibilityVerdict]:
        return [self.judge(s) for s in scores]
