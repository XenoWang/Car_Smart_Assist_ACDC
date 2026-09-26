"""读取并校验规则策略配置；所有阈值来源于 advisory_llm.yaml。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from car_smart_assist.advisory.schema import RiskLevel


@dataclass(frozen=True)
class PolicyConfig:
    distance_thresholds: Mapping[str, float]
    weather_risk_multiplier: Mapping[str, float]
    weather_risk_levels: Mapping[str, RiskLevel]
    handover_level_to_action: Mapping[int, str]
    min_confidence_for_clear: float
    oncoming_stricter_factor: float
    uncertainty_sigma: float
    max_distance_uncertainty_ratio: float
    require_distance_uncertainty: bool
    require_handover_head: bool

    @classmethod
    def from_mapping(cls, cfg: Mapping[str, Any]) -> PolicyConfig:
        # 配置模块也可独立导入，避免 advisory 的公开入口与配置形成循环依赖。
        from car_smart_assist.advisory.schema import RiskLevel

        def number(value: Any, name: str, minimum: float = 0.0) -> float:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"policy.{name} 必须是数值")
            value = float(value)
            if not isfinite(value) or value < minimum:
                raise ValueError(f"policy.{name} 必须是大于等于 {minimum} 的有限值")
            return value

        try:
            if set(cfg) != set(cls.__dataclass_fields__):
                raise ValueError("policy 配置项缺失或包含未知字段")
            if set(cfg["distance_thresholds"]) != {"critical", "warning", "notice"}:
                raise ValueError("distance_thresholds 仅接受 critical/warning/notice")
            distances = {
                k: number(cfg["distance_thresholds"][k], k)
                for k in ("critical", "warning", "notice")
            }
            if not 0 < distances["critical"] < distances["warning"] < distances["notice"]:
                raise ValueError(
                    "policy.distance_thresholds 必须满足 0 < critical < warning < notice"
                )
            confidence = number(cfg["min_confidence_for_clear"], "min_confidence_for_clear")
            if not 0 < confidence <= 1:
                raise ValueError("policy.min_confidence_for_clear 必须在 (0, 1] 内")
            weather = {
                str(k): number(v, f"weather_risk_multiplier.{k}", 1.0)
                for k, v in cfg["weather_risk_multiplier"].items()
            }
            levels = {str(k): RiskLevel(v) for k, v in cfg["weather_risk_levels"].items()}
            if (
                not weather
                or weather.keys() != levels.keys()
                or RiskLevel.UNKNOWN in levels.values()
            ):
                raise ValueError("天气风险等级和倍率必须有相同的非空类别，等级不能为 unknown")
            raw_actions = cfg["handover_level_to_action"]
            if any(type(k) is not int and k not in ("0", "1", "2") for k in raw_actions):
                raise ValueError("接管等级键必须为整数 0/1/2 或其字符串")
            actions = {int(k): str(v) for k, v in raw_actions.items()}
            if len(actions) != len(raw_actions):
                raise ValueError("接管等级键不能重复")
            if (
                set(actions) != {0, 1, 2}
                or not set(actions.values())
                <= {
                    "no_action",
                    "light_notice",
                    "request_takeover",
                }
                or actions[2] != "request_takeover"
            ):
                raise ValueError("接管头需要 0/1/2 三档有效动作，2 必须为 request_takeover")
            for key in ("require_distance_uncertainty", "require_handover_head"):
                if not isinstance(cfg[key], bool):
                    raise ValueError(f"policy.{key} 必须是布尔值")
            return cls(
                distance_thresholds=MappingProxyType(distances),
                weather_risk_multiplier=MappingProxyType(weather),
                weather_risk_levels=MappingProxyType(levels),
                handover_level_to_action=MappingProxyType(actions),
                min_confidence_for_clear=confidence,
                oncoming_stricter_factor=number(
                    cfg["oncoming_stricter_factor"], "oncoming_stricter_factor", 1
                ),
                uncertainty_sigma=number(cfg["uncertainty_sigma"], "uncertainty_sigma"),
                max_distance_uncertainty_ratio=number(
                    cfg["max_distance_uncertainty_ratio"], "max_distance_uncertainty_ratio"
                ),
                require_distance_uncertainty=cfg["require_distance_uncertainty"],
                require_handover_head=cfg["require_handover_head"],
            )
        except (KeyError, TypeError, AttributeError, OverflowError) as exc:
            raise ValueError(f"policy 配置缺失或结构不正确：{exc}") from exc


def load_policy_config(
    overrides: Mapping[str, Any] | None = None,
    path: str | Path | None = None,
) -> PolicyConfig:
    """加载项目 YAML 的 policy 段并应用覆盖项；生成器首次评估时读取一次。"""
    source = (
        Path(path)
        if path is not None
        else (Path(__file__).resolve().parents[3] / "configs/model/advisory_llm.yaml")
    )
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
        cfg = dict(raw["advisory"]["policy"])
    except (yaml.YAMLError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"policy YAML 配置无效：{exc}") from exc
    if overrides is not None and not isinstance(overrides, Mapping):
        raise ValueError("policy 覆盖配置必须是字典")
    for key, value in (overrides if overrides is not None else {}).items():
        if key not in cfg:
            raise ValueError(f"未知 policy 配置项：{key}")
        if isinstance(cfg[key], dict) and isinstance(value, Mapping):
            cfg[key] = {**cfg[key], **value}
        else:
            cfg[key] = value
    return PolicyConfig.from_mapping(cfg)
