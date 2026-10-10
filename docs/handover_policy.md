# 风险与接管输出

## 输入

`PerceptionResult` 提供能见度、天气属性、目标列表及各分支是否可用。

- `object_detection_available=true` 且目标列表为空，表示检测成功但没有框；空框不单独触发接管。
- 分支不可用、字段缺失或无效值，都作为判断不可用的证据。
- 目标距离使用米，缺失为 `None`；方向未知为 `unknown`。
- 天气为 fog／rain／snow／night 独立属性。天气不确定不单独触发接管。

字段定义见 [标签与接口](label_spec.md)。

## 风险等级

`risk.assess_risk` 返回最高已知风险、识别证据、诊断、缺失信息和 `reliable`。

| 等级 | 含义 |
|---|---|
| none | 没有触发当前风险规则 |
| notice | 轻度提醒 |
| warning | 风险提醒 |
| critical | 紧急风险条件 |
| unknown | 无法形成可靠判断 |

存在部分有效证据时保留其风险等级，同时可设置 `reliable=false`。
距离、方向、天气倍率和置信度门槛读取 `configs/model/advisory_llm.yaml → advisory.policy`。

有效目标的规则距离：

```text
保守距离 = max(0, distance_m - uncertainty_sigma * distance_uncertainty_m)
规则距离 = 保守距离 / weather_risk_multiplier / direction_factor
```

按配置中的 critical／warning／notice 距离阈值分档，等于阈值时落入下一档。
目标天气倍率目前读取旧的 `road_condition` 字段；独立多标签天气用于提醒和辅助接管。

## 接管请求

`handover_rules.evaluate_handover` 返回 `should_takeover`、`unable_to_judge`、动作、原因和风险结果。
以下条件可请求接管：

1. 无法可靠判断。
2. 已知风险达到 critical。
3. 可用接管分类头要求接管。
4. 已确认配置内的恶劣天气，且能见度为 DEGRADED。

BLIND 在 pipeline 中阻断感知，直接生成接管提示。
可信的低接管等级不能覆盖其他接管条件；接管分类头默认可缺省。
DEGRADED 置信度乘子与可靠性门槛目前存在配置冲突，仍待调整。

## 输出

`AdvisoryResult` 包含风险等级、提示文本、`should_takeover`、`evidence`、`source` 和 `policy_details`。
模板依据规则结果渲染；缺少可用输入或策略故障时返回兜底提示。
当前输出接管请求，没有驾驶员确认或车辆控制执行接口。
