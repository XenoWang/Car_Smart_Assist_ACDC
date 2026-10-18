# 风险与接管输出

## 输入

`PerceptionResult` 提供能见度、天气属性、目标列表及各分支是否可用。

- `object_detection_available=true` 且目标列表为空，表示检测成功但没有框；空框不单独触发接管。
- 检测不可用、目标距离／方向缺失或置信度无效，会影响判断可靠性；天气未知只作诊断。
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
风险与辅助接管共用 `src/car_smart_assist/advisory/policy/risk.py::confirmed_weather`。已确认的 `weather_attributes` 优先；多属性取最高基础风险和最大距离倍率，不叠乘，不再统一用0.7筛一次天气概率。仅属性输入也可用，概率只作有效性检查。

仅当新属性与概率均为空且格式有效时，才读取旧 `road_condition`，其原始置信度仍需达到配置门槛。新属性为空但概率有值，表示没有确认正属性，不能回退旧天气。任一新属性名称、格式或概率非法时，整份新天气不采信，只记诊断，不回退旧字段，也不单独请求接管。

## 接管请求

`handover_rules.evaluate_handover` 返回 `should_takeover`、`unable_to_judge`、动作、原因和风险结果。
以下条件可请求接管：

1. 无法可靠判断。
2. 已知风险达到 critical。
3. 可用接管分类头要求接管。
4. 已确认配置内的恶劣天气，且能见度为 DEGRADED。

BLIND 在 pipeline 中阻断感知，直接生成接管提示。
可信的低接管等级不能覆盖其他接管条件；接管分类头默认可缺省。
目标和接管头按原始置信度与既有门槛比较；DEGRADED另作风险证据。0.6乘子与 `effective_confidence` 保留作兼容，不再把折减值与0.7可靠性门槛混用。距离／方向未知、低置信度和critical仍沿用保守处理。

fog／rain／snow与DEGRADED可以联合辅助接管；night单独不触发该条件。兼容的旧天气也使用相同辅助规则。

## 门控边界

信息量低于原BLIND阈值时直接阻断，不因AE缺校准降为DEGRADED。information为NaN或无穷大时抛出评分故障，由pipeline既有兜底输出unknown风险并请求接管。

默认 `use_recon_z=false` 时，诊断用重建z不改变档位。显式启用重建判据后，z为NaN且要求校准时走故障；不要求校准时跳过重建判据；z为无穷大走故障。已确定的信息量BLIND优先于这些重建检查。AE本轮仍会运行，没有实现跳过AE的加速路径。

门控故障仍导致同批帧一起保守兜底，坏图解码和逐帧隔离未在本轮修改。YOLO尚不输出目标距离与方向，检出目标后仍可能因这些未知信息请求接管。

## 输出

`AdvisoryResult` 包含风险等级、提示文本、`should_takeover`、`evidence`、`source` 和 `policy_details`。能见度输出中的information／recon_z若为非有限诊断值，会序列化为null；内部评分及故障接管语义不变。
模板依据规则结果渲染；缺少可用输入或策略故障时返回兜底提示。
当前输出接管请求，没有驾驶员确认或车辆控制执行接口。

## 本轮验证（2026-10-18，Alpha）

此前571项回归通过，包含真实已校准best和未校准last的黑帧阻断及严格JSON输出，本轮收尾未重跑。三模型独立重训、最终评估与新权重六帧pipeline检查已完成；黑帧BLIND接管并跳过感知，正常与四类天气帧三分支可用，输入及严格JSON检查通过。原配置数值和默认权重未改；距离／方向、坏图逐帧隔离和全设备覆盖仍见前述限制。
