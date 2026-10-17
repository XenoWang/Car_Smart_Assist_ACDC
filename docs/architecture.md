# 架构示意图

## 当前推理链路

```mermaid
flowchart TD
    Image[单帧图片] --> RGB[读取并转换为 RGB]
    RGB --> Gate[能见度评分与门控]
    Gate --> Level{门控结果}
    Level -->|BLIND| Blind[阻断天气与目标感知]
    Blind --> Direct[直接生成接管提示]
    Level -->|VISIBLE / DEGRADED| Weather[默认预训练属性注意力天气模型]
    Choice[默认天气路由或显式配置] -.-> Weather
    Level -->|VISIBLE / DEGRADED| YOLO[联合 YOLO：八类交通目标＋道路杂物候选]
    Weather --> Warning[独立天气提醒 weather_warning]
    Weather -->|属性与概率| Perception[PerceptionResult]
    YOLO -->|原图框、类别、置信度、分支可用状态| Perception
    Gate -->|能见度与置信度倍率| Perception
    Perception --> Risk[风险评估与判断可靠性]
    Perception -->|结构化接管证据| Handover
    Risk --> Handover[接管请求规则]
    Handover --> Template[模板提示与证据]
    Template --> Result[PipelineResult / AdvisoryResult]
    Direct --> Result
    Warning --> Result
    Gate -.->|评分运行故障| Fallback[不可用状态与兜底提示]
    YOLO -.->|检测运行故障| Fallback
    Fallback --> Result
```

天气结果先用来提醒驾驶员；识别到恶劣天气、同时能见度降到 DEGRADED 时，也会给接管规则提供辅助信息。目标风险计算目前仍读旧的天气字段。
YOLO 目前给出检测框，还不预测距离和方向。学习式接管分类头、分割模型和 LLM 也还没接上。

## 预训练属性注意力天气模型

```mermaid
flowchart TD
    Data[已有录制组四分与图像缓存] --> Train[训练集：来源、标签和录制组采样]
    Train --> Encoder[MobileNetV3 预训练特征；按配置微调后层，BN冻结]
    Encoder --> Spatial[stride 8／16／32 三层空间特征]
    Spatial --> Queries[四属性查询分别选择空间证据]
    Spatial --> Context[全局平均与最大特征]
    Context --> Queries
    Queries --> ASL[雾／雨／雪／夜独立输出与 ASL]
    ASL --> Validation[每轮验证：来源等权 AP＋F1，雨雾加权]
    Validation --> Best[验证选 best；每轮保存 last 与随机状态]
    Best --> Fusion[推理副本合并卷积与 BN]
    Fusion --> Calibration[独立校准集确定四属性阈值]
    Calibration --> Test[开发测试：最后评估]
    Old[原模型及已有金字塔 ASL，仅作对照] --> Test
    Test --> Report[model_report.md 与分来源、录制组、逐图成绩]
    Calibration -.-> Pipeline[默认路由或显式配置接入 pipeline]
```

A组配方完成300轮，沿用原500轮学习率计划的前300轮；SWAD另做300轮首轮与600轮续接，累计900轮。验证选权重，校准选阈值，预算结束后再评估开发测试。
推理优化作用于独立副本，训练权重结构不变；天气不确定仍不单独触发接管。

## A 组与 SWAD

```mermaid
flowchart TD
    Data[同一录制组四分，种子42] --> Train[ImageNet起点，A组微调范围，ASL]
    Train --> Step[每次有效参数更新]
    Step --> Single[每轮验证评分选择单点best]
    Step --> Dense[第6轮起密集平均，每轮形成一个区间]
    Val[仅验证集：来源等权BCE] --> Valley[检测稳定谷底和过拟合尾部]
    Dense --> Valley
    Valley --> Average[保留谷底区间的平均权重]
    Single --> CalSingle[单点权重独立校准]
    Average --> CalAverage[平均权重独立校准]
    Single --> Compare[只比较验证评分，选择导出best]
    Average --> Compare
    Compare --> Export[导出被选中版本，保留另一份候选]
    CalSingle --> Export
    CalAverage --> Export
    Export --> Test[预算结束后做开发测试]
    Prior[A组第85轮与第147轮基线，只读] --> Test
    Test --> Report[分属性、来源及昼夜雨条件报告]
    Step --> Last[每轮保存完整last训练状态]
    Last --> Continue[独立续接：保留第300轮last状态，再训练600轮]
    Continue --> NewAverage[重开平均区间；实际收集301—526轮]
    NewAverage --> Decision{验证评分胜过A85及同轨迹单点？}
    Decision -->|是| NewDefault[选择SWAD]
    Decision -->|否；本轮结果| Restore[恢复A85]
    NewAverage --> ReadOnly[复制权重、状态与配置为只读对照]
    NewDefault --> Check[显式与默认pipeline实际验证]
    Restore --> Check
    Check --> Pipeline[默认路由；显式参数仍可覆盖]
```

300 轮预算及原 500 轮学习率计划保留；谷底结束只停止平均，不提前结束普通训练。BN 继续使用冻结的预训练统计，推理读取一份普通模型，没有线上多模型平均。

默认路由在 `configs/model/weather_default.yaml`。原300轮导出第52轮单点，追加600轮没有刷新这个best；新SWAD验证评分仍不达标，因此默认已按约定恢复A85。两次SWAD及其状态都有只读对照。谷底关闭只停止平均，普通训练仍完成累计900轮；追加学习率从第300轮末值连续衰减，不重置优化器或随机序列。旧雨天残差增强实现和多余权重已删除，旧CNN／金字塔ASL保留对照及共享工具。
