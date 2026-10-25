# 项目视觉模型优化论文调研与落地建议

调研日期：2026-10-23（Asia/Shanghai）  
项目：Car Smart Assist ACDC；根目录 `E:/Car_Smart_Assist_ACDC`  
性质：论文研究与实施决策支持，尚未选择或实施下述新策略。

## 1. 结论与推荐顺序

按本次要求，YOLO 以“精度守住后降低实际推理延迟”为目标，天气与能见度以提升有效识别能力为目标。建议优先讨论以下路线：

| 顺序 | 模块与方案 | 为什么适合本项目 | 落地条件与成本 | 本项目效果证据 |
|---|---|---|---|---|
| 1 | YOLO：保留原始检测输出的 TensorRT FP16 后端 | 不先压缩已经很小的 YOLO11n；先尝试运行时优化 | 中；需核对固定依赖、驱动、输入形状和自定义后处理 | 未实测；属于工程方案，不冒充论文方法 |
| 2 | 天气：人工审计标签后，单项试 Group DRO | 当前主要短板是来源间差异，平均分掩盖 PAB 雨漏检 | 中；需要可靠分组、多标签损失适配；推理网络不增加 | 旧研究候选，未找到已实施效果记录 |
| 3 | 能见度：任务相关标注与轻量质量评分；必要时引入 RankIQA 式排序学习 | 当前 AE 重建误差不参与默认判档，继续降低 MSE 不会直接提高门控准确度 | 中至高；主要前提是可信可用性标签 | 新的具体学习方向，未实测 |
| 4 | YOLO：小幅结构化通道剪枝，必要时用检测蒸馏恢复精度 | 通过真实缩小算子降低计算量，以蒸馏保护旧类与小障碍物 | 高；YOLO11n 结构适配、重训和部署图均需验证 | 未实测；只减少 FLOPs 不算成功 |
| 条件项 | 天气：经确认的缺标修正／部分标注学习 | 若雨雾共存被当作负标签，单纯换优化器无法解决监督冲突 | 先审计，再从简单 known-mask 方案开始；CSL 后置 | 缺标数量与成因尚未确认 |
| 后置 | 天气局部线索监督；能见度局部区域评分；YOLO INT8 | 有对应研究依据，但标签、兼容性或精度风险更高 | 中至高，不能直接换配置即用 | 未实测 |

这些名次是基于当前工程条件的判断，不是论文榜单排名。**不承诺任何 FPS、提速倍数或精度增长。** “可以实现”指有明确算法和接入路径；“可以直接部署”还需本项目证据。

## 2. 本次核对的项目基线

| 模块 | 当前配置／实现 | 对调研的约束 |
|---|---|---|
| 检测 | `configs/model/yolo_detection.yaml`：YOLO11n 九类，imgsz=960，IoU=0.5，max_det=100；默认为 `yolo11n_acdc_laf_v2/weights/best.pt` | 保留原八类，新增 road_obstacle；ACDC 与 Lost & Found 的监督区域不同 |
| 检测后处理 | `perception/yolo_postprocess.py`：固定原始最高类，先检查该类阈值资格，再进行 NMS／数量截断 | 导出后不能替换成顺序不同的内嵌 NMS；低精度分数比较、几何及 NMS 至少 FP32 |
| 天气 | `weather_default.yaml` → A85；MobileNetV3-Large、384×672、三层特征、属性查询注意力、ASL、ACDC:PAB=0.85:0.15 | ASL、多尺度、注意力和来源采样已经存在，不重复列成新优化 |
| 能见度 | 多尺度去噪 AE＋对比度、熵、边缘、高频信息；`use_recon_z=false` | AE 仍执行诊断，但默认判档不使用重建 z；精度优化须对准判档依据 |
| 尚未接入 | 学习式距离／方向、分割、学习式接管头与 LLM/VLM | 本报告不把这些占位能力视为已有模型，也不以新增整套系统替代当前优化 |

本机静态包信息：PyTorch 2.8.0+cu129、torchvision 0.23.0+cu129、Ultralytics 8.3.162；目录中有 ONNX 1.23.0 与 ONNX Runtime GPU 1.30.0，未见 TensorRT Python 包。目录存在不证明 GPU provider 可用，本轮未导入或运行这些后端。RTX 5070 来自项目既有真实量测记录，本轮未重新查询设备或驱动。

当前分支读取为 `pipeline_stop_test`，工作区有此前未提交改动。本轮只新增本文；不改现有源码、配置、权重、PRD、台账或训练任务。根目录未发现独立 `AGENTS.md`，本轮遵循用户直接提供的规则及项目台账。

### 2.1 不把历史指标当作新实验

- A85 历史开发测试 macro F1=92.55%；雨正例在 ACDC 检出 232/238，在 PAB 检出 1/40。该差异支持分来源调查，不证明某一种纹理或标签问题就是根因。
- 默认 YOLO 在新后处理口径下，ACDC AP50/AP50–95=26.9073%/14.7834%，L&F 标注 ROI 为 71.4754%/41.4319%；这些不是整图开放集障碍物成绩。
- 2026-10-18 的独立重训候选与默认权重不同。新天气候选 PAB 雨检出 13/40，但整体 macro F1=90.82%，未替换 A85；不能概括为“重训完全无效”。
- 800 轮任务交接记录为用户暂停：天气与 AE 已完成相应训练，YOLO 尚未完成，未发布默认包。用户暂停不等于算法失败。本次没有读取正在进行任务的进度，也未使用未交付的 600 轮任务结果作推荐依据。
- 项目 test 已多次用于开发分析，只能作为开发回归证据。后续确认泛化应保留新的录制族／场景封存集，不能把反复看过的 test 重新称为未见测试。

## 3. 按用户要求排除与去重

| 已有方向 | 本地证据 | 本次处理 |
|---|---|---|
| SWAD 及无新假设地继续延长该轨迹 | 900 轮实验 `decision.json`：SWAD 验证综合分 0.932971，低于 A85 0.944457 和同次单点 0.945554；`swad_effective=false` | 从近期推荐清单剔除，不再建议“多训几百轮 SWAD” |
| L2-SP | 单项消融平均 F1 仅 +0.04 个百分点，雨 F1 −5.20 个百分点，PAB 雨漏判 40/40 | 按本项目目标排除；不将雾类收益包装成整体适用 |
| 将 AE 重建误差直接作为退化强度／重新打开 recon_z | 台账 P26：平滑或变暗的退化可能更容易重建，误差方向不可靠 | 排除；不能仅取负号、调 z 阈值或降 MSE 宣称解决 |
| 再提出 ASL、多尺度注意力、预训练、A85 采样和来源／ROI 监督 | 当前配置与代码已有 | 保留为基线，不算新增方案 |
| 2026-10-22 三项局部优化：输入扫描、uint8 搬运、天气构建副本 | 已实现并有保真回归；没有实际速度或内存收益量测 | 不重复推荐实施，也不宣称无效或已提速；待独立量测 |
| 真实批处理、information-only、离线分块 | `OPTIMIZATION_ROADMAP_20261022.md` 仅规划 | 不作为新论文发现；批处理可能改善吞吐，不能当单帧低延迟保证 |
| Group DRO、CSL、旧类蒸馏、SAM、JTT、MWSC | V0_0_2 §9 为未选择研究候选，未找到已实施无收益证据 | 允许重新筛选排序，但明确不是首次提出；本轮不把 SAM/JTT 扩成并行训练清单 |

排除的是当前证据下不值得重复投入的方案，不删除历史代码、报告或权重。“已选择但尚无可核验结果”也不自动等于失败；未完成任务保持待定。

## 4. YOLO：先守住精度，再优化速度

### Y1：TensorRT FP16 原始输出后端——优先

**依据与定位。** Ultralytics 和 NVIDIA 官方文档支持 TensorRT 导出与低精度推理；这是工程路线。当前官方支持资料包含 Windows／Blackwell，但滚动文档并不证明项目固定的 Ultralytics 8.3.162 与最新 TensorRT 兼容。[E1][E2][E3]

**建议做法。** 保留同一 `.pt` 基线，先建立 FP32 导出的一致性参照，再尝试 FP16。只导出检测网络原始输出，接回项目现有后处理。构建形状须来自真实 RGB／letterbox 结果；`imgsz=960` 不等于每次输入必然是 960×960，不能为了静态 engine 暗改 rect padding。常见形状可用明确 profile，或经同口径验证的分桶方案。

**接入位置。** `perception/detection_yolo.py`、`perception/yolo_postprocess.py` 和检测配置／产物身份管理。当前 `.pt` 路径经过 Ultralytics 加载；即便底层支持 `.engine`，项目的校准绑定、原始输出和自定义 predictor 合同仍须适配确认，不是只改后缀。

**保准度边界。** 不内嵌不同语义的 NMS，不提前 top-k 截断，不把阈值／框偏移降成 FP16；新 engine、形状、精度和后处理版本需可追溯。先锁定旧阈值作对照，再在独立 calibration 上按原协议校准并报告两套结果。FP16 可以改变网络输出，后处理回升 FP32 不会恢复已经丢失的信息。

**收益判断。** 工程成本中；无需先重训，但后端安装、导出和精度评价有成本。只有端到端 p95/p99 确实下降且精度门槛满足才采用；GPU 前向变快、解码和后处理占比更高时，整条链路收益可能很小。

### Y2：DepGraph 结构化剪枝＋必要的蒸馏恢复——第二阶段

**论文。** Fang 等，*DepGraph: Towards Any Structural Pruning*，CVPR 2023；作者 Torch-Pruning 仓库提供依赖感知通道剪枝，标注 MIT。[R1] 不采用仅把权重置零却保留稠密算子的“伪提速”。

**本项目迁移建议。** 从少量可裁剪通道开始，按实际网络依赖传播，保护检测输出类别数和高分辨率小目标分支；不预设固定剪枝比例。YOLO11n 已是 nano，须先确认是否还有有意义的压缩余量。官方仓库有其他 YOLO 版本示例，不等于已证明适配当前 C3k2／C2PSA／Detect 结构。

**精度恢复参考。** Yang 等，*Focal and Global Knowledge Distillation for Detectors*（FGD），CVPR 2022，区分局部与全局特征知识；官方实现为 Apache-2.0，但依赖旧 MMDetection 2.11.0／mmcv-full 1.2.4，需迁移损失思想，不能直接替换本项目训练器。[R2] DeFeat（CVPR 2021）也提供前景／背景解耦参考，但本轮未确认其代码许可证，不作为首选代码来源。[R3]

**实施边界。** 教师可从本地未压缩九类模型起步；若另用旧八类教师，只在已确认的类别和有效监督范围内约束学生。ACDC 未标杂物、L&F ROI 外区域不能机械当背景蒸馏；教师错误也不能当真值。保留来源监督与真实标签损失，不重写整个框架。

**必须分开的对照。** 原模型、剪枝后恢复训练、剪枝后恢复训练＋蒸馏，按顺序单项评价。蒸馏到同等尺寸学生本身不会减少推理算量；更小学生也只有实际耗时改善才算提速。不能用九类平均分掩盖旧八类或小障碍物退化。

### Y3：INT8 量化——后置条件项

FP16 获得可重复基线后才考虑 INT8。校准图须覆盖训练来源、昼夜、天气与小障碍物，量化范围拟合不读取 test；量化统计和最终类别阈值校准分开记录。项目已存在 ONNX Runtime GPU 包，可作为后端比较候选，但未验证 provider，不能据安装目录承诺可运行。[E1][E2]

本轮不推荐直接替换为新的 YOLO 代际、降低 960 输入尺寸、SAHI 多切片或增加前置去雾模型来完成“提速”：这些会显著改变模型／输入／延迟，且尚未证明能守住小目标召回。

## 5. 天气：优先提高弱来源组与共存天气识别

### W0：先复核训练标签，避免优化错误目标

抽样单位采用完整录制族，核对 ACDC 目录标签的负属性是否确有依据、PAB 元数据是否对应实际可见现象，分别保留正、负、未知和争议。训练、验证、校准按固定协议各司其职；已看 test 中的 39 个漏判可描述缺陷，不能直接搬入训练或用来挑损失参数。

A85 在 PAB 雨上的失败可能来自标签口径、来源混杂、视觉细节、样本覆盖或概率尺度，现有证据不能选定唯一原因。审计后若标签可信，优先 W1；若存在确定缺标，先纠正监督，再比较 W1。

### W1：Group DRO 最差组优化——主要算法候选

**论文。** Sagawa 等，*Distributionally Robust Neural Networks for Group Shifts: On the Importance of Regularization for Worst-Case Generalization*，2019 预印本／2020 修订；作者提供 MIT 代码。[R4] 研究提醒：训练损失接近零时，单独最小化最差训练组损失也可能失效，需要合适正则约束。

**适配方式是本报告建议。** 沿用 A85 网络，基于已知标签构建“来源×属性正负”的训练组；有足够独立录制组时才增加昼夜上下文。每个属性只对 known=1 的样本计算损失，先按有效数量归一化，再做组风险加权，防止大量易负例淹没 PAB 雨正例。标签组可重叠，不能把整图强制成互斥天气类别。

保持 ASL 作为受控起点，但 Group DRO 与 ASL 的组合没有本项目验证，不能宣称现成即用。与原采样策略共同使用时要避免重复放大小组权重。正则强度在训练／验证范围内比较；不自动重启失败的 L2-SP。当前 600 轮任务的禁早停要求保持，本文不改变它；将来独立实验如需不同停止策略，须单独确定。

**接入与代价。** `weather_attention_training.py` 的 unreduced loss／group 聚合与训练数据元信息；推理网络不增加，训练和调参成本中。验收看最差来源组雨／雾 AP、F1、召回及误报，同时守住 snow/night；不是只看平均训练 loss。

### W2：部分标注学习／CSL——只在缺标证实后采用

**论文。** Ben-Baruch 等，*Multi-Label Classification With Partial Annotations Using Class-Aware Selective Loss*，CVPR 2022；作者 PyTorch 实现为 MIT，包含 `PartialSelectiveLoss`。[R5]

先比较最简单的人工核实标签／known-mask 修正，保留可靠负监督。只有部分标注确实普遍且简单 mask 不足时，再考虑 CSL 的选择性学习；论文在大类别图像数据上的收益不等于四属性天气收益。不能根据本模型预测不一致就把标签改为未知，不能把所有非目录属性一律置未知，更不能把 known=0 当负样本。

接入点主要为 `data/weather_pyramid.py` 与天气损失聚合，数据版本／签名也需更新；推理结构可以保持。必须将“标签改善”和“损失方法改善”分开对照，避免无法归因。

### W3：局部证据监督——有标注后再做

MWSC（IEEE Access 2025）研究全局／局部视觉和文本交叉注意力的多标签天气与严重度分类，作者提供 MIT 仓库。[R6] 中文论文《基于图像分块和多头注意力机制的气象识别研究》（2024）也讨论局部块特征，但属于 Softmax 分类，不能直接代替当前共存属性输出。[R7]

本项目 A85 已有多尺度属性空间注意力，单纯再加注意力不算新优化。可选新增点是：为经确认的雨滴、雨丝、积雪等线索提供局部辅助监督，让模型在保持全图属性判断的同时学习可复核证据。湿路、反光或白色物体不能直接当作天气真值；注意力热图也不是因果证明。

MWSC 数据排除夜间且包含 ACDC 来源，整套 CLIP 路线既增加成本，也有重叠数据和 night 覆盖问题，因此不建议直接替换。MixStyle／CORAL 同样不列近期首选：天气本身就是颜色、纹理和对比度变化，无条件消除“风格”可能去除目标信号。这是项目迁移风险判断，不是论文被证明无效。

## 6. 能见度：优化可用性判断，避免继续追 AE loss

### V1：真实任务标签＋排序学习——准确度主线

**先做的事。** 定义“目标是否仍可辨认／是否应阻断”的可复核标签和不确定项，包含低对比、模糊、遮挡、夜间和局部失真；天气类别不等于 VISIBLE／DEGRADED／BLIND 真值。先用现有四类信息特征建立简单受控参照，再判断是否需要新网络。

**论文。** Liu 等，*RankIQA: Learning From Rankings for No-Reference Image Quality Assessment*，ICCV 2017，通过已知退化顺序学习相对质量，再使用有标签数据微调；作者仓库为 MIT、旧 Caffe 实现。[R8]

**可实现迁移。** 在训练录制组内对同一底图生成可确认顺序的模糊／遮挡等成对退化，用共享轻量编码器学习排序，再用真实人工可用性标签校准三档。保留底图组绑定，不能把一个底图的不同退化版本放到不同集合。仅约束同底图、同退化类型的可信顺序，不强行规定不同场景或雨／雾／夜之间谁更差。

这是对论文训练思想的改造，需要 PyTorch 实现和新监督，不是直接启用现有 AE。真实浓雾、噪声、逆光等未必服从简单合成顺序，因此真实标注是条件，不能只用合成最大强度召回交付。未完成真实验证前保留现有门控，不能直接换阈值上线。

### V2：局部区域质量融合——条件性补充

唐榕、李骞、唐绍恩，《基于多目标的能见度检测方法》，《计算机工程》2023，49(2):314–320，研究不同景深目标子图的质量与动态权重融合。[R9] 这提示全图平均可能掩盖局部可见度差异；本轮未确认作者公开代码，不将其列为可直接复现项目。

本项目可借鉴“局部区域＋全图”评分，优先讨论固定图像区域或独立轻量区域支路。**当前门控先于 YOLO，不能为了评分先调用被门控阻断的 YOLO，形成循环依赖。** 图像纵向位置不等于真实深度，不能将区域分数输出为物理可见米数。完整复现论文的距离相关融合仍需目标深度／能见度真值。

2025 年 *Data Quality Matters: Quantifying Image Quality Impact on Machine Learning Performance* 研究图像变化与下游检测效果的关联，可借鉴其配对评价流程。[R10] 其主要场景是压缩／虚拟化，不能直接证明天气门控有效；配对质量指标依赖参照图，也不能直接成为在线单图门控。

### V3：为何不直接换一个“大质量模型”

MUSIQ（ICCV 2021）有多尺度质量建模与官方推理实现，但通用主观质量分数不等于道路可用性，直接接入还增加框架和模型成本。[R11] 可作为将来离线参考，不能把它的预测当真值自动给训练集打标签。

中文《可见光-远红外图像的多模态能见度深度学习模型》（2021）需要配准的可见光／远红外双模态输入；当前单 RGB 项目不满足条件，故不列入近期方案。[R12] 不为追求论文准度临时扩展传感器任务。

## 7. 实施前应约定的验收指标

以下是未来决策标准，未设计测试文件、未执行测试；不是对当前研究任务追加训练授权。

| 模块 | 主要验收 | 必须同时报告 |
|---|---|---|
| YOLO 精度 | 同输入／同后处理下逐类 AP50、AP50–95；原八类和 road_obstacle 分列；关键类及小框召回守底线 | 固定旧阈值结果＋独立校准后的操作点结果；ROI 与整图标注范围；FP/FN 和按天气／尺寸分组 |
| YOLO 速度 | 实际目标设备上 B=1 端到端 p50/p95/p99 改善，输入／解码条件一致 | 网络前向、CPU 预处理、搬运、NMS、内存峰值；冷启动与稳态分开；吞吐不冒充单帧响应 |
| 天气 | 最差来源组与雨／雾 AP、F1、召回提高，同时限制误报及其他属性退化 | source×属性×昼夜／录制族样本量，macro/micro 分开；AP 不能由阈值调整伪造 |
| 能见度 | 在真实可用性标注上降低危险漏放，控制清晰图误阻断；报告风险—覆盖关系 | 三档混淆、不同退化分组、局部遮挡与模糊；AE MSE 和合成召回仅为辅助 |

“保证准度”需在立项时量化为可接受的非劣界限、关键类漏检底线和置信区间；本报告没有擅自设定允许下降 1% 或任何数值。若要求观测指标一点不降，某些压缩候选可能全部被淘汰，应如实接受结果。

选择实验使用训练／验证信息，阈值使用独立 calibration，最终留出仅用于冻结后的报告。采用录制族而不是相邻帧作为统计重采样单位；只有一个场景的小组不能支撑稳定泛化结论。单因素先成立再尝试组合，不同时叠加剪枝、INT8、输入缩小和蒸馏。

如果后续获选：先写清具体策略及 PRD 范围，再实施；开发结束并冻结后，完成审查、测试方案接受和用户测试授权才运行。当前已经启动或暂停的训练继续遵守原有任务合同，本文不改变预算、早停或发布规则。

## 8. 论文与官方实现目录

访问日期均为 2026-10-23。检索覆盖 CVF、arXiv、OpenReview、国内期刊原站、作者仓库及 NVIDIA／Ultralytics 官方资料；不以博客或二次转载作为技术结论依据。部分站点直开遇到 403、500 或验证页时，使用原站可读取摘要与作者仓库交叉核对，没有冒称逐页全文精读或复现实验。许可证列的是仓库页面标示，权重、数据和依赖须分别核对。

| 编号 | 原始来源／年份 | 实现与可用性 | 本报告用途 |
|---|---|---|---|
| R1 | [DepGraph: Towards Any Structural Pruning，CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Fang_DepGraph_Towards_Any_Structural_Pruning_CVPR_2023_paper.html) | [作者 Torch-Pruning，MIT](https://github.com/VainF/Torch-Pruning)；当前 YOLO11n 需适配 | Y2 结构化压缩 |
| R2 | [Focal and Global Knowledge Distillation for Detectors，CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Yang_Focal_and_Global_Knowledge_Distillation_for_Detectors_CVPR_2022_paper.html) | [作者 FGD，Apache-2.0](https://github.com/yzd-v/FGD)，旧 MMDetection | Y2 剪枝后精度恢复思想 |
| R3 | [Distilling Object Detectors via Decoupled Features，CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/html/Guo_Distilling_Object_Detectors_via_Decoupled_Features_CVPR_2021_paper.html) | [DeFeat 作者实现](https://github.com/ggjy/DeFeat.pytorch)，本轮许可证未确认 | 蒸馏补充，非直接插件 |
| R4 | [Distributionally Robust Neural Networks for Group Shifts: On the Importance of Regularization for Worst-Case Generalization，2019／2020](https://arxiv.org/abs/1911.08731) | [作者 group_DRO，MIT](https://github.com/kohpangwei/group_DRO) | W1 最差组泛化与正则约束 |
| R5 | [Multi-Label Classification With Partial Annotations Using Class-Aware Selective Loss，CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Ben-Baruch_Multi-Label_Classification_With_Partial_Annotations_Using_Class-Aware_Selective_Loss_CVPR_2022_paper.html) | [作者 PartialLabelingCSL，MIT](https://github.com/Alibaba-MIIL/PartialLabelingCSL) | W2 条件性缺标学习 |
| R6 | [Multi-Modal Cross-Attention-Based Multi-Label Weather and Severity Level Classification，IEEE Access 2025，作者项目页](https://yonsei-stl.github.io/MWSC/)；DOI 10.1109/ACCESS.2025.3645363 | [作者 MWSC，MIT](https://github.com/Yonsei-STL/MWSC)；数据无夜间且含 ACDC | W3 局部证据与迁移限制 |
| R7 | [基于图像分块和多头注意力机制的气象识别研究，2024，44(2):83–90](https://journal.lnpu.edu.cn/CN/10.12422/j.issn.1672-6952.2024.02.013) | 国内期刊原站；本轮未确认官方代码 | 中文局部特征参考；单标签方法不直接移植 |
| R8 | [RankIQA: Learning From Rankings for No-Reference Image Quality Assessment，ICCV 2017](https://openaccess.thecvf.com/content_iccv_2017/html/Liu_RankIQA_Learning_From_ICCV_2017_paper.html) | [作者 RankIQA，MIT，Caffe](https://github.com/xialeiliu/RankIQA) | V1 排序预训练思想 |
| R9 | [基于多目标的能见度检测方法，计算机工程，2023](https://www.ecice06.com/CN/10.19678/j.issn.1000-3428.0063745) | 国内期刊原站；未确认官方代码；需要对应真值 | V2 局部质量与融合参考 |
| R10 | [Data Quality Matters: Quantifying Image Quality Impact on Machine Learning Performance，2025](https://arxiv.org/abs/2503.22375) | 原始预印本；本轮未确认作者代码 | 下游检测关联评价，不直接作为天气算法 |
| R11 | [MUSIQ: Multi-scale Image Quality Transformer，ICCV 2021](https://arxiv.org/abs/2108.05997) | [Google Research 官方实现说明](https://github.com/google-research/google-research/blob/master/musiq/README.md) | 通用 IQA 后置对照 |
| R12 | [可见光-远红外图像的多模态能见度深度学习模型，2021](https://www.jcad.cn/article/doi/10.3724/SP.J.1089.2021.18420) | 国内期刊；本轮仅核对可检索原站摘要，直开失败 | 因传感器条件不符排除 |
| R13 | [JarvisIR: Elevating Autonomous Driving Perception with Intelligent Image Restoration，CVPR 2025](https://arxiv.org/abs/2504.04158) | VLM 调度多个恢复专家 | 新近研究补充；因额外推理链路不列本轮提速方案 |
| E1 | [Ultralytics 导出文档](https://docs.ultralytics.com/modes/export/) | 官方滚动文档；参数已演进，需对照 [v8.3.162](https://github.com/ultralytics/ultralytics/tree/v8.3.162) | 后端与原始输出导出 |
| E2 | [NVIDIA TensorRT 支持矩阵](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/support-matrix.html)／[安装前提](https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/prerequisites.html) | Windows、驱动、CUDA、精度支持需按选定版本核对 | 兼容性条件，非本机可运行证据 |
| E3 | [NVIDIA CUDA GPU 能力表](https://developer.nvidia.com/cuda/gpus) | RTX 5070 属 compute capability 12.0 | 目标 GPU 架构核对 |

## 9. 本地证据索引与交付界限

以下路径均相对项目根，保持原文件只读；旧记录的日期、版本与默认／候选身份不能混用：

| 证据 | 用途 |
|---|---|
| `configs/model/{yolo_detection,weather_default,weather_attention_a,visibility}.yaml` | 当前默认模型和参数 |
| `requirements/requirements-{detection,torch}.txt` 与 `.venv` 包元信息 | 固定依赖和静态安装信息 |
| `src/car_smart_assist/perception/{detection_yolo,yolo_postprocess,weather_attention}.py`；`visibility/{gate,scorer}.py` | 接入与语义边界 |
| `artifacts/reports/weather_attention_swad_extend_600/decision.json` | SWAD 未胜出的验证依据 |
| `artifacts/reports/weather_attention_ablation/REPORT.md` | A85／L2-SP 对照和分来源雨误判 |
| `model_report.md` | 2026-10-19 默认 YOLO 后处理指标、2026-10-18 独立重训候选 |
| `AI_Skills/PRD/PRD_V0_0_2.md` §4、§9–13 | 历史候选指标、已有研究方向及尚未选择状态 |
| `AI_Skills/PRD/PRD_V0_0_3.md`；`AI_Skills/Pitfall/ThingsShouldBeAvoided.md` | 已修问题、未知标签、分组、后处理、重建反向及测试边界 |
| `AI_Skills/Reports/Optimization/{LOCAL_OPTIMIZATIONS_20261022,OPTIMIZATION_ROADMAP_20261022}.md`、`AI_Skills/Reports/Training/TRAIN800_PAUSED_20261023.md` | 已实施未测速／仅规划／暂停任务的区分 |

`artifacts/reports/visibility/evaluation.md` 是历史报告，仍有“ACDC 没有看不清帧”等旧说明；本次不沿用这些断言，不把该旧报告的清晰图或合成成绩当作真实驾驶可用性真值。该文字口径问题已有项目审查记录，本轮不重复修文件或新增 bug。

本轮完成了本地静态核对、论文与官方实现检索、方案去重和文档整理；没有训练、推理、导出、测试、测速、下载模型／数据或替换默认权重。未产生新运行踩坑，台账不更新。执行方案与性能结论仍须后续独立验证。

文档提交建议：`docs(research): 汇总视觉模型优化论文与落地优先级`。用途是记录本次调研及排除依据；当前分支 `pipeline_stop_test`，由用户决定是否提交，未暂存、提交、amend、推送、合并或切换分支。

[R1]: https://openaccess.thecvf.com/content/CVPR2023/html/Fang_DepGraph_Towards_Any_Structural_Pruning_CVPR_2023_paper.html
[R2]: https://openaccess.thecvf.com/content/CVPR2022/html/Yang_Focal_and_Global_Knowledge_Distillation_for_Detectors_CVPR_2022_paper.html
[R3]: https://openaccess.thecvf.com/content/CVPR2021/html/Guo_Distilling_Object_Detectors_via_Decoupled_Features_CVPR_2021_paper.html
[R4]: https://arxiv.org/abs/1911.08731
[R5]: https://openaccess.thecvf.com/content/CVPR2022/html/Ben-Baruch_Multi-Label_Classification_With_Partial_Annotations_Using_Class-Aware_Selective_Loss_CVPR_2022_paper.html
[R6]: https://yonsei-stl.github.io/MWSC/
[R7]: https://journal.lnpu.edu.cn/CN/10.12422/j.issn.1672-6952.2024.02.013
[R8]: https://openaccess.thecvf.com/content_iccv_2017/html/Liu_RankIQA_Learning_From_ICCV_2017_paper.html
[R9]: https://www.ecice06.com/CN/10.19678/j.issn.1000-3428.0063745
[R10]: https://arxiv.org/abs/2503.22375
[R11]: https://arxiv.org/abs/2108.05997
[R12]: https://www.jcad.cn/article/doi/10.3724/SP.J.1089.2021.18420
[E1]: https://docs.ultralytics.com/modes/export/
[E2]: https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/support-matrix.html
[E3]: https://developer.nvidia.com/cuda/gpus
