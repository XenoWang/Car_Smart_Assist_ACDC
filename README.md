# Car Smart Assist

这个项目用单目相机识别复杂路况，给出能见度、天气属性、目标框和风险提醒。
目前能跑的部分有数据清洗、能见度模型、独立多标签天气模型、YOLO 检测和规则提示。

### 清单、设备及续训修复（2026-10-21验证）

- 默认能见度训练／评估和pipeline自动选图接入ACDC清洗清单，只排除`invalid`，保留`suspect`和原始文件。清单缺失时明确报错，提示先进行数据清洗，并给出项目根目录下的命令：

  ```powershell
  .\.venv\Scripts\python.exe -X utf8 scripts\clean_data.py --only acdc --corrupt-images-only
  ```

- 显式`--device`覆盖本次构建的能见度、天气和YOLO模型；不传时保留各自配置。YOLO直接使用已解析的逻辑设备，避免CPU调用改写CUDA可见设备后影响后续GPU调用。
- AE追加训练从恢复的学习率连续衰减，零或异常学习率会停止并提示另定微调方案；未完成的同配置追加段恢复时沿用原定终点，不重复增加extra预算。

本轮最终全量674项通过、0跳过；三类天气后端的CPU／CUDA共六个真实设备组合、三条小型AE续训轨迹及CUDA可见性守卫均通过。测试准备曾发生文件写入故障，恢复后采用备份、临时文件与写后读回核对。原权重和历史报告保留，未正式重训或重新校准；完整测试过程见本地PRD第17节。

### 检测选模修复（KB-P02，2026-10-20验证）

训练选best的验证尺寸、NMS IoU和最多框数现读取同一份`inference`配置；当前是960、0.5、100，训练尺寸与增强不变。联合／非联合检测都使用项目的AP后处理，验证数据也按同一规则对齐尺寸。39项定向、64项检测模块及620项全回归均通过，无失败或跳过。新增训练路径用例使用合成数据和mock；本轮未正式重训或重新校准，旧模型成绩不因此变化。

按已确认的兼容策略，旧目录缺少或不匹配选模记录时会拒绝续训，避免旧best分数参与新口径比较。需要沿用旧权重时，可用`train_detection.py --name 新目录名`从配置checkpoint对应的旧best微调；新目录的优化器和选优分数重新开始。匹配新口径的记录仍默认续训。`--fresh`仍表示从配置的初始模型重开，与旧best微调不同。

### RQ-01／RQ-02已完成结果

坏帧隔离与检测后处理已完成验证：79项定向、595项回归及5项校准发布专项通过；默认YOLO的313张校准、1394张留出评估，以及9帧真实pipeline与两条CLI检查均完成。坏输入保留原位置、记录不可用原因并请求接管，好图继续处理。

检测现按“原始最高类别 → 类阈值资格 → NMS → 最终数量限制”处理。新版校准已发布到原报告目录下的`best_class_threshold_v1/`子目录，默认检测可用，旧校准和权重保留。其他权重仍需匹配自己的新版校准；缺失或身份不符时，检测分支明确不可用，pipeline保守请求接管，不静默换模型或套用旧阈值。

本次没有重训或替换模型。当前默认YOLO留出结果：ACDC mAP50／50–95为26.91%／14.78%，L&F为71.48%／41.43%（仅标注ROI）；道路杂物精确计数P／R／F1为82.36%／59.57%／69.13%。旧操作点来自曲线插值，不把这点分数变化当作模型提升。校准与test各遍历两次，本机RTX5070合计约1分41秒，不能当作逐帧部署速度。下文新训练候选的分数属于不同权重，默认尚未替换。后续测试仍须先请示。

## 环境

Python 3.12，用项目里的 `.venv`。

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements/requirements-torch.txt
.venv\Scripts\python.exe -m pip install -r requirements/requirements.txt
.venv\Scripts\python.exe -m pip install -r requirements/requirements-detection.txt
.venv\Scripts\python.exe -m pip install -e . --no-deps
```

测试工具：

```powershell
.venv\Scripts\python.exe -m pip install -r requirements/requirements-dev.txt
```

查看训练设备：

```powershell
.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## 数据目录

| 数据 | 位置 |
|---|---|
| ACDC | `data/raw/acdc/` |
| KITTI | `data/external/kitti/` |
| Pixel Accurate | `data/external/pixel_accurate_depth_benchmark/pixel_accurate_depth_benchmark/` |
| Lost & Found | `data/external/lost_and_found/` |

数据结构与获取入口见 [数据说明](docs/dataset.md)。

## 清洗

```powershell
.venv\Scripts\python.exe scripts\clean_data.py --dataset acdc
.venv\Scripts\python.exe scripts\clean_data.py --dataset pixel_accurate_benchmark
.venv\Scripts\python.exe scripts\clean_data.py --config configs/data/cleaning_lost_and_found.yaml
```

清洗会列出损坏文件，原文件不会被修改或删除。

## 能见度与检测训练

能见度：

```powershell
.venv\Scripts\python.exe scripts\train_visibility.py
```

检测：

```powershell
.venv\Scripts\python.exe scripts\prepare_detection.py --config configs/data/acdc_lost_and_found_detection.yaml
.venv\Scripts\python.exe scripts\train_detection.py
```

## 独立三模型重训结果（2026-10-18）

三模型独立重训与最终评估已完成：天气／AE／YOLO分别在74／57／65轮早停，best为34／49／35轮。新天气macro F1为90.82%，低于A85的92.55%，默认仍保留A85；新YOLO的ACDC与L&F测试mAP均高于旧联合模型，默认权重暂未替换。AE测试／训练重建误差比2.24，仍有重建泛化差距。新权重正常／雾／雨／雪／夜五张真实帧与黑帧pipeline检查通过；这不是实车或泛化验证。

执行顺序为预检与缓存 → 天气效率检查 → 三模型串行训练 → 校准与最终评估。以下为本轮候选与旧模型的结果，默认模型保持不变。

| 测试指标（%） | 本轮候选 | 旧模型 |
|---|---:|---:|
| 天气macro F1 | 90.82 | 92.55（A85） |
| ACDC检测mAP50／mAP50–95 | 31.55／18.85 | 26.91／14.78（旧联合） |
| Lost & Found检测mAP50／mAP50–95 | 76.00／46.73 | 71.48／41.43（旧联合） |
| 道路杂物F1 | 72.88 | 69.12（旧联合） |

天气分数来自2158图的项目开发测试；检测两次使用同一1394图划分与评估参数，各自校准操作点。L&F只统计已标注ROI，不是整图误报率。AE新旧划分不同，重建误差不能直接横比。新联合模型ACDC mAP50仍低于历史八类基线33.02%，不能宣布旧类已全面保持。


三个新配置推荐入口：

| 阶段 | 推荐入口 |
| --- | --- |
| weather | `configs/model/weather_attention_retrain_300.yaml` |
| AE / 能见度 | `configs/model/visibility_retrain_300.yaml` |
| YOLO | `configs/model/yolo_retrain_300.yaml` |

各阶段 checkpoint 落在独立 `artifacts/checkpoints` 或 `artifacts/runs`，报告集中写入 `artifacts/reports/retrain_300_20261017/`。训练方案：seed 42，各模型从初始化起最多 300 轮早停——weather 用本地 ImageNet A13/ASL，输入 384x672，schedule 500，ES 40、delta 1e-4；AE 随机初始化，ES 8、delta 1e-5；YOLO 用本地 COCO 骨干加新九类头，patience 30。原默认配置与只读对照保持不改。


## 天气模块

### 当前默认

当前默认已恢复为 **A组第85轮 best**：路由 `configs/model/weather_default.yaml` 指向 `configs/model/weather_attention_a.yaml`，读取 `artifacts/checkpoints/weather_attention_ablation/reduced_tuning/best.pt`。SWAD追加实验已从300轮last继续到累计900轮，新平均覆盖301—526轮共40,228次更新，验证评分93.30%，仍低于A85的94.45%和保留单点52的94.56%，因此按约定回退。SWAD没有进入默认pipeline；显式配置与权重仍可覆盖默认行为。

训练配方为 A 组：`fine_tune_from=13`、384×672 输入、MobileNetV3 三层特征加属性注意力、ASL 损失，固定 seed42。A组完成300轮，SWAD首轮从同一ImageNet起点训练300轮，再从其last追加600轮。追加阶段保留参数、优化器和随机状态，学习率连续衰减；新增轮次没有超过原第52轮单点best。CUDA未强制逐位确定性，原A85与单点52之间的分数变化不能归因于SWAD。

### 训练与推理

当前A组配方预检：

```powershell
.venv\Scripts\python.exe -X utf8 scripts\train_weather_attention.py --config configs/model/weather_attention_a.yaml --preflight
```

续训（默认恢复 last）：

```powershell
.venv\Scripts\python.exe -u -X utf8 scripts\train_weather_attention.py --config configs/model/weather_attention_a.yaml
```

已完成300轮时，重复该命令只重导出评估，不追加训练。需要保留历史重开时加 `--fresh`。

推理：

```powershell
.venv\Scripts\python.exe scripts\run_pipeline.py --image "图片路径" --json
```

如需复核保留的SWAD实验导出，可显式指定；它仍是第52轮单点副本：

```powershell
.venv\Scripts\python.exe scripts\run_pipeline.py --image "图片路径" --json --weather-config configs/model/weather_attention_swad.yaml
```

SWAD实验中的单点 `single_best.pt` 和平均 `swad.pt` 各自校准；实验目录的 `best.pt` 保存验证评分胜出版本的副本。它与默认路由的选择分开记录，测试集不参与挑权重。

### 从300轮继续追加600轮

```powershell
.venv\Scripts\python.exe -u -X utf8 scripts\extend_weather_swad.py
```

配置在 `configs/model/weather_attention_swad_extend_600.yaml`。这条命令从保留的第300轮last继续到累计900轮，延续参数、优化器和随机序列，只重开SWAD平均区间；学习率从源末轮值平滑衰减。已有追加检查点时继续本阶段，不重放前300轮。

本次900轮已跑完，无需重复执行。上述命令用于启动或恢复未完成的同一追加阶段，累计目标固定900，不会在每次执行时再加600轮。追加阶段SWAD和完整last等7个文件已复制到 `artifacts/checkpoints/weather_attention_swad_readonly/extend_600/eb91299566d7/` 并设置只读，原300轮也有独立只读副本。

训练完整结束后才统一比较：只有SWAD验证综合评分同时胜过旧A组和同轮单点时，才选择SWAD作为默认；否则恢复 `configs/model/weather_attention_a.yaml` 对应的A组第85轮，并将本轮文件复制到只读对照目录。测试成绩用于报告，不决定默认模型。入口会在切换前后实际运行pipeline并检查黑帧阻断。

### 与旧模型对比

平均列为四属性 macro F1，其余列为各属性 F1，单位均为百分比。四个属性可共存，`night` 表示夜间光照，`snow` 包括降雪和积雪。

| 模型 | 平均 | 雨 | 雾 | 雪 | 夜 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 旧 CNN | 76.37 | 52.21 | 76.92 | 83.33 | 93.01 |
| 金字塔 ASL | 77.70 | 42.86 | 75.79 | 95.26 | 96.89 |
| 147 基线 | 90.26 | 89.85 | 84.68 | 95.04 | 91.49 |
| A85（当前默认） | 92.55 | 89.62 | 87.82 | 97.95 | 94.82 |
| 本轮早停候选，best34 | 90.82 | 90.66 | 85.42 | 96.92 | 90.26 |
| 单点52（保留对照） | 92.69 | 94.25 | 88.02 | 97.71 | 90.77 |
| 原300轮 SWAD | 90.85 | 89.94 | 84.63 | 97.95 | 90.87 |
| 追加阶段 SWAD（301—526） | 90.50 | 86.53 | 84.47 | 96.74 | 94.27 |

### 废弃与保留

旧雨天残差增强路线已由当前方案优化替代并废弃。已删除相关源码、配置、专用测试及多余旧权重，共19个文件，约21.03 MiB。旧 CNN、ASL 的 `best`、147 基线和全部天气对照保留；历史复现入口与共享数据工具保留兼容，原始数据不变。清理前30帧及清理后默认10帧验证通过，旧CNN/ASL仍可读取；此次回退前后各6帧实际pipeline验证通过，黑帧仍阻断感知。训练后的全项目453项回归通过，3份归档权重各5张真实图的输出与原文件一致。

### 评估边界

继续增加轮次没有解决过拟合：训练ASL从第300轮2.71e-8下降到第900轮2.08e-11，来源等权验证BCE反而由0.28878升到0.46509。新增阶段最好的单点也只有第356轮94.07%的验证评分，原第52轮best保持不变。

追加SWAD雨漏判66/278，受控白天轻雨／白天强雨／夜间轻雨／夜间强雨检出为0/10、1/10、0/10、0/10。恢复的A85受控雨同样仅1/40，不能理解为雨问题已解决；保留的单点52可检出21/40，但夜间F1比A85低4.06个百分点。按约定回退A85，并不代表其所有分项都更好。

单点52的平均 F1 比A85高0.13个百分点，单次试验不能证明稳定收益。旧 CNN 见过部分开发图，测试集也已用于先前分析，因此这不是新的官方未见测试。测试集为2158图，全部数据9612图，train/val/cal/test 为5696/796/962/2158，组、配对与重复图绑定，seed42；fog/rain/snow/night 独立且可共存，snow 含降雪与积雪，没有物理雨量或水深标签。完整分组结果和曲线保留在本地 `model_report.md`，这里展示当前概况。

## 推理与回归

```powershell
.venv\Scripts\python.exe scripts\run_pipeline.py --image "图片路径" --json
.venv\Scripts\python.exe scripts\run_pipeline.py --json --log-level INFO
.venv\Scripts\python.exe -X utf8 -m pytest tests -q --basetemp artifacts/pytest-tmp
```

推理结果里有 `visibility`、`weather`、`weather_warning`、`perception`、`advisory` 和 `skipped`。
看 `skipped` 能知道这一帧哪些模块没跑；距离和方向目前还没有预测结果。

## 配置

| 功能 | 配置 |
|---|---|
| 默认天气路由 | `configs/model/weather_default.yaml` |
| 当前默认A组训练配方 | `configs/model/weather_attention_a.yaml` |
| 首轮SWAD只读对照配方 | `configs/model/weather_attention_swad.yaml` |
| SWAD追加600轮 | `configs/model/weather_attention_swad_extend_600.yaml` |
| 147 轮基线与注意力结构 | `configs/model/weather_attention.yaml` |
| 固定种子单项对照 | `configs/model/weather_attention_ablation.yaml` |
| 历史 CNN / ASL 对照 | `configs/model/weather_classifier.yaml` / `configs/model/weather_pyramid.yaml` |
| 能见度 | `configs/model/visibility.yaml` |
| 联合 YOLO / 旧八类 YOLO | `configs/model/yolo_detection.yaml` / `configs/model/yolo_acdc_baseline.yaml` |
| 独立300轮重训 | `configs/model/weather_attention_retrain_300.yaml` / `visibility_retrain_300.yaml` / `yolo_retrain_300.yaml` |
| 风险与接管规则 | `configs/model/advisory_llm.yaml` |

## 文档

- [架构示意图](docs/architecture.md)
- [数据说明](docs/dataset.md)
- [标签与接口](docs/label_spec.md)
- [风险与接管输出](docs/handover_policy.md)
- [变更记录](CHANGELOG.md)

代码许可见 [LICENSE](LICENSE)。


## 离线审计与性能量测（RQ06／07／09，已验证）

这三个工具读取已有结果或现有模型，把误例、证据缺口和运行成本单独整理出来。不会自动重标图片，也不会训练或替换模型。输出目录必须是新目录。

```powershell
# 天气：按模型、来源、录制组、场景和属性统计，生成待人工复核表
.\.venv\Scripts\python.exe -X utf8 scripts\audit_weather_predictions.py --input artifacts\reports\weather_attention\test_predictions.csv --output artifacts\reports\weather_audit_v003

# 接管：读取之前保存的pipeline JSON，统计缺距离／方向及风险触发代码
.\.venv\Scripts\python.exe -X utf8 scripts\summarize_pipeline_evidence.py --input 已保存的pipeline结果.json --output artifacts\reports\evidence_audit_v003

# 性能：显式指定图片，可重复--image；auto沿现有入口优先使用可用GPU
.\.venv\Scripts\python.exe -X utf8 scripts\benchmark_pipeline.py --image 图片路径.png --device auto --batch-size 1 --warmup 1 --repeats 5 --output artifacts\reports\benchmark_v003
```

天气的`known=0`不算负样本，人工标签栏留空；接管原因按触发次数和涉及帧数分别统计，原因之间可以重叠。已有test只用于开发审计，不重新包装成未见测试。

量测区分模型构建、首次调用和预热后的整批耗时，并比较路径输入与预解码数组输入。内部`timings_ms`只是诊断值，不能当精确的CUDA分阶段耗时；细分缩放、数据搬运和NMS暂未独立量测。桌面结果不代表车载30／60FPS。本轮已通过806项回归及真实审计／量测；旧674项记录保留，不与本轮结果混称。


### 本轮能见度安全边界调整（已验证）

门控阈值出现NaN、无穷大或档位顺序冲突时，现在会在构建时明确报错，不再让错误配置绕过阻断。合法默认阈值保持原样。

AE新检查点格式为3，续训前核对训练目录中的best权重摘要、数据划分和历史最优指标；缺失或不匹配就停止，不用当前轮补造历史best。旧格式2缺少身份绑定，不能直接完整续训，但原权重仍可推理／评估。请保留旧文件；`--fresh`表示重新初始化实验，沿旧权重微调需另定方案。新工具和这两项修复已通过本轮验证；以后新一轮测试仍先请示。


### 本轮结果概况（2026-10-22）

全量806项通过，0失败／错误／跳过；真实审计和量测12个子命令均成功。CPU／CUDA:0各batch1／2，5张图片重复运行，各50条推理记录通过设备与输出检查；这是分支可用性验证，不是识别准确率100%。原数据、模型和校准的29项保护哈希均未变化。

按既有2158张开发图片的导出标签直接汇总：

| 天气模型 | 属性行micro F1 | 雨F1 |
|---|---:|---:|
| 当前默认A85 | 91.39% | 89.62% |
| 旧500轮注意力对照 | 88.94% | 89.85% |
| 原四属性CNN | 76.11% | 52.21% |
| 金字塔ASL | 79.73% | 42.86% |

这次没有训练，以上是旧CSV复核，不替代来源平衡选模成绩。A85在ACDC雨正样本中检出232/238，但PAB只检出1/40，后者白天也大量漏判；雨雾标签及来源差异仍需人工审计，不能只看总体分数。

本机RTX5070稳态文件输入约14.07—14.31FPS，预解码数组约22.94—23.95FPS；CPU分别约8.62—8.89与11.55—11.86FPS。样本仅5张，含三模型及规则，不构成车载30／60FPS保证。
