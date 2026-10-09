# Car Smart Assist — 复杂路况智能驾驶辅助

面向**恶劣天气与复杂路况**的两阶段驾驶辅助系统：判断辅助驾驶的能力边界，
并在需要时用自然语言告诉司机该怎么调整。

基于 [ACDC](https://acdc.vision.ee.ethz.ch/)（Adverse Conditions Dataset with Correspondences）
的雾 / 夜 / 雨 / 雪四类场景构建。

当前可运行的推理链路是：能见度门控 → 天气多属性识别与 YOLO 常规目标检测 → 规则与模板建议。
模型按配置中的现有权重加载；检测未运行和成功运行但未检出目标分别记录。
当前检测覆盖八类交通目标与新增道路杂物候选；空框仍不能排除所有未知障碍。

> **实现状态：部分功能可运行，完整两阶段系统仍在开发中。**
> 能见度门控、独立天气模型、YOLO 框检测、风险/接管规则和模板提示已实现；
> 距离、方向、道路分割、基于行驶路径的完整异常障碍判断及 LLM/VLM 后端尚未实现。下面的架构图表示目标设计，
> 不代表当前所有模块都已运行。详细状态见 [`TECH_STACK.md`](TECH_STACK.md)；
> 开发计划见 [`docs/roadmap.md`](docs/roadmap.md)。

---

## 这个项目做什么

给定一帧来自行车记录仪的画面，系统输出两样东西：

1. **接管边界的判断** —— 当前条件下辅助驾驶还能不能继续，是否该把控制权交还司机
2. **给司机的提示** —— 路况如何、前车与来车多远、需不需要做对应调整

```
摄像机画面
    │
    ▼
┌─────────────────────────────────────────────┐
│ Stage 1 · 多任务感知                          │
│                                             │
│   共享骨干 (SegFormer-MiT)                    │
│     ├── 语义分割      → 可行驶区域、场景结构      │
│     ├── 接管边界      → 三档分级 (0/1/2)        │
│     └── 检测 + 距离   → 前车/来车 + 距离(米)     │
└────────────────────┬────────────────────────┘
                     ▼
          PerceptionResult（结构化契约）
                     │
        ┌────────────┴────────────┐
        ▼                         ▼
  规则策略引擎                提示词渲染
  （决定要不要接管）           （组织上下文）
        │                         │
        │                         ▼
        │                   Stage 2 · 语言生成
        │                   (3B VLM, 4-bit QLoRA)
        └────────────┬────────────┘
                     ▼
          一致性校验 + 降级兜底
                     ▼
        「夜间有雾，前方约 30 米有车，建议减速并准备接管」
```

天气与光照使用共享骨干之外的独立模型，按 `fog/rain/snow/night` 分别判断；Stage 1 不重复训练天气头。
分类规范与数据映射见 [天气属性标签](docs/label_spec.md#2-天气现象与光照属性标签)。

### 两个核心设计取舍

**决策与表达分离。** 要不要接管由规则引擎决定，语言模型只负责把结论说清楚。
生成模型对同一场景的措辞可以有波动，但结论不能波动 —— 驾驶场景没有这个容错空间。
当两者冲突时，以规则的保守结论为准。

**感知结果结构化。** 每条最终提示都能追溯到具体的结构化证据
（哪一档边界、哪个目标的多少米），而不是一个无法解释的文本。
这既是为了可调试，也是为了能算出真实的安全指标（召回率、漏报率），
而不是只能算 BLEU。

详细的设计理由与备选方案对比见 [`docs/adr/0001`](docs/adr/0001-two-stage-architecture.md)。

---

## 硬件与环境

本项目在以下环境上开发与验证：

| 项目 | 值 |
|------|-----|
| GPU | NVIDIA GeForce RTX 5070（Blackwell, **sm_120**, **12 GB**） |
| 驱动 | 610.74 (CUDA UMD 13.3) |
| CUDA Toolkit | 12.9.41 |
| Python | 3.12.10 |
| PyTorch | **2.8.0+cu129** |
| 操作系统 | Windows 11 |

> ⚠️ **RTX 5070 是 sm_120 架构，必须有 torch ≥ 2.8 的 cu128/cu129 版本才能原生支持。**
> 更低版本会退回 PTX JIT 编译，首次运行极慢。
> 安装后请用 `make check-env` 确认 `arch_list` 中包含 `sm_120`。

12GB 显存是贯穿全项目的硬约束，它直接决定了模型规格的选择：
骨干网取 MiT-B1 而非更大档位，Stage 2 必须走 4-bit 量化。
相关取舍记录在 [`docs/architecture.md`](docs/architecture.md#5-硬件约束及其影响)。

---

## 快速开始

### 1. 环境准备

```bash
# 创建虚拟环境（Python 3.12）
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS
```

### 2. 安装依赖

依赖**分两份**，因为国内镜像源上的 torch 是 CPU-only 构建，用镜像装会导致
`torch.cuda.is_available()` 返回 `False`，而版本号看起来完全正常，极难排查。

```bash
# 第一步：torch 栈 —— 走 PyTorch 官方 CUDA 索引，不要加 -i 参数
pip install -r requirements/requirements-torch.txt

# 第二步：其余运行时依赖 —— 可以走国内镜像
pip install -r requirements/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 第三步：把本项目以可编辑模式装上（--no-deps，依赖已由上面两步装好）
pip install -e . --no-deps

# 开发工具（可选）
pip install -r requirements/requirements-dev.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**安装顺序不能颠倒。** 先装 torch 再装其余依赖。

### 3. 自检

```bash
make check-env
# 期望输出：
#   torch   : 2.8.0+cu129
#   cuda ok : True
#   arch    : ['sm_70', 'sm_75', 'sm_80', 'sm_86', 'sm_90', 'sm_100', 'sm_120']
```

如果 `cuda ok` 是 `False`，说明装成了 CPU 版 —— 卸载后按上面的第一步重装。

### 4. 数据准备

**ACDC** —— 没有免注册的下载途径，必须申请（详见 [docs/dataset.md](docs/dataset.md#51-acdc需注册申请)）：

1. 在 https://acdc.vision.ee.ethz.ch/register 注册并确认邮箱
2. 登录后同意使用条款
3. 在 https://acdc.vision.ee.ethz.ch/packages 申请
   `rgb_anon_trainvaltest.zip`（15.6 GB）、`gt_trainval.zip`、`gt_detection_trainval.zip`
4. 审批通过后放到 `data/raw/acdc/`，用脚本校验 md5 并解压

**KITTI** —— 免注册 AWS S3 直链：

```bash
BASE=https://s3.eu-central-1.amazonaws.com/avg-kitti
cd data/raw/distance
for f in data_object_image_2.zip data_object_label_2.zip data_object_calib.zip; do
  curl -L -C - -O "$BASE/$f"
done
cd "e:/Car_Smart_Assist_ACDC"
unzip -q 'data/raw/distance/data_object_*.zip' -d data/external/kitti
```

> KITTI 只用于补充**距离**标签 —— ACDC v2 自带检测框，
> 只有距离这一项是 ACDC 没有的。不要下 `data_object_velodyne.zip`（27.4 GB），
> 距离可直接从 `label_2` 的 3D 位置取，用不上点云。

**道路障碍数据 Lost & Found（已获取，2026-10-08）：**
匿名下载入口：[埃斯林根大学公开镜像](https://huggingface.co/datasets/iis-esslingen/LostAndFoundDataset)。
数据位于 `data/external/lost_and_found/`，保留 `leftImg8bit.zip`、`gtCoarse.zip`、
标签说明及官方 `leftImg8bit/{train,test}`、`gtCoarse/{train,test}` 目录。
实际有 2,239 组图像／标签（train 1,036、test 1,203），压缩包 SHA256、ZIP CRC 和所有 PNG
完整性检查通过。获取记录为 `download_manifest.json`；道路杂物候选已通过联合 YOLO 训练接入 pipeline。

Lost & Found 日常清洗统一使用已有入口，逻辑全部位于 `data/preprocessing.py`：

```powershell
.venv\Scripts\python.exe scripts\clean_data.py --config configs/data/cleaning_lost_and_found.yaml
```

该配置只完整解码 RGB 和 PNG 标注，损坏文件进入 `data/processed/manifests/lost_and_found.json`
的 `invalid` 列表；`retained` 列出保留图片，其他文件单独列为原样保留。
不按亮度、清晰度、尺寸、重复、标签内容或配对情况剔除文件；权限等未完成读取记录待复核。
原始文件和压缩包不删除、不改写。报告位于 `artifacts/reports/cleaning/lost_and_found/`。

**预处理：**

```bash
python scripts/prepare_acdc.py          --config configs/data/acdc.yaml
python scripts/build_distance_labels.py --config configs/data/distance_supplement.yaml
python scripts/analyze_dataset.py
```

**最后一步不要跳过。** 先看统计数据确认数据没问题，再开训。

### 5. 训练、测试与推理

#### 当前可训练模型

能见度门控与天气分类模型已有独立训练入口。当前开发工作区已准备 ACDC 数据、处理缓存及两类模型权重；
数据和模型权重不随代码仓库分发，新环境需要先按上文准备数据。

```powershell
# 能见度门控（ACDC 正常天气参考图，无监督重建）
.venv\Scripts\python.exe scripts\train_visibility.py

# 天气/光照多属性模型（ACDC 按序列切分；Pixel Accurate 按 scene 留出；ACDC test 留出）
.venv\Scripts\python.exe scripts\train_weather.py

# 天气模型评估：ACDC 官方 test
.venv\Scripts\python.exe scripts\evaluate_weather.py

# 开发验证集及 Pixel Accurate scene 4 验证
.venv\Scripts\python.exe scripts\evaluate_weather.py --dataset validation
.venv\Scripts\python.exe scripts\evaluate_weather.py --dataset pixel-accurate
```

两个入口默认续训已有 `last.pt`；天气模型使用 `--fresh` 从头训练，能见度训练也支持 `--fresh`。
天气属性模型权重写入 `artifacts/checkpoints/weather_attributes/`，能见度权重写入 `artifacts/checkpoints/visibility/`。
pipeline 会加载可用的能见度权重；天气配置和权重都存在时也会自动加载天气模型。

天气模型分别输出雾、雨、雪和夜间概率，配置阈值把这些属性独立转成提醒；例如夜雾可以同时输出 `fog` 与 `night`。
天气现象分组为 fog/rain/snow，光照分组为 night。没有属性达到阈值时表示“未检出这些属性”，不直接推断晴天或白天。

当前模型的 ACDC 官方 test 报告（2,000 张，2026-10-06 核对已有报告）如下；阈值属于推理配置，不是天气强度：

| 分组 | 属性 | 当前阈值 | Precision | Recall | F1 |
|------|------|---------:|----------:|-------:|---:|
| 天气现象 | 雾 fog | 0.50 | 97.1% | 85.8% | 91.1% |
| 天气现象 | 雨 rain | 0.77 | 91.6% | 87.0% | 89.2% |
| 天气现象 | 雪 snow | 0.68 | 88.1% | 90.4% | 89.2% |
| 光照属性 | 夜间 night | 0.78 | 99.8% | 98.0% | 98.9% |

多属性 exact-match accuracy 为 88.15%，macro-F1 为 92.11%。这些指标不能解释为积水深度或雨雾强度的准确率。
完整指标在本机 `artifacts/reports/weather_attributes/acdc_test_metrics.json`；详细分类口径见 [标签规范](docs/label_spec.md#2-天气现象与光照属性标签)。

天气提醒通过 `weather_warning` 单独返回；天气属性单独不触发接管。已确认 fog/rain/snow 且能见度同时为 DEGRADED 时，
可作为接管辅助条件；night 或不确定天气不触发该条件。能见度 BLIND 门控和其他必要感知缺失仍按原规则处理。

天气模型输出的反光、疑似湿润区域、亮白覆盖与低对比度是图像代理指标，不代表水深、摩擦力、
实际积雪面积或雾中可视距离。训练期间使用验证集选模；正式测试结果由 `evaluate_weather.py` 单独报告。
Pixel Accurate 训练使用 scene 1–3，scene 4 整组留作验证，避免同场景图像跨集合；数据含 clear、雾等级、雨强度及昼夜组合，
不含 snow。scene 4 雨类样本较少，跨场景雨类仍需更多验证。ACDC 官方 test 指标与 Pixel 验证指标分开报告。

#### YOLO 检测：数据准备、训练与四分评估

当前联合检测参数在 `configs/model/yolo_detection.yaml`，联合数据参数在
`configs/data/acdc_lost_and_found_detection.yaml`；原 ACDC 配置保留于 `configs/model/yolo_acdc_baseline.yaml`。
YOLO11n 使用 PyTorch 后端，`device: auto` 优先使用可用 CUDA。

```powershell
# 保留当前 CUDA PyTorch，在项目 .venv 添加检测依赖
.venv\Scripts\python.exe -m pip install --no-deps -r requirements\requirements-detection.txt
.venv\Scripts\python.exe scripts\prepare_detection.py
.venv\Scripts\python.exe scripts\prepare_detection.py --config configs/data/acdc_lost_and_found_detection.yaml
.venv\Scripts\python.exe scripts\prepare_detection.py --verify-only --verify-hashes
.venv\Scripts\python.exe scripts\train_detection.py
# 默认检测已有模型并续训；显式重新开始与另外两个模型统一为 --fresh
.venv\Scripts\python.exe scripts\train_detection.py --fresh
.venv\Scripts\python.exe scripts\evaluate_detection.py --split val
.venv\Scripts\python.exe scripts\evaluate_detection.py --split calibration
.venv\Scripts\python.exe scripts\evaluate_detection.py --split test
```

使用 ACDC train+val 合并检测标注，固定种子 42，按 GoPro 原始拍摄组分层四分；
GOPR 与 GP01/GP02 章节属于同一组。train 更新权重，val 早停和选择 best，
calibration 单独选择置信度阈值，test 使用锁定的权重与阈值评估。四分本身不能消除过拟合。
官方 2,000 张 test 没有公开检测真值，单独列为推理图片，不作为空目标负样本或本地 mAP 测试集。

实际导出 train/val/calibration/test 为 **1,053 / 201 / 226 / 191** 张。
335 张含 `iscrowd` 的图片按已确认方案列为待支持，原图和标注完整保留；损坏图 0 张。
有效小框、夜间和低对比度图片均保留。输出位于 `data/processed/acdc_detection_yolo/`，
清洗复用 `data/preprocessing.py` 完整性检查。默认硬链接节省磁盘，输出图片也应只读使用。

目标比例 70/10/10/10 受整组约束影响，实际约 63.0/12.0/13.5/11.4。
四集合都有八类目标和四类天气，但 val 雨天仅 6 张、test 雨天仅 9 张，
calibration 雪天仅 3 张；少量天气子集指标不能代表可靠泛化。
报告见 `artifacts/reports/detection_data/summary.md` 及 `artifacts/reports/detection/yolo11n_acdc/`。

2026-10-07 已在 RTX 5070 完成 30 轮基线训练。仅在 val 上比较后，推理尺寸选择 960
（训练仍为 640）；val mAP50 从 25.75% 提升到 30.51%。calibration 选出的置信度阈值为 0.2042。
本地独立 test 的 mAP50 / mAP50–95 为 **32.99% / 16.72%**，宏召回率为 **33.73%**。
汽车相对较好，小目标和少数类别漏检仍明显；这版是可运行基线，尚不能宣称可靠识别。
分项结果、速度测量范围和 pipeline 结果见 `artifacts/reports/detection/yolo11n_acdc/summary.md`。
本次权重的原始训练参数保存在运行目录 `args.yaml`；配置已为后续新的 AdamW 训练明确
`warmup_bias_lr: 0.0`，本轮基线未按该新增参数重训。

中断续训同时需要 `weights/last.pt` 与配对的 `weights/last_training_state.pt`。
后者保存完整精度模型、优化器、EMA、AMP 缩放器与调度器，避免依赖库的 FP16 优化器保存丢失精度。
默认优先查找运行目录的 last/best 权重，再查找配置 `checkpoint`；已有模型就复用，不会自动改回 COCO 预训练权重。
中断运行恢复原计划；新版完整检查点即使已完成也可恢复状态继续训练。已完成旧版权重没有配对状态时，
自动从已有 best 权重继续微调并重新建立优化器，日志明确区分 `resume`、`finetune` 与 `fresh`。
完成已有计划后，默认再训练配置 `epochs` 轮；也可用配置 `train.resume_extra_epochs` 指定追加轮数。
`--epochs` 覆盖轮数：有完整状态时未达到该上限就补足，已达到时将其视为追加轮数。
`--fresh` 忽略已有训练状态，从配置 `model` 重新开始；`--resume` 保留为显式要求已有模型的兼容参数，不能与 `--fresh` 同用。
重新开始或进入下一轮前，旧权重与训练记录保存在运行目录 `history/round_XXXX/`。
权重更新后应重新运行校准与测试，pipeline 不会误用与新权重不匹配的旧校准阈值。

**联合道路杂物检测（2026-10-08）：**
原八类 ID 保持 0–7，新增 `road_obstacle=8`，含义是标注杂物候选。
从原 ACDC 权重扩展时复制原八类分类输出行；联合回放 ACDC 与 Lost & Found，训练 20 轮后微调 8 轮。
联合 train/val/calibration/test 为 1,914/289/313/1,394；ACDC 原四分保持不变，
Lost & Found 官方 train 按地点分为 861/88/87，官方 test 1,203 张完整留出。

未标注类别／区域不提供背景负样本，正样本类别按互斥监督；真值 ROI 只用于训练和定义评估范围，
推理不读取真值。为保持 ROI 对齐，空间混合增强暂关闭，保留颜色／轻度模糊增强。
验证采用与实际推理一致的单类别 NMS，按来源等权选 best；校准集确定分类阈值。

最终测试：Lost & Found 已标注 ROI 的 mAP50/mAP50–95 为 **71.48%/41.43%**，
杂物 Precision/Recall/F1 为 **82.32%/59.57%/69.12%**。原八类 ACDC mAP50/mAP50–95 为
**26.91%/14.78%**，低于同口径旧基线 **33.02%/16.94%**；当前存在旧任务退化，不能宣称整体精度提升。
框图也出现车头误报、极小杂物漏检，尚未验证所有未知类型与恶劣天气中的杂物泛化。
详细报告见 `artifacts/reports/detection/yolo11n_acdc_laf_v2/summary.md`。
默认推理加载联合模型；设 `enabled_for_inference: false` 可使用保留的旧八类配置回退。

#### 测试和当前推理链路

```powershell
# 全量自动化测试；项目内临时目录可避开 Windows 默认临时目录权限问题
.venv\Scripts\python.exe -m pytest tests -q --basetemp=artifacts\pytest-tmp

# ACDC 样例图端到端演示（同时查看 perception 与 skipped）
.venv\Scripts\python.exe scripts\run_pipeline.py --json --log-level INFO
```

推理演示自动加载已配置的 YOLO best 权重及匹配的校准阈值。
检测返回原图坐标框、类别与置信度；距离为 `None`，方向为 `unknown`，不会编造测距结果。
`object_detection_classes` 明确类别覆盖范围；成功运行联合检测时
`road_obstacle_detection_available=true` 表示已运行杂物候选检测，不能理解为覆盖全部异常障碍。

#### 尚不能运行的完整训练

Stage 1 多任务感知模型（分割、距离、方向和接管边界）与 Stage 2 LLM/VLM 微调尚未实现。
`scripts/train_perception.py`、`scripts/train_advisory.py` 目前只是职责说明，完整的
`make train-perception`、`make train-advisory`、`make eval` 训练/评测链路尚不可用。
规则和模板已消费 YOLO 的真实框检测结果，缺失方向／距离时仍按现有保守规则请求接管。
常规目标与道路障碍的分支设计、数据需求见 `docs/road_obstacle_detection.md`。

---

## 目录结构

```
Car_Smart_Assist_ACDC/
├── README.md
├── pyproject.toml              # 工程元数据 + ruff/mypy/pytest/coverage 配置
├── Makefile                    # 项目生命周期的统一入口
├── requirements/               # 依赖，按「能否用镜像加速」拆分
│   ├── requirements.txt        #   运行时依赖（可用国内镜像）
│   ├── requirements-torch.txt  #   torch 栈（必须走官方 CUDA 索引）
│   └── requirements-dev.txt    #   开发工具（可用国内镜像）
│
├── configs/                    # 所有可调参数的单一来源
│   ├── default.yaml            #   全局基础配置，其他配置继承它
│   ├── data/                   #   acdc.yaml / distance_supplement.yaml
│   ├── model/                  #   perception_multitask.yaml / advisory_llm.yaml
│   ├── train/                  #   perception.yaml / advisory_sft.yaml
│   └── inference/default.yaml
│
├── data/                       # 数据集（不进版本库，见下方「数据许可」）
│   ├── raw/                    #   原始下载，只读
│   ├── interim/                #   中间产物
│   ├── processed/              #   训练直接读取
│   └── external/               #   第三方数据集
│
├── docs/
│   ├── architecture.md         # 系统架构、数据流、分层约束
│   ├── dataset.md              # 数据来源、获取方式、域差异问题
│   ├── handover_policy.md      # ⭐ 接管边界标签的定义与已知方法论问题
│   ├── label_spec.md           # ⭐ 标签口径的权威来源（距离口径在此）
│   ├── roadmap.md              # 里程碑与验收标准
│   └── adr/                    # 架构决策记录
│       └── 0001-two-stage-architecture.md
│
├── scripts/                    # 面向人的薄入口脚本（逻辑在 src/ 里）
│   ├── download_acdc.py        #   下载
│   ├── prepare_acdc.py         #   预处理
│   ├── build_distance_labels.py#   ⭐ 构造距离标签（ACDC 原生缺失）
│   ├── build_instruction_dataset.py  # 构造 Stage 2 指令数据
│   ├── analyze_dataset.py      #   统计与可视化
│   ├── train_perception.py     #   Stage 1 训练入口
│   ├── train_advisory.py       #   Stage 2 训练入口
│   ├── evaluate.py             #   统一评测入口
│   └── export_onnx.py          #   导出部署（可选）
│
├── src/car_smart_assist/       # 全部业务逻辑
│   ├── config/                 #   配置加载与 schema 校验
│   ├── data/                   #   数据集 / 变换 / 采样器 / 标签映射
│   ├── perception/             #   Stage 1：骨干网、任务头、损失、几何测距
│   ├── advisory/               #   Stage 2：策略、提示词、语言模型
│   ├── engine/                 #   训练引擎：循环、优化器、检查点、回调
│   ├── evaluation/             #   指标实现与报告生成
│   ├── inference/              #   推理管线与后处理
│   ├── utils/                  #   日志、种子、IO、注册表、可视化
│   └── cli/                    #   统一命令行入口
│
├── tests/                      # 单元与集成测试
├── notebooks/                  # 探索性分析（结论需固化成脚本）
└── artifacts/                  # 训练产物：权重、日志、报告（不进版本库）
```

**每个文件内部都有 docstring 说明它负责什么、依赖什么、被谁调用。**
这份结构不是随手分的：分层依赖方向是单向的（`cli` → `inference` → `perception`/`advisory`
→ `data`/`engine` → `config`/`utils`），约束写在
[`docs/architecture.md`](docs/architecture.md#3-分层与依赖方向)。

---

## 数据许可（重要）

本项目**代码**采用 MIT 许可，但**不包含任何数据集内容**：

- **ACDC** 仅限研究用途，**禁止再分发**
- **KITTI** 为 CC BY-NC-SA 3.0
- **nuScenes** 为 CC BY-NC-SA 4.0（禁止商业用途）

因此 `data/` 整个目录已在 `.gitignore` 中。
公开本仓库时只能说明「如何获取数据」，不要附带数据本身。

---

## 已知局限

主动列出来，因为它们是设计的一部分，而不是待修复的 bug：

- **接管边界不是真实接管行为数据。** 标签由可观测的感知量构造而成
  （口径见 [`docs/handover_policy.md`](docs/handover_policy.md)），
  与真实驾驶员的接管决策存在差异。这一构造方式存在规则蒸馏的循环性，
  缓解手段与残余风险在同一文档中说明。
- **单目测距在远处不可靠。** 有效范围标定为 2–80 m，且必须在报告里分桶呈现
  误差 —— 总体 MAE 会掩盖「近处准、远处崩」的事实。
- **恶劣天气下的测距存在域差异退化。** 距离监督来自 KITTI（以晴天为主），
  用于 ACDC 的雨雾雪场景时会有退化。本项目不声称在恶劣天气下达到晴天同等精度，
  并会分天气子集报告退化程度。
- **「前车 / 来车」的方向标签是推导出来的，不是标注的。** ACDC 的检测框和
  KITTI 的类别都不区分方向，必须由朝向角或车道几何推导（见
  [`docs/label_spec.md`](docs/label_spec.md#41-方向推导两个数据集都不给必须自己算)）。
  推导置信度低的目标会标为 unknown 并排除出方向损失 ——
  方向判断错误会导致完全相反的驾驶建议，因此宁可不对方向表态。
- **无时序建模。** 单帧推理无法获得相对速度，因此无法计算 TTC，
  策略层只能使用距离绝对值与保守先验。
- **不覆盖的场景**：车辆机械故障、传感器遮挡、非视觉信号（交警手势、施工指示）、
  道路结构损坏。这些超出单目视觉感知能力，系统不声称覆盖。

---

## 文档索引

| 文档 | 内容 |
|------|------|
| [architecture.md](docs/architecture.md) | 系统架构、数据流、分层依赖约束、硬件约束的影响 |
| [dataset.md](docs/dataset.md) | 数据集来源、获取步骤、域差异问题与缓解 |
| [handover_policy.md](docs/handover_policy.md) | **接管边界的定义、标签构造、已知方法论问题** |
| [label_spec.md](docs/label_spec.md) | **标签口径权威来源**（距离参考系定义在此） |
| [roadmap.md](docs/roadmap.md) | 里程碑、验收标准、明确不做的事 |
| [adr/](docs/adr/) | 架构决策记录（含备选方案对比与未采纳原因） |

---

## 许可

代码：[MIT](LICENSE)

数据集：各有其自身许可，见 [`LICENSE`](LICENSE) 末尾说明与上文「数据许可」一节。
