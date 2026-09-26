# 数据集说明

## 1. ACDC —— 视觉输入与天气标签的来源

**ACDC: The Adverse Conditions Dataset with Correspondences**
ETH Zurich, ICCV 2021
官网：https://acdc.vision.ee.ethz.ch/

| 项目 | 内容 |
|------|------|
| 版本 | v2（ICCV 2021 初版 + TPAMI 2025 扩展版） |
| 规模 | **8012 张** = 4006 张恶劣天气 + 4006 张对应正常天气参考图 |
| 天气子集 | fog / night / rain / snow，各约 1000 张，均匀分布 |
| 有标注的图像 | **2006 张**（train + val；test 标注按基准惯例不公开） |
| 原生标注 | 语义分割、**目标检测框**、全景分割、不确定性分割、跨图像对应关系 |
| 类别体系 | Cityscapes 19 类（可直接复用 Cityscapes 预训练权重） |
| 分辨率 | 1920 × 1080 |
| 许可 | **仅限研究用途，禁止再分发**；需注册申请 |

> 以上数字依据官方包的元数据核实（`GET /api/packages`），不是转述二手资料。

### 1.1 本项目从 ACDC 取什么

- **图像**：全部任务的视觉输入（4006 张恶劣天气 + 4006 张正常天气参考图）
- **语义分割标注**：监督分割头
- **目标检测框**：监督检测头（2006 张）
- **天气子集标签**：监督路况分类头，也用于分层采样与分层指标
- **正常天气参考图**：作为「同场景、不同天气」的配对样本，
  用于跨天气泛化实验 —— 这是 ACDC 相对其他恶劣天气数据集最独特的地方，
  也是「模型是真的理解了场景，还是在拟合天气纹理」这个问题的实验抓手

### 1.2 本项目从 ACDC 取不到什么

这一节比上一节更重要，因为它决定了项目的额外工作量：

- ❌ **没有距离标注** → 距离头必须用外部数据集监督（**这是唯一必须外部的任务**）
- ❌ **没有「前车 / 来车」的方向区分** → 检测框只有类别标签，方向要自己推导
- ❌ **没有接管/决策相关标注** → 接管边界标签必须自己构造（见 `handover_policy.md`）
- ❌ **没有积水深度、路面积雪覆盖率和雾能见距离的逐图数值标签** → 不能直接监督这些物理量

> ⚠️ **勘误记录（2026-09-21）**
> 本文档早期版本写着「没有车辆检测框 → 检测头必须用外部数据集监督」，**这是错的**。
> 该结论基于 ICCV 2021 的 v1 版本，而 ACDC 在 TPAMI 2025 扩展版中新增了
> `gt_detection_trainval.zip`（2006 张图的检测框标注）。
>
> 影响：检测任务不必再跨域，域差异问题从「检测 + 距离」缩小到只剩「距离」。
> 相应地，`configs/model/perception_multitask.yaml` 中检测头的
> `label_source` 应从 `external` 改为 `acdc`。

> ACDC 官方提供 train/val/test 图像划分；四类条件由 fog/night/rain/snow 子集确定。
> 独立天气小模型只用 train 训练、val 选优，test 保留作最终评估。
> 类别来源见 [ACDC 原论文](https://openaccess.thecvf.com/content/ICCV2021/papers/Sakaridis_ACDC_The_Adverse_Conditions_Dataset_With_Correspondences_for_Semantic_Driving_ICCV_2021_paper.pdf)。

### 1.3 数据划分注意事项

ACDC 的图像来自连续采集的 session，**同一 session 的相邻帧高度相似**。

如果按图像随机划分，同一场景的相邻帧会同时出现在训练集和验证集里，
验证指标会显著虚高（这个虚高幅度可能达到十几个点），
而模型在真正的新场景上并没有那么好。

因此划分策略必须是 `by_scene`（见 `configs/data/acdc.yaml` 的 `split.strategy`），
且划分结果落盘到 `data/processed/acdc/split.json`，保证所有实验用同一套划分。

---

## 2. 距离标注的补充来源

> **只有距离这一个任务需要外部数据。** 检测框已由 ACDC v2 自身提供。

### 2.1 候选数据源对比

| 数据集 | 距离来源 | 3D 框 | 场景条件 | 许可 | 建议 |
|--------|----------|-------|----------|------|------|
| KITTI | LiDAR / 立体视觉 | ✅ | 白天、晴朗 | CC BY-NC-SA 3.0 | **首选**，标签质量最高 |
| nuScenes | LiDAR | ✅ | 含夜间与雨天 | CC BY-NC-SA 4.0 | 次选，数据量大（~300GB） |
| 自采行车记录仪 | 单目深度估计 | ❌ | 任意 | — | 置信度低，仅作补充 |

### 2.2 核心设计：视觉域与几何监督分离

本项目对距离任务采用如下设计：

```
输入图像  ←  ACDC（恶劣天气，真实场景分布）
检测框    ←  ACDC 自身（2006 张，与图像同域，无域差异）
距离监督  ←  KITTI（LiDAR / 3D 框真值，几何精确）
相机内参  ←  各自数据集的 calib（ACDC 的 P2 用于推理，KITTI 的 P2 用于标定）
```

即：**检测在 ACDC 域内完成，只有「距离」这一个标量需要跨域迁移。**

这比早期设计好很多：域差异被压缩到一个输出维度上，
而不是像原方案那样连「目标在哪」都要跨域对齐。

余下的代价：KITTI 的 3D 框与 ACDC 的 2D 框不是同一批目标，
距离标签本质上是「从 KITTI 学到的 框→距离 映射迁移到 ACDC 的框上」，
仍需要 2.3 的域适配手段。

### 2.3 域差异及其缓解

白天晴朗数据训练出的距离估计，直接用在夜间雨雾场景上会退化。
这是本项目的**已知局限**，处理方式：

1. **训练时施加恶劣天气增强**（`data/transforms/weather_aug.py`），
   缩小输入图像的视觉域差异
2. **评测时分天气子集报告距离误差**，如实呈现退化程度，
   而不是只给一个总体 MAE 把退化藏起来
3. 在报告中明确说明：本项目不声称在恶劣天气下达到与晴天同等的测距精度

---

## 3. 标签口径（必须统一，改口径是破坏性变更）

完整定义见 `docs/label_spec.md`。要点：

- **距离参考系**：自车后轴中心 → 目标框底边中心
- **单位**：米
- **有效范围**：2.0 – 80.0 m（超出范围直接丢弃，不做截断）
- **无标注帧**：标记 `valid=False` 并在损失中 mask，**绝不填 0**

---

## 4. 数据目录结构

```
data/
├── raw/                       # 原始下载数据（zip），只读，不修改
│   ├── acdc/                  #   ACDC 解压结果，保持官方原样结构
│   │   ├── License.pdf
│   │   ├── README.md          #   官方说明，是目录约定的权威来源
│   │   ├── rgb_anon/{condition}/{split}/{sequence}/*_rgb_anon.png
│   │   ├── gt/{condition}/{split}/{sequence}/*_gt_*.png
│   │   ├── gt_detection/      #   COCO JSON，见 label_spec.md 4.4
│   │   └── gt_panoptic/
│   └── distance/              #   KITTI 三个 zip
├── interim/                   # 中间产物（解析后但未最终成型的）
├── processed/                 # 训练直接读取的数据
│   ├── acdc/
│   │   ├── split.json         # 按 scene 的划分，固定不变
│   │   └── direction_labels.jsonl  # 序列时序推导的方向标签，见 label_spec.md 4.1
│   ├── distance/
│   │   └── labels.jsonl       # (frame_id, box, distance_m, source, valid)
│   ├── handover/
│   │   └── labels.json        # 构造的接管边界标签
│   └── instruction/           # Stage 2 的指令数据集
│       ├── train.jsonl
│       └── val.jsonl
└── external/                  # 第三方数据集解压后原样存放
    └── kitti/
        ├── training/{image_2,label_2,calib}/
        └── testing/{image_2,calib}/
```

### 4.1 ACDC 实际目录约定（已核实的实测结果）

文件命名遵循 `{root}/{type}/{condition}/{split}/{sequence}/{sequence}_frame_{frame:0>6}_{type}{ext}`。

- `type`：`rgb_anon`（匿名化 RGB）或 `gt`（标注）
- `condition`：`fog` / `night` / `rain` / `snow`
- 每张图有 **5 个 gt 变体**：`labelIds`（Cityscapes ID）、`labelTrainIds`（trainID）、
  `labelColor`（可视化的彩色图）、`invIds`（无效掩码，无效=1）、`invGray`（无效=255）

**实测划分与数量：**

| condition | train | train_ref | val | val_ref | test | test_ref | 小计 |
|-----------|-------|-----------|-----|---------|------|----------|------|
| fog | 400 | 400 | 100 | 100 | 500 | 500 | 2000 |
| night | 400 | 400 | **106** | 106 | 500 | 500 | 2012 |
| rain | 400 | 400 | 100 | 100 | 500 | 500 | 2000 |
| snow | 400 | 400 | 100 | 100 | 500 | 500 | 2000 |
| **合计** | 1600 | 1600 | 406 | 406 | 2000 | 2000 | **8012** |

恶劣天气图 4006 张（1600+406+2000），正常天气参考图 4006 张。
night 的 val 比其他三类多 6 张，这是官方就这样，不是下载问题。

**有语义标注的是 2006 张**（train 1600 + val 406）—— test 标注按基准惯例不公开。

> 注意：解压后的结构保持官方原样，**不做重排**。
> 15.6 GB 的文件搬来搬去风险高于收益，且保持原样才能和官方文档/其他论文对得上。
> 代码侧按上述约定解析即可（`data/datasets/acdc_seg.py`）。

`data/` 整个目录已在 `.gitignore` 中，原因见 `LICENSE` 末尾的说明：
数据集许可不允许再分发。

---

## 5. 数据获取步骤

### 5.1 ACDC（需注册申请）

**ACDC 没有免注册的公开下载途径。** 实测确认：官网是一个完整的注册门户
（`/register` → `/agreeToLicense` → `/requestPackage` → 管理员审批），
所有可能的静态文件路径都返回同一个 SPA 首页，不存在匿名直链。

> ⚠️ **缩写撞名警告。** "ACDC" 至少对应三个互不相关的数据集：
> 1. **Adverse Conditions Dataset with Correspondences**（ETH，本项目用的这个）
> 2. **Automated Cardiac Diagnosis Challenge**（心脏 MRI，Kaggle / HuggingFace 上满地都是，可自由下载）
> 3. **Canadian Adverse Driving Conditions**（冬季驾驶，CC BY-NC 4.0）
>
> 搜索"ACDC 数据集下载"时命中的绝大多数是第 2 个。这一点已实测：
> Kaggle 搜 acdc 的 20 条结果、HuggingFace 的 29 条结果，**全部是心脏 MRI**。

**获取流程：**

1. 在 https://acdc.vision.ee.ethz.ch/register 注册并完成邮箱确认
2. 登录后同意使用条款
3. 在 https://acdc.vision.ee.ethz.ch/packages 申请下列包
   （`personalData` 全为 `false`，均为匿名化图像，理由填学术研究用途即可）

| 包名 | 大小 | 内容 | md5 |
|------|------|------|-----|
| `rgb_anon_trainvaltest.zip` | 15.6 GB | 匿名化图像：4006 恶劣天气 + 4006 正常天气 | `3350587a08502b4dfee47750bfd2a052` |
| `gt_trainval.zip` | 127 MB | 语义分割 + 不确定性感知分割 | `54cf06e3f2d8a8c5d297dc55aa107cee` |
| `gt_detection_trainval.zip` | 4 MB | 目标检测框 | `32598aacfe0f3c5138262849be8f35f3` |
| `gt_panoptic_trainval.zip` | 42 MB | 全景分割 | `bd2a6da68f0aba22cb6ae0536aab1a22` |

md5 取自官方 `GET /api/packages` 接口，下载后必须校验。

4. 审批通过后下载，放入 `data/raw/acdc/`，由 `scripts/download_acdc.py` 校验并解压

> 认证方式：官网 API 用 `Authorization: Bearer <token>`，
> token 存在浏览器 localStorage 的 `user` 键下。脚本据此实现，不需要账号密码。

### 5.2 KITTI（免注册直链）

KITTI 目标检测基准的下载是 **AWS S3 直链，不需要注册**：

| 文件 | 大小 | 用途 |
|------|------|------|
| `data_object_image_2.zip` | 12.0 GB | 左目彩色图像，7481 张训练帧 |
| `data_object_label_2.zip` | 5.3 MB | 2D/3D 框标注（距离标签的来源） |
| `data_object_calib.zip` | 25.6 MB | 相机内参 P0–P3、R0_rect |
| `data_object_velodyne.zip` | 27.4 GB | LiDAR 点云，**本项目不需要** |

```bash
BASE=https://s3.eu-central-1.amazonaws.com/avg-kitti
cd data/raw/distance
for f in data_object_image_2.zip data_object_label_2.zip data_object_calib.zip; do
  curl -L -C - -O "$BASE/$f"
done
cd "e:/Car_Smart_Assist_ACDC"
unzip -q 'data/raw/distance/data_object_*.zip' -d data/external/kitti
```

**已下载并验证**：7481 训练图 / 7481 标注 / 7481 标定 / 7518 测试图，
与官方基准完全一致；抽查 800 帧，3036 个 Car 目标全部带有效 3D 位置。

**不下载 velodyne 的理由**：距离可以直接从 `label_2` 的 3D 位置取
（第 13 字段即 z 值 = 距离），不需要点云。省下 27.4 GB。

### 5.3 预处理

```bash
python scripts/prepare_acdc.py          --config configs/data/acdc.yaml
python scripts/build_distance_labels.py --config configs/data/distance_supplement.yaml
python scripts/analyze_dataset.py
```

**最后一步不要跳过。** 直接开训然后花几天调参去追一个本可以五分钟发现的数据问题，
是这类项目最常见的时间浪费。
