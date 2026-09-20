# 数据集说明

## 1. ACDC —— 视觉输入与天气标签的来源

**ACDC: The Adverse Conditions Dataset with Correspondences**
ETH Zurich, ICCV 2021
官网：https://acdc.vision.ee.ethz.ch/

| 项目 | 内容 |
|------|------|
| 规模 | 4006 张图像 |
| 天气子集 | fog / night / rain / snow |
| 原生标注 | 语义分割、panoptic、跨图像对应关系 |
| 分辨率 | 1920 × 1080 |
| 许可 | **仅限研究用途，禁止再分发** |

### 1.1 本项目从 ACDC 取什么

- **图像**：全部任务的视觉输入
- **语义分割标注**：监督分割头
- **天气子集标签**：监督路况分类头，也用于分层采样与分层指标

### 1.2 本项目从 ACDC 取不到什么

这一节比上一节更重要，因为它决定了项目的额外工作量：

- ❌ **没有车辆检测框** → 检测头必须用外部数据集监督
- ❌ **没有距离标注** → 距离头必须用外部数据集监督
- ❌ **没有接管/决策相关标注** → 接管边界标签必须自己构造（见 `handover_policy.md`）
- ❌ **没有官方 train/val/test 划分** → 必须自己按 scene 切分

### 1.3 数据划分注意事项

ACDC 的图像来自连续采集的 session，**同一 session 的相邻帧高度相似**。

如果按图像随机划分，同一场景的相邻帧会同时出现在训练集和验证集里，
验证指标会显著虚高（这个虚高幅度可能达到十几个点），
而模型在真正的新场景上并没有那么好。

因此划分策略必须是 `by_scene`（见 `configs/data/acdc.yaml` 的 `split.strategy`），
且划分结果落盘到 `data/processed/acdc/split.json`，保证所有实验用同一套划分。

---

## 2. 距离与检测标注的补充来源

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
距离监督  ←  KITTI / nuScenes（LiDAR 真值，几何精确）
```

即：**用 ACDC 的恶劣天气图像作为输入，用外部数据集的几何真值作为监督。**

这样做的好处是模型的视觉分布贴近目标场景；
代价是「几何真值对应的目标」和「图像里的目标」不是同一个，
所以需要额外的域适配手段（域差异缓解见下节）。

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
├── raw/                       # 原始下载数据，只读，不修改
│   ├── acdc/
│   │   ├── fog/
│   │   ├── night/
│   │   ├── rain/
│   │   └── snow/
│   └── distance/              # 外部数据集的原始文件
├── interim/                   # 中间产物（解析后但未最终成型的）
├── processed/                 # 训练直接读取的数据
│   ├── acdc/
│   │   ├── images/
│   │   ├── masks/
│   │   └── split.json         # 按 scene 的划分，固定不变
│   ├── distance/
│   │   └── labels.jsonl       # (frame_id, box, distance_m, source, valid)
│   ├── handover/
│   │   └── labels.json        # 构造的接管边界标签
│   └── instruction/           # Stage 2 的指令数据集
│       ├── train.jsonl
│       └── val.jsonl
└── external/                  # 第三方数据集原样存放
```

`data/` 整个目录已在 `.gitignore` 中，原因见 `LICENSE` 末尾的说明：
数据集许可不允许再分发。

---

## 5. 数据获取步骤

```bash
# 1. 下载 ACDC（需要先在官网注册并同意使用条款）
python scripts/download_acdc.py --config configs/data/acdc.yaml

# 2. 下载 KITTI（用于距离与检测监督）
#    自行从 https://www.cvlibs.net/datasets/kitti/ 获取，放到 data/external/kitti/

# 3. 预处理
python scripts/prepare_acdc.py --config configs/data/acdc.yaml

# 4. 构造距离标签
python scripts/build_distance_labels.py --config configs/data/distance_supplement.yaml

# 5. 构造接管边界标签（口径见 docs/handover_policy.md）
#    由 prepare 阶段附带产出

# 6. 统计与可视化，确认数据没问题再开训
python scripts/analyze_dataset.py
```

**第 6 步不要跳过。** 直接开训然后花几天调参去追一个本可以五分钟发现的数据问题，
是这类项目最常见的时间浪费。
