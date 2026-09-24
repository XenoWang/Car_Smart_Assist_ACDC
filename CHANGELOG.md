# Changelog

本文件记录项目的重要变更。格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

写这个文件的目的不只是规范：简历项目里「有没有变更记录」是判断
「这是持续迭代的项目还是三天做完的一次性作业」的直接信号。

## [Unreleased]

### 本轮开发版本更新（基于 0.0.1，尚未发布）

#### Added
- 能见度参考图改为 train / validation / calibration / test 四划分（默认 80% / 5% / 5% / 10%），
  按视频序列互斥；validation 负责早停与 best 权重选择，calibration 只计算正常误差分布，
  test 留给最终评估。
- 将能见度特征阈值、归一化尺度、聚合权重与阶数、分块误差网格，以及 DEGRADED 置信度乘子
  放入 `configs/model/visibility.yaml` 的配置字典；训练检查点保存打分配置，推理入口支持显式配置覆盖。

#### Changed
- `best.pt` 现在按 validation 重建损失选择，回滚到对应权重后再独立计算 calibration 统计。
  train / validation 损失改为按样本数汇总；训练集不再丢弃不满一个 batch 的尾部样本。
- 消融实验只用 validation 比较误报和合成退化召回，不再读取 test 图像；旧协议缓存结果会要求重跑。
- 拟合评估改为在各切分内固定随机抽样，不再取排序后开头的样本。
- 检查点加入数据切分签名，避免数据、分辨率或切分变化时沿用不匹配的早停历史。

#### Fixed
- 拒绝空的 train / validation / calibration / test 切分，避免空 validation 被误当成零损失并错误选为 best。
- 旧格式检查点不再用于新流程续训（其最优权重可能由 calibration 选择）；训练入口提示用 `--fresh`，
  旧 `best.pt` 仍可用于推理。

### Added
- 初始化项目骨架：两阶段架构（Stage 1 多任务感知 / Stage 2 自然语言建议）的完整目录结构
- 配置文件体系：`configs/` 下按 data / model / train / inference 分层，支持继承与命令行覆盖
- 依赖拆分：镜像源可加速的运行时依赖与必须走官方 CUDA 索引的 torch 栈分离
- 工程元数据：`pyproject.toml`（ruff / mypy / pytest / coverage 配置）、`.editorconfig`、
  `.pre-commit-config.yaml`

### Fixed
- **勘误：ACDC v2 是有检测框标注的。** 早期文档称「ACDC 无检测标注，检测监督必须来自
  外部数据集」，这是基于 ICCV 2021 v1 的过时信息。ACDC v2（TPAMI 2025 扩展版）提供
  `gt_detection_trainval.zip`（2006 张图的检测框）。影响：检测任务不再需要跨域，
  域差异问题从「检测 + 距离」缩小到只剩「距离」。
  涉及文件：`docs/dataset.md`、`configs/model/perception_multitask.yaml`（`label_source`
  由 `external` 改为 `acdc`）、`configs/data/distance_supplement.yaml`、
  `docs/roadmap.md`、`README.md`、`scripts/build_distance_labels.py`、
  `perception/heads/vehicle_detection.py`
- **补充：ACDC 与 KITTI 都不提供「前车 / 来车」方向标签。** 方向必须由朝向角或车道几何
  推导，这是原设计漏掉的一步。已在 `docs/label_spec.md` 4.1 定义推导规则与 unknown 策略，
  并在 `configs/model/perception_multitask.yaml` 增加 `direction_derivation` 配置段
- **补充：KITTI `DontCare` 类占目标总数 26%，不是背景。** 当负样本用会教出
  「车辆附近是背景」的错误信号。处理方式见 `docs/label_spec.md` 4.2
- **补充：ACDC(1920×1080) 与 KITTI(1224×370) 长宽比不同**，内参不能共用一套。
  见 `docs/label_spec.md` 4.3

### Notes
- 本阶段只建立结构，各模块文件内以 docstring 描述职责，尚未实现具体逻辑
- 目标硬件：NVIDIA RTX 5070（sm_120, 12GB），CUDA 12.9，torch 2.8.0+cu129
- ACDC 无免注册下载途径（已实测确认），必须走官网申请流程。注意 "ACDC" 缩写
  至少对应三个数据集，搜索时极易命中同名的「心脏 MRI」数据集

### Data
- 已下载并验证 KITTI 目标检测数据集：7481 训练帧 + 7518 测试帧，
  含 2D/3D 框标注与相机内参（AWS S3 直链，免注册）
- 未下载 `data_object_velodyne.zip`（27.4 GB）—— 距离可从 `label_2` 的 3D 位置直接取得
- 已下载并解压 ACDC 全部四个包，**md5 全部与官方 `/api/packages` 一致**：
  `rgb_anon_trainvaltest.zip` 8012 张图（4006 恶劣 + 4006 正常参考）、
  `gt_trainval.zip` 2006 张语义标注、`gt_detection_trainval.zip`、`gt_panoptic_trainval.zip`

### Added — 能见度门控（无监督第一阶段）
- `perception/visibility/` 模块：多尺度卷积自编码器 + 确定性信息量特征 + 三档门控
  - `autoencoder.py`  **3×3 / 5×5 / 7×7 三路并行卷积分支**，通道维拼接后 1×1 融合；
    各分支 padding=d(k-1)/2 保证任意 (k,d) 组合输出尺寸一致。支持空洞卷积
    （`dilations` 可配，3×3@(1,2,3) 与 3×5×7 感受野相同但参数量约 1/3）
  - `dataset.py`     参考图数据集 + 磁盘 npy 缓存；三划分与按序列整组划分
  - `scorer.py`      重建误差 z 分数 + 对比度/熵/边缘密度/高频能量占比
  - `gate.py`        三档判定：VISIBLE / DEGRADED / BLIND，含触发原因
  - `degradation.py` 合成雾/黑暗/遮挡/模糊，用于定量验证召回率
  - `trainer.py`     无监督训练 + **检查点保存与续训**（原子写、last/best 双文件）
- `scripts/train_visibility.py` / `scripts/evaluate_visibility.py`
- `configs/model/visibility.yaml`

### Added — 能见度门控接入推理管线
- `inference/pipeline.py`  **完整实现**。管线结构改为「门控 -> 感知 -> 建议」三段，
  门控是第一级判断：
  - BLIND 时**跳过感知**，直接出门控驱动的接管请求（省算力，且避免
    在看不见的帧上硬跑感知、把噪声当结果喂给决策层）
  - DEGRADED/VISIBLE 才进入感知阶段
  - **两个阶段用不同分辨率**：门控跑 144×256 降采样（能见度是全局属性，
    且 AE 就是按这个尺寸训练的），感知必须用**原始分辨率**
    （检测框与单目测距依赖原始像素尺度与内参，混用会让距离整体偏掉）
  - 每阶段耗时分开记录；未接入的阶段记进 `skipped` 并说明原因，不假装跑过
- `advisory/schema.py`  实现数据契约：`PerceptionResult` / `AdvisoryResult` /
  `TargetObject`。两条关键约定：
  - **未知距离用 `None` 而非 0** —— 0 米表示「贴脸」，与「不知道」是两回事，
    混用会让下游把「没测出来」当成「很近」，恰好造成最危险的误判
  - **`AdvisoryResult.text` 禁止为空**（构造期抛错）—— 宁可回退保守文案，
    也不允许静默失声
  - `effective_confidence()` 统一施加门控降级乘子，避免下游各自重复判断能见度
- `advisory/generator.py`  实现**门控驱动的接管路径**（安全关键，不依赖任何未实现模块）。
  感知驱动的常规路径明确 `NotImplementedError` —— 在安全链路上，
  一个「看起来能跑但输出是编的」占位实现比缺实现危险得多
- `advisory/prompt/templates.py`  司机文案模板 + `check_wording()` 可编程措辞自检
- `scripts/run_pipeline.py`  端到端演示，含各天气子集与合成退化对照
- `tests/unit/test_inference_pipeline.py`  40 个测试，锁住门控阻断、
  分辨率分离、输入格式、数据契约、文案约束

### Fixed
- `tests/unit/test_pipeline.py` 与 `tests/integration/test_pipeline.py` **重名**，
  pytest 在无 `__init__.py` 时无法区分同名测试模块，全量收集直接失败。
  已重命名为 `test_inference_pipeline.py`
  （单文件跑能过、全量跑才暴露，这类问题只有跑全量才会发现）

### Added — 消融实验驱动器
- `scripts/run_visibility_ablations.py`：统一跑 config 里的 6 个预设 + 基线。
  保证**同一套划分**、**独立 checkpoint 目录**（共用会续训而非重训，结论全错）、
  **参考图只解码一次**；结果逐组落盘可断点重入
- **跑完的结论：本次消融无效，且原因是数学必然的。**
  `use_recon_z=false` 时门控判定只依赖 `compute_information_features(图像)`，
  该函数不经过模型 —— 换任何编码器，召回率与误报率必然逐位相同。
  实测 7 组结果：平均召回全部 95.4%、误报全部 0.0%、信息量曲线逐位一致。
  全表唯一依赖模型的只有 `overfit_ratio`（1.062~1.142）。
  已加**有效性闸门**：`use_recon_z` 为 false 时直接拒绝运行并说明原因，
  避免再花 56 分钟（7 组 × 8 分钟）测一个恒等于零的效应
- 新增 `TestAblationPresetsAreValid`：校验每个预设都能建出模型并前向通过。
  起因是 `kernels_3_5` 预设只改 kernel_sizes 没改 dilations，长度不匹配，
  消融跑到第 4 组才崩 —— 前面三组各 7 分钟已白跑

### Fixed — 能见度门控（均为实测驱动，非预防性修改）
- **信息量聚合方式**：加权几何平均会被「未塌陷的维度」稀释，导致模糊召回仅 5.0%、
  遮挡仅 3.3%。改为**广义平均 p<0（趋近最小值）**，并给单维加 eps 下限防止
  绝对支配（hf_ratio 在轻度模糊时就饱和到 0.024，无下限时会把分数钉死、
  反而制造误报）。修复后模糊 92.5%、遮挡 100%，且模糊重新具备分级能力
- **重建误差与退化反相关**：实测清晰图 z=-0.57，合成雾 -1.80、黑暗 -1.80、
  遮挡 -1.60、模糊 -1.36，真实雾 -1.58。两个方向都不可用，因此
  `use_recon_z` **默认关闭**，判定只依赖确定性信息量特征
- **过拟合（1.456）实为划分假象**：原按图随机划分，而 ACDC 参考图来自视频序列、
  相邻帧近乎重复，校准集里混进了训练样本的复制品。改为**按序列整组划分**后
  比值降到 **1.013**（拟合充分）。峰值时曾误判为容量问题并降了容量、
  加了去噪正则 —— 两项都保留了，但它们不是真正的解因
- **训练日志的 train/calib 对比不可信**：训练损失在开增强的数据上算、
  校验损失没有，口径不一致会把真实差距压小（日志口径 1.14 vs 直接测量 1.46）。
  评估脚本改为直接测原始图像
- **模型容量**：瓶颈前缺 1×1 投影，两个全连接层占模型约 90% 参数（9.46M）。
  加 `pre_latent_channels=32` 后降到 2.36M，模型 10.5M -> 2.6M
- **Dataset 持有 memmap 导致 worker pickle 截断**：Windows spawn 下
  443MB 数组被内联进 pickle 直接报 `pickle data was truncated`。
  改为只传路径、每个 worker 惰性打开
- **续训时 `self.model_cfg` 未同步**：用检查点结构建了模型却保留配置里的旧值，
  下次保存会把与权重不匹配的结构写进检查点

### Changed
- 能见度门控的判定阈值重新标定：`info_blind` 0.12 -> 0.34、`info_degraded` 0.30 -> 0.50。
  依据：合成退化最大强度得分上界 0.280（blur）与真实 ACDC 最低分 0.416（night）
  之间的可分区间
- **清洗策略修正：内容类判断不再排除样本。** 原先「灰度方差 < 2.0」被当作 ERROR 直接排除，
  但该判据无法区分「采集失败的全黑帧」与「合法的大雾/夜路帧」—— 而后者正是 ACDC 的核心内容。
  按方差阈值自动排除会系统性删掉最该被学会的数据，且报告上只显示「排除 N 张退化图」，
  看不出问题。
  现改为：**只有「数据不可用」才排除**（无法解码、配套文件缺失、结构不一致），
  **「图像不寻常」一律只告警**（方差偏低、尺寸偏小、宽高比异常、疑似重复、统计离群）。
  内容类检查的严重度可在 `configs/data/cleaning.yaml` 的
  `image_statistics.severity` 显式改为 error，但需先人工抽查确认。
  已加回归测试（`TestDimensionsAndDegeneracy` / `TestExclusionContract`），
  并用「注入回归 → 测试必须失败」验证过保护有效。
- **方向标签方案改为「ACDC 序列时序推导」。** 解压后确认 ACDC 检测标注
  不含任何朝向/3D 信息（字段仅 `area,bbox,category_id,id,image_id,iscrowd,segmentation`），
  原先配置的 `strategy: yaw_angle` 在 ACDC 上无法执行。改为在同一视频序列内跟踪目标、
  据横向运动判定同向/对向，**时序仅用于离线造标签，推理仍是单帧**。
  已实测可行：val 集 21 个序列、18 个超 3 帧、最长 56 帧
- **数据集划分改用 ACDC 官方划分。** 早期配置写「官方不提供划分，自己按 70/15/15 切」
  是错的 —— ACDC 按 sequence 切好了 train(1600)/val(406)/test(2000)。
  用自造划分将无法与已发表结果对比

### Fixed
- **`.gitignore` 漏掉了 `data/` 根目录下的大文件。** 原先只忽略 `raw/interim/processed/external`
  四个子目录，导致放在 `data/` 根目录的 16 GB ACDC zip 不在忽略范围，
  一次 `git add .` 就会把它提交进仓库。改为 `data/**` + `artifacts/**` 整体忽略、
  仅重包含 `.gitkeep`。已用 `git add -A --dry-run` 验证：只会暂存代码与文档，零数据文件

---

## 待办里程碑（Milestone 草案）

- **v0.1.0 — 数据管线**：ACDC 下载/预处理/统计可用，距离标签口径定稿
- **v0.2.0 — Stage 1 感知**：多任务模型跑通，分割/分类/距离三组指标有基线
- **v0.3.0 — 接管边界**：接管边界标签体系定稿，分类器达到可用指标
- **v0.4.0 — Stage 2 建议**：结构化输出到司机提示的生成链路可用，含降级路径
- **v0.5.0 — 评测与报告**：端到端评测脚本 + 失败案例分析报告
- **v1.0.0 — 收尾**：文档完整、指标可复现、demo 可演示、可选 ONNX 导出
