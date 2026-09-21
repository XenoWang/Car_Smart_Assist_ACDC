# Changelog

本文件记录项目的重要变更。格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

写这个文件的目的不只是规范：简历项目里「有没有变更记录」是判断
「这是持续迭代的项目还是三天做完的一次性作业」的直接信号。

## [Unreleased]

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

### Changed
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

> 这一段在开工后应逐步替换成真实的版本记录。先留着是因为它能说明项目是
> 「有节奏推进」而不是「想到哪做到哪」。

- **v0.1.0 — 数据管线**：ACDC 下载/预处理/统计可用，距离标签口径定稿
- **v0.2.0 — Stage 1 感知**：多任务模型跑通，分割/分类/距离三组指标有基线
- **v0.3.0 — 接管边界**：接管边界标签体系定稿，分类器达到可用指标
- **v0.4.0 — Stage 2 建议**：结构化输出到司机提示的生成链路可用，含降级路径
- **v0.5.0 — 评测与报告**：端到端评测脚本 + 失败案例分析报告
- **v1.0.0 — 收尾**：文档完整、指标可复现、demo 可演示、可选 ONNX 导出
