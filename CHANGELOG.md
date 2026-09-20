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

### Notes
- 本阶段只建立结构，各模块文件内以 docstring 描述职责，尚未实现具体逻辑
- 目标硬件：NVIDIA RTX 5070（sm_120, 12GB），CUDA 12.9，torch 2.8.0+cu129

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
