# =============================================================================
# 常用命令入口
# =============================================================================
# Windows 提示：如果没装 make（或不在 Git Bash 里），可以
#   (a) 直接执行 `python -m car_smart_assist.cli.main <子命令>`
#   (b) 或安装 make：choco install make
# 这里保留 Makefile 是为了让「项目怎么跑」有一个单一、可读的入口 ——
# 面试时对方看 Makefile 三秒就能知道你项目的完整生命周期。
# =============================================================================

PYTHON     ?= python
PIP        ?= $(PYTHON) -m pip
MIRROR     ?= https://pypi.tuna.tsinghua.edu.cn/simple
CONFIG_DIR := configs

.DEFAULT_GOAL := help

# -----------------------------------------------------------------------------
# 环境
# -----------------------------------------------------------------------------
.PHONY: venv
venv:  ## 创建虚拟环境（Python 3.12）
	$(PYTHON) -m venv .venv

.PHONY: install
install:  ## 安装依赖：先 torch（官方 CUDA 索引），再运行时依赖（镜像源）
	$(PIP) install -r requirements/requirements-torch.txt
	$(PIP) install -r requirements/requirements.txt -i $(MIRROR)
	$(PIP) install -e . --no-deps

.PHONY: install-dev
install-dev: install  ## 安装依赖 + 开发工具
	$(PIP) install -r requirements/requirements-dev.txt -i $(MIRROR)
	pre-commit install

.PHONY: freeze
freeze:  ## 冻结精确版本，保证实验可复现
	$(PIP) freeze > requirements/requirements.lock.txt

.PHONY: check-env
check-env:  ## 自检：确认 torch 用的是 CUDA 版而不是 CPU 版
	$(PYTHON) -c "import torch,sys; \
		print('torch   :', torch.__version__); \
		print('cuda ok :', torch.cuda.is_available()); \
		print('arch    :', torch.cuda.get_arch_list()); \
		sys.exit(0 if torch.cuda.is_available() else 1)"

# -----------------------------------------------------------------------------
# 数据
# -----------------------------------------------------------------------------
.PHONY: data-download
data-download:  ## 下载 ACDC 原始数据到 data/raw/
	$(PYTHON) scripts/download_acdc.py --config $(CONFIG_DIR)/data/acdc.yaml

.PHONY: data-prepare
data-prepare:  ## 预处理 ACDC，产出 data/processed/ 与索引清单
	$(PYTHON) scripts/prepare_acdc.py --config $(CONFIG_DIR)/data/acdc.yaml

.PHONY: data-distance
data-distance:  ## 构造前车/来车距离标签（ACDC 原生缺失，见 docs/label_spec.md）
	$(PYTHON) scripts/build_distance_labels.py --config $(CONFIG_DIR)/data/distance_supplement.yaml

.PHONY: data-instruction
data-instruction:  ## 构造 Stage 2 指令微调数据集
	$(PYTHON) scripts/build_instruction_dataset.py --config $(CONFIG_DIR)/data/acdc.yaml

.PHONY: data-analyze
data-analyze:  ## 数据分布统计与抽样可视化 -> artifacts/reports/
	$(PYTHON) scripts/analyze_dataset.py

# -----------------------------------------------------------------------------
# 训练
# -----------------------------------------------------------------------------
.PHONY: train-perception
train-perception:  ## Stage 1：训练多任务感知模型
	$(PYTHON) scripts/train_perception.py --config $(CONFIG_DIR)/train/perception.yaml

.PHONY: train-advisory
train-advisory:  ## Stage 2：训练司机提示生成（QLoRA）
	$(PYTHON) scripts/train_advisory.py --config $(CONFIG_DIR)/train/advisory_sft.yaml

.PHONY: dry-run
dry-run:  ## 只跑几个 step，验证训练通路是否打通
	$(PYTHON) scripts/train_perception.py --config $(CONFIG_DIR)/train/perception.yaml --dry-run

# -----------------------------------------------------------------------------
# 评测与推理
# -----------------------------------------------------------------------------
.PHONY: eval
eval:  ## 两个 Stage 的完整评测
	$(PYTHON) scripts/evaluate.py --config $(CONFIG_DIR)/inference/default.yaml --stage all

.PHONY: infer
infer:  ## 单图端到端推理 demo：图像 -> 司机提示
	$(PYTHON) -m car_smart_assist.cli.main infer --config $(CONFIG_DIR)/inference/default.yaml

.PHONY: export
export:  ## 导出 Stage 1 为 ONNX（可选）
	$(PYTHON) scripts/export_onnx.py --config $(CONFIG_DIR)/inference/default.yaml

# -----------------------------------------------------------------------------
# 质量
# -----------------------------------------------------------------------------
.PHONY: test
test:  ## 跑单元测试（跳过慢速与 GPU 用例）
	$(PYTHON) -m pytest -m "not slow and not gpu"

.PHONY: test-all
test-all:  ## 跑全部测试（含 GPU，需本机有显卡）
	$(PYTHON) -m pytest

.PHONY: cov
cov:  ## 测试 + 覆盖率报告
	$(PYTHON) -m pytest --cov=car_smart_assist --cov-report=html --cov-report=term-missing

.PHONY: lint
lint:  ## 代码检查（不修改文件）
	ruff check src tests scripts
	ruff format --check src tests scripts

.PHONY: fmt
fmt:  ## 自动格式化与修复
	ruff check --fix src tests scripts
	ruff format src tests scripts

.PHONY: typecheck
typecheck:  ## 类型检查（只覆盖核心模块）
	mypy

.PHONY: clean
clean:  ## 清理 Python 缓存（不动数据与权重）
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache .mypy_cache build dist *.egg-info

# -----------------------------------------------------------------------------
.PHONY: help
help:  ## 显示本帮助
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'
