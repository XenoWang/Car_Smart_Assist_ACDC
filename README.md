# Car Smart Assist

面向复杂路况的单目驾驶辅助项目，输出能见度、天气属性、目标检测框，以及风险和驾驶提醒。
当前可运行部分包括数据清洗、能见度模型、独立多标签天气模型、YOLO 检测和规则提示。

## 环境

Python 3.12，使用项目 `.venv`。

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

检查训练设备：

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

清洗输出损坏文件过滤清单，原文件保留。

## 训练

能见度：

```powershell
.venv\Scripts\python.exe scripts\train_visibility.py
```

天气增强：

```powershell
.venv\Scripts\python.exe scripts\train_weather.py --config configs/model/weather_enhanced.yaml
```

增强配置引用原多标签模型，候选写入 `artifacts/checkpoints/weather_enhanced/`。
默认续训；`--fresh` 在所选配置的输出目录开始新运行。

检测：

```powershell
.venv\Scripts\python.exe scripts\prepare_detection.py --config configs/data/acdc_lost_and_found_detection.yaml
.venv\Scripts\python.exe scripts\train_detection.py
```

## 验证天气候选

```powershell
.venv\Scripts\python.exe -X utf8 scripts\evaluate_weather.py --config configs/model/weather_enhanced.yaml --dataset validation --compare-teacher
.venv\Scripts\python.exe -X utf8 scripts\evaluate_weather.py --config configs/model/weather_enhanced.yaml --dataset calibration --compare-teacher
```

`--compare-teacher` 使用同一批图片比较蒸馏原模型与候选。报告位于 `artifacts/reports/weather_enhanced/`。
候选尚未切换为 pipeline 默认天气模型。

官方天气测试及未见录制子集：

```powershell
.venv\Scripts\python.exe -X utf8 scripts\evaluate_weather.py --config configs/model/weather_enhanced.yaml --dataset acdc --compare-teacher
.venv\Scripts\python.exe -X utf8 scripts\evaluate_weather.py --config configs/model/weather_enhanced.yaml --dataset unseen-recordings --compare-teacher
```

## 推理与回归

```powershell
.venv\Scripts\python.exe scripts\run_pipeline.py --image "图片路径" --json
.venv\Scripts\python.exe scripts\run_pipeline.py --json --log-level INFO
.venv\Scripts\python.exe -X utf8 -m pytest tests -q --basetemp artifacts/pytest-tmp
```

推理结果包含 `visibility`、`weather`、`weather_warning`、`perception`、`advisory` 和 `skipped`。
查看 `skipped` 可确认当前帧未运行的模块；距离和方向尚无预测结果。

## 配置

| 功能 | 配置 |
|---|---|
| 能见度 | `configs/model/visibility.yaml` |
| 默认天气模型 | `configs/model/weather_classifier.yaml` |
| 天气增强候选 | `configs/model/weather_enhanced.yaml` |
| 联合 YOLO | `configs/model/yolo_detection.yaml` |
| 旧八类 YOLO | `configs/model/yolo_acdc_baseline.yaml` |
| 风险与接管规则 | `configs/model/advisory_llm.yaml` |

## 文档

- [架构示意图](docs/architecture.md)
- [数据说明](docs/dataset.md)
- [标签与接口](docs/label_spec.md)
- [风险与接管输出](docs/handover_policy.md)
- [变更记录](CHANGELOG.md)

代码许可见 [LICENSE](LICENSE)。
