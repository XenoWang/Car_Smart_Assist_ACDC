# YOLO 基线质量与性能诊断

本工具用于定位交通目标和道路杂物检测的短板。它不训练模型、不改变选模权重，也不自动替换默认权重。评估与测速需要遵守项目的测试审查和精确命令批准流程。

## 质量诊断

在验证集上沿用当前部署校准，报告官方 AP／操作点与额外诊断：

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts\evaluate_detection.py --config configs/model/yolo_detection.yaml --split val --device cuda:0 --use-deployment-calibration --diagnostics --output-dir artifacts/reports/yolo_baseline_20261105/quality
```

输出位于给定目录下的后处理版本子目录（当前 best_class_threshold_v1）：

- val.json：原有正式预算 AP、九类和来源分项、锁定阈值操作点；记录实际校准来源。
- val_diagnostics.json：逐图框、匹配、来源／天气／类别分组，以及原图和网络输入两种面积坐标下的尺寸召回。
- 已有校准只读，必须与权重、数据签名、类别和后处理身份匹配；新目录已存在时拒绝覆盖。改变输入尺寸后不能继续使用身份不匹配的旧校准。

尺寸桶按框面积划分为 <16²、[16²,32²)、[32²,96²)、>=96²。网络输入面积直接来自验证 batch 的真实归一化框和输入 H/W，包括实际缩放，不把 imgsz 当作正方形真实 shape。尺寸结果是召回统计，不是 COCO AP-small/medium/large。

额外诊断使用同类 IoU>=0.5、IoU 降序的一对一贪心匹配；并列时按 GT／预测索引决定。这是公开且可复查的诊断口径，可能与官方验证器在竞争／并列框上不同，不能替代原 AP 或官方 TP/FP/FN。误检分为重复重叠、错类重叠、同类低 IoU 重叠和其他未匹配，只表述几何关系，不声称找到因果根因。无 GT 的召回为 null。

联合数据继续遵守已监督类别和 Lost & Found 标注 ROI；不会把未标注交通目标或 ROI 外区域当负例。逐类 image_count 只统计该类被覆盖的图。天气分组来自现有目录标签，不表示完整多天气共现真值。默认使用 val，反复查看过的 test 不会恢复独立性。

## 性能与耗时分解

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts\benchmark_pipeline.py --image-list AI_Skills/Reviews/yolo_baseline_profile_20261105/images.json --device cuda:0 --batch-size 1 --warmup 2 --repeats 5 --profile-stages --profile-repeats 3 --target-fps 30 --output-dir artifacts/reports/yolo_baseline_20261105/performance
```

图片清单是 UTF-8 JSON 字符串数组；相对路径以项目根为基准。保持输入顺序，清单身份进入 metadata.json。每次使用新的输出目录。

- summary.json／measurements.csv：原有无阶段包装的完整管线计时，区分文件输入、预解码输入、first_pass、warmup、steady。目标判断只使用 batch=1 的稳态预解码完整感知路径，检查吞吐>=30 FPS 且 P95<=33.333 ms；仅称在观测样本上满足。批量吞吐、BLIND 跳过或故障快速返回不能冒充单帧完整感知达标。
- stage_profile.json：额外的预解码、batch=1 诊断遍历，与上述正常测速数据完全分开。记录门控／缩放／评分、天气、检测总调用、YOLO letterbox／预处理／前向／后处理、规则的同步墙钟、P50/P95/P99、失败调用与逐帧路径状态；记录真实 YOLO 输入 shape、dtype、device。
- 阶段时间是包含子调用的时间，不能相加；额外同步与包装会增加开销。预处理包含布局转换、传输、归一化，后处理包含 NMS 和坐标映射；未单独量测每个传输、内核或 NMS。
- 模型构建、首次调用与稳态分开。当前工具的文件模式先运行、数组模式使用已预热模型；路径与数组时间差不能当纯解码耗时。摄像头采集、排队、显示与车载设备延迟未测。
- 诊断结束或异常后恢复实例方法；不改第三方全局函数，不改变依赖或默认配置。原始数据、权重和校准保持原位，各实验使用独立输出目录。

当前实现需经过独立软件测试后再进行真实模型评估／测速。软件通过不代表模型达到速度或精度目标。
