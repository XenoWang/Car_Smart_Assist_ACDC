# YOLO 的 PyTorch 精度对照

默认检测配置使用 FP32。候选配置 configs/model/yolo_fp16.yaml 选择 CUDA FP16 卷积，并引用相同的九类权重。校准、独立评估、演示和完整管线基准使用同一检测配置。

## 运行方式

先在 calibration 划分上生成本精度的校准记录，再使用 run_pipeline.py 或 benchmark_pipeline.py 的 --detection-config 指定候选配置。校准和基准执行遵循项目的逐命令审批流程。

    scripts/evaluate_detection.py --config configs/model/yolo_fp16.yaml --split calibration
    scripts/run_pipeline.py --device cuda:0 --detection-config configs/model/yolo_fp16.yaml --image IMAGE
    scripts/benchmark_pipeline.py --device cuda:0 --detection-config configs/model/yolo_fp16.yaml --image-list IMAGES.json --output-dir NEW_DIRECTORY

上述为参数示例，运行时使用项目解释器和明确的输入路径。FP16 校准文件位于配置的校准根目录下 best_class_threshold_v1/fp16/。FP32 继续使用 best_class_threshold_v1/。两种精度分别建立模型实例，因为后端会就地转换已载入的权重。

## 数值边界

标准 Detect 头的回归和分类卷积使用半精度。回归塔输出转为 FP32 后，原生检测头将其与分类 logits 拼接。DFL、锚点、框解码、分数计算和后续的阈值、NMS、原图坐标还原在 FP32 张量上进行。两种精度需要实测比较，因为卷积计算产生的舍入差异会传递到框与分数。兼容范围为本项目使用的标准 PyTorch Detect 权重及 CUDA 设备。

## 衡量速度与质量

在 289 张验证图上比较九类与交通八类 mAP50–95、道路障碍物标注区域 AP、夜间和障碍物小框召回。评估脚本的 --reference-fp32-calibration 参数可在独立 val 输出目录中复用经过身份校验的 FP32 阈值，以分离精度变化和重新校准的影响。部署操作点另用各自 313 张校准图的阈值评估。

速度以相同 38 张图片、单帧、三组交替独立进程测量完整管线吞吐和 P95，同时记录真正运行的检测后端精度。文件输入和预解码数组分别汇总。三组测速使用 --reference-fp32-calibration，在各自校准身份验证成功之后为本次基准绑定相同参考阈值，记录参考文件哈希。报告另列独立校准后的质量和测试进程的确定性设置。

当前实现待软件回归与真实测量。默认路由的采用决定由实测速度和质量代价共同决定。
