"""评测循环。

职责:
    - 在 val/test 上跑推理，累积指标（通过 car_smart_assist.evaluation.metrics）
    - 对分割做多尺度/滑动窗口推理（若配置开启），对距离做分桶统计
    - 输出结构化结果 dict，交给 evaluation.report 落盘
    - 训练中途调用时只算轻量指标，全量指标留给独立评测

依赖: car_smart_assist.evaluation
被谁调用: engine.Trainer, scripts/evaluate.py
"""
