"""模型加载与单帧/批量推理。

职责:
    - 从 checkpoint 加载 Stage 1 模型，处理 device / dtype / eval 模式切换
    - 预处理必须复用训练时的 transforms（同一套归一化参数），否则指标会对不上
    - 提供 batch 推理接口，供评测与 demo 共用
"""
