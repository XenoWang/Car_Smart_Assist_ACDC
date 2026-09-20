"""学习率调度器。

职责:
    - warmup + cosine / poly / step 等策略，配置驱动
    - 提供按 step 计数的接口（不是只有 epoch），配合梯度累积正确推进
"""
