"""通用训练循环。

职责:
    - 前向/反向/梯度裁剪/累积/AMP 混合精度；按 step 或 epoch 组织
    - 调用回调（日志、checkpoint、早停、评测）
    - 支持 resume（含优化器与 scheduler 状态恢复，不只是权重）
    - 不感知具体模型结构，只按 forward 返回的 dict + 配置里的 loss 权重来算

依赖: engine.optimizer, engine.scheduler, engine.callbacks, engine.checkpoint
被谁调用: scripts/train_perception.py, scripts/train_advisory.py
"""
