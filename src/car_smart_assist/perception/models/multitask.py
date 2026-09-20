"""多任务感知模型：共享骨干 + 四个任务头。

职责:
    - 组装 backbone 与 seg / road_condition / handover / detection+distance 各 head
    - forward 返回 dict（每个 key 对应一个任务的输出），不在这里算 loss
    - 管理与多任务损失加权的交互：支持固定权重与不确定性自动加权两种模式
    - 支持只跑部分 head（消融与分阶段训练）

依赖: perception.backbones, perception.heads
被谁调用: engine.Trainer, inference.Predictor
"""
