"""把数据集、变换、采样器组装成训练/验证/测试三个 DataLoader。

职责:
    - 依据 DataConfig 构建 train/val/test 划分，保证同一 scene 不跨集合
    - 组装 collate_fn（多任务模型需要把 seg mask / cls label / box / distance 打包）
    - 提供 num_classes、class_weights 等下游需要的元信息
    - 这是 data 层对外的唯一入口，engine 只认它

依赖: data.datasets, data.transforms, data.samplers
被谁调用: car_smart_assist.engine.Trainer, 各 scripts
"""
