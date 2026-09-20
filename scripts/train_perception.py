"""Stage 1 训练入口：多任务感知模型。

职责:
    - 解析 configs/train/perception.yaml 与其继承链，构建 datamodule / model / trainer
    - 实例化 car_smart_assist.engine.Trainer 并跑完整训练循环
    - 支持 --resume 从 checkpoint 续训、--dry-run 只跑几个 step 验证通路
    - 本身不写训练逻辑，只做装配（composition root）

依赖: car_smart_assist.engine, car_smart_assist.perception, car_smart_assist.data
被谁调用: 人工执行 `python scripts/train_perception.py --config configs/train/perception.yaml`
"""
