"""Stage 2 训练入口：司机提示生成（VLM/LLM 微调）。

职责:
    - 解析 configs/train/advisory_sft.yaml，加载指令数据集
    - 冻结 Stage 1 权重，只训练语言侧（LoRA / QLoRA，适配 12GB 显存）
    - 支持 --resume；支持只做推理不训练（--eval-only）

依赖: car_smart_assist.advisory, car_smart_assist.engine
被谁调用: 人工执行
备注: 显存预算紧，默认走 4-bit QLoRA；不要默认全参微调
"""
