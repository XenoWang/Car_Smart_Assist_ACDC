"""Stage 2 指令微调数据集。

职责:
    - 读取 data/processed/instruction/*.jsonl，产出 (image, conversation) 样本
    - 负责 chat template 套用与 label mask（只对 assistant 回复算 loss）
    - 支持图像分辨率与 patch 数控制，避免显存爆掉

依赖: transformers processor（具体模型在 configs/model/advisory_llm.yaml 指定）
"""
