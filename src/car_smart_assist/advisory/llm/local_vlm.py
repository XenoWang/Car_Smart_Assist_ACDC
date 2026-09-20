"""本地 VLM 加载与推理封装。

职责:
    - 加载量化后的 VLM（12GB 显存 -> 4-bit 量化），处理 processor 的图像预处理
    - 提供 batch 推理与 KV cache 复用，控制首 token 延迟
    - 权重路径、量化配置、dtype 全部来自配置，不硬编码

依赖: transformers, peft, bitsandbytes
备注: 具体基座模型在 configs/model/advisory_llm.yaml 里定，先别把模型名写死在代码里
"""
