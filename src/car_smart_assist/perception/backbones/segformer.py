"""SegFormer / MiT 骨干网封装。

职责:
    - 加载预训练权重（timm 或 HF），输出多尺度特征金字塔供分割头与检测头共享
    - 暴露 out_channels / stride 等元信息给下游 head 自动对齐通道数
    - 支持 freeze / 分层学习率，便于小数据量下微调

依赖: timm 或 transformers, torch
备注: 备选骨干网（ResNet / ConvNeXt）后续以同样接口加进来；12GB 显存优先选轻量档
"""
