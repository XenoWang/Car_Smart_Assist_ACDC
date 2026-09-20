"""ACDC 语义分割数据集。

职责:
    - 读取 data/processed/acdc/ 的图像与语义标注，输出 image + seg mask
    - 天气子集标签（fog/night/rain/snow）一并返回，供采样器做平衡与做分层指标
    - 支持按天气子集筛选（用于消融：只在 night 上训/评）

依赖: data.datasets.base, data.label_mapping
"""
