"""把 ACDC 原始数据转换成训练可用的中间/处理后格式。

职责:
    - 解析官方标注（semantic / panoptic / correspondences）并统一成项目内部标签空间
    - 调用 car_smart_assist.data.label_mapping 完成官方 label id -> 内部 class id 的映射
    - 产出 data/processed/acdc/ 下的索引清单（图像路径 + 标注路径 + 天气标签）
    - 统计类别分布、天气分布，写进 artifacts/reports/dataset_stats.json

依赖: car_smart_assist.data.label_mapping, car_smart_assist.utils.io
被谁调用: 人工执行；训练前必须先跑一次
"""
