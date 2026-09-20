"""针孔相机模型与基础测距几何。

职责:
    - 相机内参矩阵的表示、归一化与逆变换（含 resize/crop 后的内参更新）
    - 由「像素宽度 + 已知真实宽度 + 焦距」推距离，由「像素质心 + 内参」推方位角
    - 纯函数、无状态、可单测；不要依赖 torch 的自动微分

依赖: numpy
被谁调用: perception.geometry.roi_depth, scripts/build_distance_labels.py
"""
