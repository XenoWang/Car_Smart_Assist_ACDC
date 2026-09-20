"""构造「前车 / 来车距离」这个任务的训练标签。

职责:
    - 这是 ACDC 原生缺失的标注，必须由外部来源派生（KITTI / nuScenes / 自采）
    - 提供两条派生路径:
        (a) 有深度图或 LiDAR 时，用 perception.geometry.roi_depth 在检测框内取稳健统计量
        (b) 只有单目图像时，用相机内参 + 已知目标尺度做针孔测距，或用单目深度模型估计
    - 输出与 ACDC 图像帧对齐的 (frame_id, box, distance_m, source) 记录
    - 必须记录每条标签的来源与置信度，便于后续做可信度加权和误差分析

依赖: car_smart_assist.perception.geometry
被谁调用: 人工执行；跑完 prepare_acdc.py 之后
备注: 距离标签的质量是整个项目最大的风险点，先在这一步把口径定死并写进 docs/label_spec.md
"""
