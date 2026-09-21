"""构造「前车 / 来车距离」这个任务的训练标签。

职责:
    - 这是 ACDC 原生缺失的标注，必须由外部来源派生（KITTI / nuScenes / 自采）
      注：ACDC v2 自带检测框，所以**只有距离这一项需要外部数据**，
      本脚本的作用范围因此比原设计小得多。
    - KITTI 路径：直接读 label_2 的第 13 字段（相机坐标系下的 z = 距离），
      这是 LiDAR 真值，不需要任何估计 —— 优先用这条路径
    - 备选路径（自采数据无 3D 标注时）:
        (a) 有深度图或点云时，用 perception.geometry.roi_depth 在框内取稳健统计量
        (b) 只有单目图像时，用相机内参 + 已知目标尺度做针孔测距
    - 输出与 ACDC 图像帧对齐的 (frame_id, box, distance_m, source, valid) 记录
    - 必须记录每条标签的来源与置信度，便于后续做可信度加权和误差分析

依赖: car_smart_assist.perception.geometry
被谁调用: 人工执行；跑完 prepare_acdc.py 之后
备注:
    距离标签的质量是整个项目最大的风险点，先在这一步把口径定死并写进 docs/label_spec.md。
    特别注意 KITTI 的 DontCare 类（占目标数 26%）不是背景，必须在框级标签里排除，
    否则模型会学到「车辆附近的位置是背景」，直接压低召回率。
"""
