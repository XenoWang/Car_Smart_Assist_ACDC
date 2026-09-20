"""数据集探查与可视化，产出给人和给报告用的统计。

职责:
    - 分布统计：天气子集占比、类别像素占比、距离标签分桶直方图、接管/不接管标签比例
    - 可视化抽样图：原图 + 分割叠加 + 检测框 + 预测距离，存到 artifacts/reports/figures/
    - 检查数据完整性：缺失标注、损坏图像、尺寸异常
    - 结论写进 artifacts/reports/dataset_analysis.md，作为 README 里数据章节的依据

依赖: car_smart_assist.data, car_smart_assist.utils.visualize
被谁调用: 人工执行；数据准备好之后、开训之前
"""
