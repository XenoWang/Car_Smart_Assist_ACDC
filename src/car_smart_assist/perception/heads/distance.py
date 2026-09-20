"""距离回归头。

职责:
    - 对每个检测框回归「到自车的距离（米）」
    - 建议预测 log(distance) 而非 distance 本身，数值更稳定；损失用 Huber/SILU 抗离群
    - 输出附加不确定度估计（如 heteroscedastic 方差），供 Stage 2 做保守决策

依赖: vehicle_detection 输出的框
备注: 只用单目图像做绝对距离误差会较大，评测里必须如实报告 MAE 并说明适用距离范围
"""
