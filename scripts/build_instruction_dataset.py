"""构造 Stage 2 指令微调数据集（图像 + 指令 -> 司机提示文本）。

职责:
    - 用 Stage 1 感知模型的输出（或人工标注）作为结构化输入
    - 套用 advisory.prompt.templates 生成多风格样本：是否接管 / 路况描述 / 距离提示 / 调整建议
    - 负样本必须包含「可以不接管」的情形，避免模型学成永远喊接管
    - 输出 chat 格式 jsonl，落盘到 data/processed/instruction/
    - 按 ACDC 的 session 切分 train/val，禁止同一 scene 跨集合泄漏

依赖: car_smart_assist.advisory.prompt
被谁调用: 人工执行；训练 Stage 2 之前
"""
