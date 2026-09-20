"""随机种子与可复现性。

职责:
    - 统一设置 random / numpy / torch（含 cuda）种子
    - 提供 cudnn deterministic 开关（会掉速，仅在做严格对比实验时开）
    - 在训练启动时调用一次并记录种子值到日志与 checkpoint
"""
