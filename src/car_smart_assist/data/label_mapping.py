"""官方标注 id 与项目内部类别空间的映射表。

职责:
    - ACDC 官方的 semantic id -> 本项目内部 class id 的映射（含 ignore_index 定义）
    - 提供官方调色板，供可视化时颜色一致
    - 映射表口径变更必须同步更新 docs/label_spec.md，并且是破坏性变更
"""
