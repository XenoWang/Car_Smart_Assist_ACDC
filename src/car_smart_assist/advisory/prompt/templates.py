"""提示词模板与结构化输入的渲染。

职责:
    - 把 PerceptionResult 渲染成紧凑、无歧义的自然语言上下文（距离保留合适精度、明确单位）
    - 维护多种输出风格模板（简洁/详细），并保证模板与 build_instruction_dataset.py 一致
    - 模板变更等于数据集变更，必须同步重新生成指令数据

依赖: advisory.schema
"""
