"""Stage 2 编排：结构化感知结果 -> 给司机的最终提示。

职责:
    - 串起 policy（决策）-> prompt（渲染）-> llm（生成）-> postprocess（清洗）
    - 保证降级路径：LLM 不可用时返回模板文案，绝不返回空
    - 保证安全兜底：LLM 生成内容与规则决策冲突时，以规则的保守结论为准
    - 是 Stage 2 对外的唯一入口

依赖: advisory.policy, advisory.prompt, advisory.llm
被谁调用: inference.pipeline
"""
