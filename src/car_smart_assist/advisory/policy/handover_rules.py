"""接管决策规则引擎（可解释、可单测）。

职责:
    - 输入 PerceptionResult，输出「是否需要接管」及触发原因
    - 规则来源：接管边界头的分级输出 + 距离阈值 + 路况权重组合
    - 阈值必须可配置（configs/model/advisory_llm.yaml 或独立 policy 段），便于做敏感性分析
    - 纯函数、无副作用、不依赖 GPU —— 上层循环里最好跑的就是它

依赖: advisory.schema
被谁调用: advisory.generator
备注: 即使后续用 LLM 出文本，决策本身也应该由这一层给出，避免把安全性交给生成模型
"""
