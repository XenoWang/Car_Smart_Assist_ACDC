"""语言模型后端抽象接口。

职责:
    - 定义统一 generate(prompt, images) -> text 接口，屏蔽本地模型与远端 API 的差异
    - 支持本地权重与 OpenAI 兼容 API 两种实现，便于本地显存不够时切换
    - 统一超时、重试、失败降级（模型挂了要退回规则模板的静态文案，不能静默失声）

依赖: advisory.llm.local_vlm
"""
