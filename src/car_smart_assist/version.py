"""包版本号的唯一来源。

职责:
    - 定义 __version__，供 pyproject.toml (dynamic version) 与运行时读取
    - 单一事实来源：不要在别处硬编码版本字符串
"""
