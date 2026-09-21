"""包版本号的唯一来源。

职责:
    - 定义 __version__，供 pyproject.toml (dynamic version) 与运行时读取
    - 单一事实来源：不要在别处硬编码版本字符串

注意:
    本变量被 pyproject.toml 的 [tool.setuptools.dynamic] 通过 attr 引用，
    **必须保持为字面量赋值**。setuptools 优先用 AST 静态读取，不做 import；
    如果改成函数调用或拼接（如 __version__ = ".".join(...)），
    静态读取会失败并回退到导入包，构建期就会出问题。
"""

__version__ = "0.0.1"
