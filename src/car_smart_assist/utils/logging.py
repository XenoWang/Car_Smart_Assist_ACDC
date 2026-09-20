"""日志配置。

职责:
    - 统一 logger 构造：控制台带颜色、文件带时间戳，输出到 artifacts/logs/
    - 多进程/多卡场景下避免重复 handler
    - 禁止在库代码里直接 print，统一走这里
"""
