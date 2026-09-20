"""统一 CLI 入口（argparse / typer）。

职责:
    - 子命令: prepare / train-perception / train-advisory / eval / infer / export
    - 解析参数并转调对应的 scripts 或 engine，保持自身非常薄
    - 统一 --config / --seed / --device / --log-level 这类全局选项
    - 负责把异常转成非零退出码，方便 CI 判定
"""
