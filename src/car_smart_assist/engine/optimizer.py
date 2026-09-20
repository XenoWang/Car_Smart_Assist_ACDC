"""优化器与参数分组。

职责:
    - 按配置构建 AdamW / SGD
    - 参数分组：骨干网与头用不同学习率、weight decay 不施加在 norm 与 bias 上
    - 支持冻结参数的过滤（requires_grad=False 的不进优化器）
"""
