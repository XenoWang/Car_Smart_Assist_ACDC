"""配置的数据结构定义与校验。

职责:
    - 用 dataclass / pydantic 定义 DataConfig / ModelConfig / TrainConfig / InferConfig
    - 字段带类型与默认值；非法组合在加载期就报错，而不是训练中途
    - 作为配置的唯一契约，改字段必须同步改这里
"""
