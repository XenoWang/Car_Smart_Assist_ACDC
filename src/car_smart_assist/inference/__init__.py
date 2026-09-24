"""推理子包。

对外只暴露 InferencePipeline —— 其余模块（predictor / postprocess）
是管线的内部组件，不应被外层直接调用。
"""

from car_smart_assist.inference.pipeline import InferencePipeline, PipelineResult

__all__ = ["InferencePipeline", "PipelineResult"]
