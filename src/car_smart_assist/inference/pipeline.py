"""端到端推理管线：图像 -> 司机提示。

职责:
    - 串起 preprocess -> Stage1 感知 -> postprocess -> Stage2 生成
    - 管理模型常驻与批处理，避免每帧重复加载
    - 记录每帧耗时（预处理/感知/生成分开计），供性能说明用
    - 提供 CLI 与最小 demo 接口，方便录演示视频

依赖: inference.predictor, advisory.generator
被谁调用: scripts 与 notebooks 中的 demo
"""
