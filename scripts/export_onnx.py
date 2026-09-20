"""把训练好的 Stage 1 模型导出为部署格式。

职责:
    - 导出 ONNX（opset 按 TensorRT 兼容性选），带 dynamic batch
    - 做导出后一致性校验：ONNX Runtime 输出 vs PyTorch 输出，逐 head 比对误差
    - 记录输入输出签名（shape / dtype / 预处理参数）到 artifacts/reports/onnx_signature.json
    - Stage 2 的语言模型不在这里导出

依赖: torch.onnx, onnxruntime
被谁调用: 人工执行；需要部署演示时
备注: 属于「加分项」，优先级低于把两个 Stage 的指标跑通
"""
