"""Precision boundary for the supported native Ultralytics Detect head."""
from contextlib import ExitStack, contextmanager

import torch


def _float_box_logits(module, inputs, output):
    return output.float()


@contextmanager
def detection_compute_precision(backend, half):
    """Bind this backend's compute dtype and keep native Detect decoding in FP32."""
    if bool(backend.fp16) != half:
        raise RuntimeError("YOLO backend did not honor the requested inference precision")
    if not half:
        yield
        return

    from ultralytics.nn.modules.head import Detect

    if not backend.pt or torch.device(backend.device).type != "cuda":
        raise ValueError("FP16 detection requires a native PyTorch CUDA backend")
    head = backend.model.model[-1]
    if type(head) is not Detect or head.end2end or head.export or head.xyxy:
        raise ValueError("FP16 decoding requires the standard raw xywh Detect head")
    if next(backend.model.parameters()).dtype != torch.float16:
        raise RuntimeError("YOLO FP16 convolution weights were not installed")
    if getattr(head, "_project_fp32_decode", False):
        raise RuntimeError("Use a separately loaded detection model for each precision")

    # Detect.forward concatenates the box and class logits before its native decoder.
    # Promoting the box towers' outputs promotes that concatenation and make_anchors.
    with ExitStack() as cleanup:
        def restore_decoder():
            head.dfl.to(dtype=next(head.cv2.parameters()).dtype)
            head.shape = None
            del head._project_fp32_decode

        head._project_fp32_decode = True
        cleanup.callback(restore_decoder)
        head.shape = None
        head.dfl.float()
        for tower in head.cv2:
            cleanup.callback(tower.register_forward_hook(_float_box_logits).remove)
        yield


def require_fp32_decoded(predictions):
    raw = predictions[0] if isinstance(predictions, (tuple, list)) and predictions else predictions
    if not isinstance(raw, torch.Tensor) or raw.dtype != torch.float32:
        raise RuntimeError("YOLO FP16 mode requires FP32 decoded boxes and scores")
