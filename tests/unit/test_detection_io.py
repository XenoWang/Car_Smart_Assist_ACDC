"""Literal color/box oracles plus unchanged vendor preprocessing; no model downloads."""
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from car_smart_assist.data.image_input import ImageFrame
from car_smart_assist.perception.detection_yolo import YoloDetectionPredictor


def bare_adapter(predict):
    adapter = object.__new__(YoloDetectionPredictor)
    adapter.model = SimpleNamespace(predict=predict, predictor=None)
    adapter.device = "cpu"
    adapter.cfg = {"inference": {"imgsz": 64, "iou": 0.5, "max_det": 100}}
    adapter.confidence = 0.3
    adapter.class_thresholds = {"car": 0.3, "road_obstacle": 0.2}
    adapter.predictor_class = type("ExpectedPredictor", (), {})
    return adapter


def empty_boxes():
    box = SimpleNamespace(
        xyxy=np.empty((0, 4)), conf=np.empty(0), cls=np.empty(0),
    )
    box.cpu = lambda: box
    return box


@pytest.mark.parametrize("layout", ["contiguous", "readonly", "strided", "bgr_frame", "pil", "unit_float"])
def test_source_has_literal_bgr_pixels_and_independent_contiguous_storage(layout):
    rgb = np.array([[[0, 85, 170], [255, 170, 85]], [[12, 34, 56], [78, 90, 123]]], dtype=np.uint8)
    source = rgb
    if layout == "readonly":
        rgb.flags.writeable = False
    elif layout == "strided":
        padded = np.zeros((2, 4, 3), dtype=np.uint8)
        padded[:, ::2] = rgb
        rgb = padded[:, ::2]
        source = rgb
    elif layout == "bgr_frame":
        source = ImageFrame(rgb[..., ::-1], pixel_format="BGR")
    elif layout == "pil":
        source = Image.fromarray(rgb)
    elif layout == "unit_float":
        source = rgb.astype(np.float32) / 255
    before = rgb.copy()
    received = []

    def predict(**kwargs):
        bgr = kwargs["source"]
        received.append(bgr.copy())
        assert bgr.dtype == np.uint8 and bgr.flags.c_contiguous and bgr.flags.writeable
        assert not np.shares_memory(bgr, rgb)
        # Model-side writes must not corrupt the caller's frame.
        bgr[:] = 0
        return [SimpleNamespace(boxes=empty_boxes(), names={0: "car", 1: "road_obstacle"})]

    output = bare_adapter(predict).predict(source)
    expected = np.array([[[170, 85, 0], [85, 170, 255]], [[56, 34, 12], [123, 90, 78]]], dtype=np.uint8)
    np.testing.assert_array_equal(received[0], expected)
    np.testing.assert_array_equal(rgb, before)
    if layout == "unit_float":
        np.testing.assert_array_equal(source, before.astype(np.float32) / 255)
    assert output.objects == []
    assert output.object_detection_available and output.road_obstacle_detection_available


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_real_vendor_preprocessing_matches_original_pil_entry_bit_for_bit(device):
    import torch
    from ultralytics.data.loaders import LoadPilAndNumpy
    from ultralytics.models.yolo.detect import DetectionPredictor

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    rgb = np.arange(11 * 19 * 3, dtype=np.uint8).reshape(11, 19, 3)
    received = []
    def predict(**kwargs):
        received.append(kwargs["source"])
        return [SimpleNamespace(boxes=empty_boxes(), names={0: "car", 1: "road_obstacle"})]
    bare_adapter(predict).predict(rgb)
    old_loaded = LoadPilAndNumpy(Image.fromarray(rgb)).im0
    new_loaded = LoadPilAndNumpy(received[0]).im0
    np.testing.assert_array_equal(old_loaded[0], new_loaded[0])
    preprocessor = object.__new__(DetectionPredictor)
    preprocessor.device = torch.device(device)
    preprocessor.imgsz = [64, 64]
    preprocessor.args = SimpleNamespace(rect=True)
    preprocessor.model = SimpleNamespace(fp16=False, pt=True, stride=32)
    original = preprocessor.preprocess(old_loaded)
    optimized = preprocessor.preprocess(new_loaded)
    assert torch.equal(original, optimized)
    assert tuple(original.shape) == (1, 3, 64, 64)
    assert original.dtype == torch.float32


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("tracked", [False, True])
@pytest.mark.parametrize("empty", [False, True])
def test_real_boxes_transfer_once_preserves_exact_fields_and_order(device, tracked, empty):
    import torch
    from ultralytics.engine.results import Boxes

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    calls = []
    class CountedBoxes(Boxes):
        def cpu(self):
            calls.append(str(self.data.device))
            return super().cpu()

    rows = ([[1, 2, 30, 40, 77, 0.5, 1], [3, 4, 11, 19, 88, 0.75, 0]] if tracked else
            [[1, 2, 30, 40, 0.5, 1], [3, 4, 11, 19, 0.75, 0]])
    data = torch.tensor(rows, dtype=torch.float32, device=device)
    if empty:
        data = data[:0]
    before = data.clone()
    boxes = CountedBoxes(data, (50, 80))
    model_result = SimpleNamespace(boxes=boxes, names={0: "car", 1: "road_obstacle"})
    output = bare_adapter(lambda **kwargs: [model_result]).predict(np.zeros((50, 80, 3), np.uint8))
    assert len(calls) == 1  # Explicit requirement: one grouped host transfer.
    assert torch.equal(data, before)
    assert output.object_detection_classes == ("car", "road_obstacle")
    assert output.road_obstacle_detection_available
    if empty:
        assert output.objects == []
    else:
        assert [o.category for o in output.objects] == ["road_obstacle", "car"]
        assert [o.bbox for o in output.objects] == [(1., 2., 30., 40.), (3., 4., 11., 19.)]
        assert [o.confidence for o in output.objects] == [0.5, 0.75]
        assert all(o.distance_m is None and o.direction.value == "unknown" for o in output.objects)


def test_invalid_input_and_model_failure_keep_the_original_error_boundary():
    calls = []
    def failure(**kwargs):
        calls.append(kwargs)
        raise RuntimeError("inference sentinel")
    predictor = bare_adapter(failure)
    with pytest.raises(ValueError, match="通道数"):
        predictor.predict(np.zeros((4, 4, 2), np.uint8))
    assert calls == []
    with pytest.raises(RuntimeError, match="inference sentinel"):
        predictor.predict(np.zeros((4, 4, 3), np.uint8))
    assert len(calls) == 1
