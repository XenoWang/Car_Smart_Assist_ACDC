"""Behavioral contracts for the three local inference optimizations."""

import numpy as np
import pytest
import torch
from PIL import Image

import car_smart_assist.inference.pipeline as pipeline
from car_smart_assist.data.image_input import ImageFrame
from car_smart_assist.perception.visibility.scorer import VisibilityScorer


def test_to_array_preserves_uint8_shape_pixels_and_copy_isolation():
    source = np.arange(4 * 6 * 4, dtype=np.uint8).reshape(4, 6, 4)
    expected = source[..., :3].copy()
    original = source.copy()

    result = pipeline._to_array(source)

    assert result.shape == (4, 6, 3)
    assert result.dtype == np.uint8
    assert result.flags.c_contiguous
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(source, original)
    result[0, 0, 0] = 255 - result[0, 0, 0]
    np.testing.assert_array_equal(source, original)


def test_to_array_repeats_uint8_grayscale_channels():
    source = np.array([[255, 0], [0, 255]], dtype=np.uint8)

    result = pipeline._to_array(source)

    expected = np.repeat(source.astype(np.uint8)[..., None], 3, axis=2)
    np.testing.assert_array_equal(result, expected)
    assert result.flags.c_contiguous


def test_to_array_does_not_scan_uint8_or_ordinary_pil_for_finiteness(monkeypatch):
    def unexpected_scan(*_args, **_kwargs):
        raise AssertionError("finite scan is unnecessary for integer, boolean, or RGB PIL input")

    monkeypatch.setattr(pipeline.np, "isfinite", unexpected_scan)
    for source in (
        np.zeros((2, 3), dtype=np.uint8),
        np.zeros((2, 3, 3), dtype=np.uint8),
        Image.new("RGB", (3, 2), color=(7, 11, 13)),
    ):
        result = pipeline._to_array(source)
        assert result.shape == (2, 3, 3)
        assert result.dtype == np.uint8


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_to_array_rejects_nonfinite_rgba_alpha(bad_value):
    source = np.zeros((2, 3, 4), dtype=np.float32)
    source[1, 2, 3] = bad_value

    with pytest.raises(ValueError, match="NaN 或 Inf"):
        pipeline._to_array(source)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf])
def test_float_pil_f_rejects_nonfinite_before_rgb_conversion(monkeypatch, bad_value):
    values = np.zeros((2, 3), dtype=np.float32)
    values[0, 1] = bad_value
    image = Image.fromarray(values)
    assert image.mode == "F"
    original_convert = Image.Image.convert
    conversion_calls = []

    def observe_conversion(self, *args, **kwargs):
        conversion_calls.append(args[0] if args else None)
        return original_convert(self, *args, **kwargs)

    monkeypatch.setattr(Image.Image, "convert", observe_conversion)
    with pytest.raises(ValueError, match="NaN 或 Inf"):
        pipeline._to_array(ImageFrame(image, value_range=(0, 255)))
    assert conversion_calls == []


@pytest.mark.parametrize(
    ("source", "exception", "message"),
    [
        (np.zeros((2,), dtype=np.uint8), ValueError, "HW"),
        (np.zeros((0, 3), dtype=np.uint8), ValueError, "非空"),
        (np.zeros((2, 3, 2), dtype=np.uint8), ValueError, "通道数"),
        (np.zeros((2, 3), dtype=object), TypeError, "real integers"),
        (np.array([[0.0, np.nan]], dtype=np.float32), ValueError, "NaN 或 Inf"),
    ],
)
def test_to_array_keeps_invalid_shape_dtype_and_float_failure_contract(
    source, exception, message
):
    with pytest.raises(exception, match=message):
        pipeline._to_array(source)


def _scorer_for_device(device):
    scorer = object.__new__(VisibilityScorer)
    scorer.device = torch.device(device)
    return scorer


def _old_float32_batch_reference(images):
    batch = np.stack(images, axis=0)
    return torch.from_numpy(batch).permute(0, 3, 1, 2).to(
        dtype=torch.float32, memory_format=torch.contiguous_format
    ).div_(255.0)


def _assert_tensor_batch_matches_old_reference(device):
    values = np.arange(16 * 16 * 3, dtype=np.uint16).reshape(16, 16, 3)
    all_values = (values % 256).astype(np.uint8)
    wide = np.arange(16 * 32 * 3, dtype=np.uint16).reshape(16, 32, 3)
    noncontiguous = (wide % 256).astype(np.uint8)[:, ::2, ::-1]
    assert not noncontiguous.flags.c_contiguous
    images = [noncontiguous, all_values, noncontiguous]
    snapshots = [image.copy() for image in images]
    expected = _old_float32_batch_reference(images)
    actual = _scorer_for_device(device)._to_tensor_batch(images)

    assert actual.shape == (3, 3, 16, 16)
    assert actual.dtype == torch.float32
    assert actual.is_contiguous()
    assert actual.device.type == torch.device(device).type
    if actual.device.type == "cuda":
        assert actual.device.index == torch.cuda.current_device()
    assert torch.equal(actual.cpu(), expected)
    for image, snapshot in zip(images, snapshots, strict=True):
        np.testing.assert_array_equal(image, snapshot)


def test_tensor_batch_cpu_matches_original_float32_path_exactly():
    _assert_tensor_batch_matches_old_reference("cpu")


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要可用 CUDA GPU")
def test_tensor_batch_cuda_preserves_every_uint8_value_exactly():
    _assert_tensor_batch_matches_old_reference("cuda")


@pytest.mark.parametrize("device", [
    "cpu",
    pytest.param("cuda", marks=[pytest.mark.gpu,
        pytest.mark.skipif(not torch.cuda.is_available(), reason="需要可用 CUDA GPU")]),
])
def test_pixel_unit_float_tensor_batch_requires_range_without_mutating_inputs(device):
    image = np.linspace(0.0, 255.0, 4 * 8 * 3, dtype=np.float32).reshape(4, 8, 3)
    inputs = [image[:, ::-1], image]
    snapshots = [value.copy() for value in inputs]
    with pytest.raises(ValueError, match="value_range"):
        _scorer_for_device(device)._to_tensor_batch(inputs)
    for value, snapshot in zip(inputs, snapshots, strict=True):
        np.testing.assert_array_equal(value, snapshot)
