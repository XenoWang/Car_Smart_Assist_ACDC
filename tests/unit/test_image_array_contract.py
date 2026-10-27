"""Invalid image units/values and unchanged canonical pixels and model inputs."""

import numpy as np
import pytest
import torch
from PIL import Image

from car_smart_assist.advisory.schema import PerceptionResult
from car_smart_assist.inference.pipeline import InferencePipeline, _to_array
from car_smart_assist.perception.visibility.scorer import (
    CalibrationStats, VisibilityScorer, compute_information_features,
)


INVALID_ARRAYS = [
    pytest.param(np.float32, np.nan, ValueError, "NaN 或 Inf", id="nonfinite-float"),
    pytest.param(np.float64, 128.0, ValueError, "outside.*value_range", id="undeclared-pixel-units"),
    pytest.param(np.uint16, 128, ValueError, "explicit value_range", id="uint16-without-range"),
    pytest.param(np.int16, 128, ValueError, "explicit value_range", id="int16-without-range"),
    pytest.param(np.bool_, True, TypeError, "real integers", id="boolean"),
    pytest.param(np.complex64, 1j, TypeError, "real integers", id="complex"),
    pytest.param(object, 128, TypeError, "real integers", id="object"),
]


class RecordingIdentity(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inputs = []

    def forward(self, value):
        self.inputs.append(value.detach().cpu().clone())
        return value


class RecordingPredictor:
    def __init__(self):
        self.inputs = []

    def predict(self, image):
        self.inputs.append(image.copy())
        return PerceptionResult(visibility_level="visible")


@pytest.mark.parametrize("dtype,value,exception,message", INVALID_ARRAYS)
def test_array_entry_points_reject_invalid_dtype_units_or_values(dtype, value, exception, message):
    image = np.full((16, 16, 3), value, dtype=dtype)
    original = image.copy()
    with pytest.raises(exception, match=message):
        _to_array(image)
    with pytest.raises(exception, match=message):
        compute_information_features(image)
    np.testing.assert_array_equal(image, original)


@pytest.mark.parametrize("dtype,value,exception,message", INVALID_ARRAYS)
@pytest.mark.parametrize("operation", ["score_arrays", "calibrate"])
def test_invalid_mixed_batch_is_rejected_before_model_or_calibration_changes(
    dtype, value, exception, message, operation,
):
    model = RecordingIdentity()
    calibration = CalibrationStats(0.25, 0.5, 0.5, 0.75, 8)
    scorer = VisibilityScorer(model, calibration, input_size=(16, 16))
    images = [np.zeros((16, 16, 3), np.uint8), np.full((16, 16, 3), value, dtype)]
    originals = [image.copy() for image in images]
    with pytest.raises(exception, match=message):
        getattr(scorer, operation)(images)
    assert model.inputs == []
    assert scorer.calibration is calibration
    for image, original in zip(images, originals, strict=True):
        np.testing.assert_array_equal(image, original)


def test_uint8_alternating_columns_match_independent_feature_values():
    image = np.zeros((16, 16, 3), np.uint8)
    image[:, 1::2] = 255
    features = compute_information_features(image)
    # Half black/white: std=.5, entropy=1 bit, Nyquist-only non-DC energy.
    # Central differences cancel inside; only 2 of 16 columns have an edge.
    assert features.contrast == pytest.approx(0.5, abs=1e-6)
    assert features.entropy == pytest.approx(1.0, abs=1e-6)
    assert features.edge_density == pytest.approx(0.125, abs=1e-6)
    assert features.hf_ratio == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_uint8_model_inputs_keep_literal_normalization_and_zero_reconstruction(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("Requires CUDA")
    image = np.tile(np.array([0, 85, 170, 255], np.uint8)[None, :, None], (16, 4, 3))
    original = image.copy()
    model = RecordingIdentity()
    scorer = VisibilityScorer(model, CalibrationStats(0, 1, 0, 0, 2),
                              device=device, input_size=(16, 16))
    score = scorer.score_arrays([image])[0]
    # Precomputed float32 rounding of the agreed 0, 1/3, 2/3, 1 units.
    expected = torch.tensor([0.0, 0.3333333432674408, 0.6666666865348816, 1.0])
    assert len(model.inputs) == 1
    assert model.inputs[0].shape == (1, 3, 16, 16)
    assert torch.equal(model.inputs[0], expected.view(1, 1, 1, 4).repeat(1, 3, 16, 4))
    assert score.recon_mean == 0.0 and score.recon_p90_block == 0.0 and score.recon_z == 0.0
    stats = scorer.calibrate([image, image])
    assert stats.to_dict() == {"mean": 0.0, "std": 0.0, "p95": 0.0, "p99": 0.0, "n": 2}
    np.testing.assert_array_equal(image, original)


def test_pipeline_reports_undeclared_float_units_but_processes_legal_frame():
    predictor = RecordingPredictor()
    pipeline = InferencePipeline(predictor=predictor)
    valid = np.full((16, 16, 3), 85, np.uint8)
    invalid = np.full((16, 16, 3), 128.0, np.float32)
    results = pipeline.run_batch([invalid, valid])
    assert len(results) == 2
    assert "value_range" in results[0].skipped["input"]
    assert results[0].advisory.should_takeover is True
    assert results[0].perception is None
    assert "input" not in results[1].skipped
    assert len(predictor.inputs) == 1
    np.testing.assert_array_equal(predictor.inputs[0], valid)


@pytest.mark.parametrize("source_kind", ["gray-array", "rgba-array", "pil", "file"])
def test_uint8_and_decoded_images_keep_pixels_and_copy_isolation(tmp_path, source_kind):
    rgb = np.full((16, 16, 3), (7, 11, 13), np.uint8)
    if source_kind == "gray-array":
        source = np.full((16, 16), 85, np.uint8)
        expected = np.full((16, 16, 3), 85, np.uint8)
    elif source_kind == "rgba-array":
        source = np.concatenate([rgb, np.full((16, 16, 1), 255, np.uint8)], axis=-1)
        expected = rgb.copy()
    elif source_kind == "pil":
        source, expected = Image.fromarray(rgb), rgb.copy()
    else:
        source, expected = tmp_path / "frame.png", rgb.copy()
        Image.fromarray(rgb).save(source)
    actual = _to_array(source)
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.uint8 and actual.flags.c_contiguous
    actual[:] = 0
    np.testing.assert_array_equal(_to_array(source), expected)
