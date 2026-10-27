"""Input-boundary checks grounded in docs/camera_inputs.md.

Recording models expose the exact RGB HWC pixels and RGB/255 tensors they
receive. These assertions prove input adaptation, not model accuracy.
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from car_smart_assist.advisory.schema import PerceptionResult
from car_smart_assist.data.image_input import ImageFrame, to_rgb_uint8
from car_smart_assist.inference.pipeline import InferencePipeline
from car_smart_assist.perception.detection_yolo import YoloDetectionPredictor
from car_smart_assist.perception.visibility.dataset import read_rgb_image
from car_smart_assist.perception.visibility.scorer import (
    CalibrationStats,
    InformationFeatures,
    VisibilityScorer,
    VisibilityScore,
    compute_information_features,
)
from car_smart_assist.perception.weather import WeatherPrediction
from car_smart_assist.perception.weather_attention import WeatherAttentionPredictor
from car_smart_assist.perception.weather_pyramid import prepare_pyramid_image


class CapturingPredictor:
    def __init__(self):
        self.images = []

    def predict(self, image):
        self.images.append(image.copy())
        return PerceptionResult(
            visibility_level="visible", object_detection_available=True,
            road_obstacle_detection_available=True,
        )


class FixedWeatherPredictor:
    def predict(self, _image):
        return WeatherPrediction((), {}, {}, {}, "controlled input-contract fixture")


class RecordingScorer:
    def __init__(self):
        self.images = []

    def score_arrays(self, images):
        self.images.extend(image.copy() for image in images)
        return [
            VisibilityScore(
                path=f"<recorded:{index}>", recon_mean=0.0, recon_p90_block=0.0,
                recon_z=float("nan"),
                features=InformationFeatures(
                    contrast=0.5, entropy=1.0, edge_density=0.0, hf_ratio=0.0
                ),
                information=1.0,
            )
            for index, _ in enumerate(images)
        ]


class ObservedPipeline:
    """Delegate to the real run_batch while recording the representations main supplied."""

    def __init__(self):
        self.predictor = CapturingPredictor()
        self.scorer = RecordingScorer()
        self.pipeline = InferencePipeline(
            scorer=self.scorer, predictor=self.predictor,
            weather_predictor=FixedWeatherPredictor(),
        )
        self.received_kinds = []

    def run_batch(self, images):
        for image in images:
            if isinstance(image, ImageFrame) and isinstance(image.data, Path):
                self.received_kinds.append("path-frame")
            elif isinstance(image, np.ndarray):
                self.received_kinds.append("array")
            else:
                self.received_kinds.append(type(image).__name__)
        return self.pipeline.run_batch(images)


def test_pipeline_single_and_batch_share_normalized_pixels_and_isolate_bad_frame():
    rgb = np.array([[[0, 85, 170], [255, 170, 85]]], np.uint8)
    bgr = ImageFrame(rgb[..., ::-1], pixel_format="BGR")
    predictor = CapturingPredictor()
    pipeline = InferencePipeline(predictor=predictor)
    single = pipeline.run(bgr)
    batch = pipeline.run_batch([rgb, b"not an encoded image", bgr])
    assert single.perception is not None
    assert len(batch) == 3 and len(predictor.images) == 3
    np.testing.assert_array_equal(predictor.images[0], rgb)
    np.testing.assert_array_equal(predictor.images[1], rgb)
    assert "input" in batch[1].skipped and batch[1].perception is None
    assert batch[1].advisory is not None
    assert batch[1].advisory.should_takeover is True
    assert "UnidentifiedImageError" in batch[1].skipped["input"]
    assert batch[1].visibility is None  # Invalid input is unavailable, not a fabricated black BLIND frame.
    np.testing.assert_array_equal(predictor.images[2], rgb)


def test_scorer_tensor_preparation_keeps_exact_cpu_uint8_divide_by_255_contract():
    frame = np.array([[[0, 1, 85], [170, 254, 255]]], np.uint8)
    scorer = object.__new__(VisibilityScorer)
    scorer.device = torch.device("cpu")
    actual = scorer._to_tensor_batch([frame])
    expected = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).to(torch.float32) / 255.0
    assert actual.dtype == torch.float32 and actual.shape == (1, 3, 1, 2)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


class RecordingIdentity(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inputs = []

    def forward(self, value):
        self.inputs.append(value.detach().clone())
        return value


def test_scorer_scores_and_calibrates_float_bgr_and_processed_12_bit_rgb_batch():
    canonical = np.tile(
        np.array([[[0, 85, 170], [255, 170, 85]], [[85, 170, 255], [170, 85, 0]]], np.uint8),
        (8, 8, 1),
    )
    unit_float = canonical.astype(np.float32) / 255.0
    bgr = ImageFrame(canonical[..., ::-1], pixel_format="BGR")
    # Widen before multiplying: 255*4095 exceeds a uint16 fixture container.
    rgb12_codes = (canonical.astype(np.uint32) * 4095 // 255).astype(np.uint16)
    rgb12 = ImageFrame(rgb12_codes, value_range=(0, 4095))
    float_before, bgr_before, rgb12_before = unit_float.copy(), bgr.data.copy(), rgb12.data.copy()
    model = RecordingIdentity()
    scorer = VisibilityScorer(model, device="cpu", input_size=(16, 16))
    representations = [unit_float, bgr, rgb12]
    scores = scorer.score_arrays(representations)
    expected = torch.from_numpy(np.stack([canonical] * 3)).permute(0, 3, 1, 2).float() / 255.0
    assert len(scores) == 3 and len(model.inputs) == 1
    torch.testing.assert_close(model.inputs[-1], expected, rtol=0.0, atol=0.0)
    assert [(item.recon_mean, item.recon_p90_block) for item in scores] == [(0.0, 0.0)] * 3
    assert scores[0].features.to_dict() == scores[1].features.to_dict() == scores[2].features.to_dict()

    calibration = scorer.calibrate(representations)
    assert calibration == CalibrationStats(mean=0.0, std=0.0, p95=0.0, p99=0.0, n=3)
    torch.testing.assert_close(model.inputs[-1], expected, rtol=0.0, atol=0.0)
    np.testing.assert_array_equal(unit_float, float_before)
    np.testing.assert_array_equal(bgr.data, bgr_before)
    np.testing.assert_array_equal(rgb12.data, rgb12_before)

    calls_before_empty = len(model.inputs)
    assert scorer.score_arrays([]) == []
    assert len(model.inputs) == calls_before_empty


def test_information_features_match_across_equivalent_rgb_and_bgr_inputs():
    rgb = np.array(
        [[[0, 0, 0], [255, 255, 255]], [[0, 85, 170], [255, 170, 85]]], np.uint8
    )
    bgr = ImageFrame(rgb[..., ::-1], pixel_format="BGR")
    rgb_features = compute_information_features(rgb)
    bgr_features = compute_information_features(bgr)
    assert rgb_features.to_dict() == bgr_features.to_dict()


def test_read_rgb_image_accepts_frame_and_closes_only_its_own_file_handle(tmp_path):
    expected = np.array([[[0, 85, 170], [255, 170, 85]]], np.uint8)
    frame = ImageFrame(expected[..., ::-1], pixel_format="BGR")
    np.testing.assert_array_equal(read_rgb_image(frame), expected)
    path = tmp_path / "camera.png"
    Image.fromarray(expected, mode="RGB").save(path)
    actual = read_rgb_image(path)
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.uint8 and actual.flags.c_contiguous


def test_weather_pyramid_preparation_uses_rgb_pixels_before_documented_letterbox():
    source = np.array([[[0, 85, 170], [255, 170, 85]]], np.uint8)
    bgr = ImageFrame(source[..., ::-1], pixel_format="BGR")
    actual = prepare_pyramid_image(bgr, (2, 2))
    # Aspect-preserving resize produces one content row; odd padding follows floor division.
    assert actual.shape == (2, 2, 3) and actual.dtype == np.uint8
    np.testing.assert_array_equal(actual[0], source[0])
    np.testing.assert_array_equal(actual[1], np.zeros((2, 3), np.uint8))


class _Column:
    def __init__(self, values):
        self.values = values

    def cpu(self):
        return self

    def tolist(self):
        return self.values


class _Boxes:
    xyxy = _Column([[1.0, 2.0, 3.0, 4.0]])
    conf = _Column([0.75])
    cls = _Column([0.0])


class _YoloResult:
    boxes = _Boxes()
    names = {0: "car"}


class _YoloModel:
    predictor = None

    def __init__(self):
        self.calls = []

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        return [_YoloResult()]


def test_yolo_adapter_receives_rgb_pil_pixels_from_bgr_frame(monkeypatch, tmp_path):
    from car_smart_assist.config import detection

    monkeypatch.setattr(detection, "resolve_yolo_device", lambda value: value)
    model = _YoloModel()
    cfg = {"device": "cpu", "inference": {"imgsz": 32, "iou": 0.5, "max_det": 10}}
    predictor = YoloDetectionPredictor.__new__(YoloDetectionPredictor)
    predictor.model = model
    predictor.cfg = cfg
    predictor.device = "cpu"
    predictor.confidence = 0.1
    predictor.class_thresholds = {"car": 0.1}
    predictor.predictor_class = type("ExpectedPredictor", (), {})
    predictor.model.predictor = None
    # Only the construction's calibration-derived predictor type is irrelevant here;
    # use a stable type so the adapter's input conversion and source argument remain observable.
    import car_smart_assist.perception.detection_yolo as yolo_module

    monkeypatch.setattr(yolo_module, "YOLO_AGNOSTIC_NMS", False)
    source = np.array([[[0, 85, 170], [255, 170, 85]]], np.uint8)
    output = predictor.predict(ImageFrame(source[..., ::-1], pixel_format="BGR"))
    pil_source = model.calls[0]["source"]
    assert isinstance(pil_source, Image.Image)
    np.testing.assert_array_equal(np.asarray(pil_source), source)
    assert output.object_detection_available and output.objects[0].category == "car"


def test_equivalent_rgb_bgr_and_rgba_representations_match_literal_rgb_pixels():
    rgb = np.array([[[0, 85, 170], [255, 170, 85]]], np.uint8)
    representations = [
        rgb,
        ImageFrame(rgb[..., ::-1], pixel_format="BGR"),
        ImageFrame(np.concatenate((rgb, np.full((1, 2, 1), 17, np.uint8)), axis=2), pixel_format="RGBA"),
    ]
    canonical = np.array([[[0, 85, 170], [255, 170, 85]]], np.uint8)
    for item in representations:
        np.testing.assert_array_equal(to_rgb_uint8(item), canonical)


class RecordingZeroWeather(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inputs = []

    def forward(self, value):
        self.inputs.append(value.detach().clone())
        return torch.zeros((value.shape[0], 4), dtype=value.dtype, device=value.device)


def test_weather_attention_predict_normalizes_the_expected_letterboxed_rgb_tensor():
    rgb = np.full((2, 2, 3), [0, 85, 170], np.uint8)
    bgr = ImageFrame(rgb[..., ::-1], pixel_format="BGR")
    cfg = SimpleNamespace(image_size=(32, 32), attributes=("fog", "rain", "snow", "night"))
    thresholds = {"fog": 0.6, "rain": 0.4, "snow": 0.5, "night": 0.7}
    model = RecordingZeroWeather()
    predictor = WeatherAttentionPredictor(model, cfg, thresholds, torch.device("cpu"))
    prediction = predictor.predict(bgr)

    expected = torch.tensor([0, 85, 170], dtype=torch.float32).reshape(1, 3, 1, 1)
    expected = expected.expand(1, 3, 32, 32) / 255.0
    assert len(model.inputs) == 1
    torch.testing.assert_close(model.inputs[0], expected, rtol=0.0, atol=0.0)
    assert prediction.probabilities == {name: 0.5 for name in cfg.attributes}
    assert prediction.decisions == {"fog": False, "rain": True, "snow": True, "night": False}
    assert prediction.attributes == ("rain", "snow")


def _rgb16_png():
    def chunk(kind, data):
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    raw = b"\x00" + struct.pack(">HHHHHH", 0, 32768, 65535, 65535, 0, 0)
    ihdr = struct.pack(">IIBBBBB", 2, 1, 16, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


def test_rgb16_png_bytes_and_path_reader_preserve_literal_channel_codes(tmp_path):
    encoded = _rgb16_png()
    path = tmp_path / "processed-rgb16.png"
    path.write_bytes(encoded)
    expected = np.array([[[0, 128, 255], [255, 0, 0]]], np.uint8)
    from_bytes = to_rgb_uint8(ImageFrame(encoded, value_range=(0, 65535)))
    from_path = read_rgb_image(ImageFrame(path, value_range=(0, 65535)))
    np.testing.assert_array_equal(from_bytes, expected)
    np.testing.assert_array_equal(from_path, expected)


def test_pipeline_cli_main_passes_explicit_range_to_real_batch_conversion(
    monkeypatch, tmp_path, capsys
):
    from _model_test_helpers import load_module

    cli = load_module("scripts/run_pipeline.py", "camera_input_pipeline_main")
    config = tmp_path / "visibility.yaml"
    config.write_text("visibility: {}\n", encoding="utf-8")
    image_path = tmp_path / "camera.png"
    image_path.write_bytes(_rgb16_png())
    observed = ObservedPipeline()
    monkeypatch.setattr(
        cli, "InferencePipeline", SimpleNamespace(from_config=lambda *args, **kwargs: observed)
    )
    monkeypatch.setattr(
        "sys.argv",
        ["run_pipeline.py", "--config", str(config), "--image", str(image_path),
         "--image-range", "0", "65535", "--json"],
    )

    assert cli.main() == 0
    capsys.readouterr()
    expected = np.array([[[0, 128, 255], [255, 0, 0]]], np.uint8)
    assert observed.received_kinds[0] == "path-frame"
    np.testing.assert_array_equal(observed.predictor.images[0], expected)
    assert observed.scorer.images[0].shape == (144, 256, 3)


def test_benchmark_main_uses_same_range_for_path_and_predecoded_array_modes(
    monkeypatch, tmp_path, capsys
):
    from _model_test_helpers import load_module
    from car_smart_assist.perception import weather as weather_module

    cli = load_module("scripts/benchmark_pipeline.py", "camera_input_benchmark_main")
    config = tmp_path / "visibility.yaml"
    config.write_text("visibility: {}\n", encoding="utf-8")
    image_path = tmp_path / "camera.png"
    image_path.write_bytes(_rgb16_png())
    output = tmp_path / "benchmark"
    observed = ObservedPipeline()
    monkeypatch.setattr(cli, "_artifact_paths", lambda *args: {})
    monkeypatch.setattr(cli, "_components", lambda _pipe: {})
    monkeypatch.setattr(cli, "_cuda_devices", lambda *args: [])
    monkeypatch.setattr(cli, "_sync", lambda *args: None)
    monkeypatch.setattr(cli, "_gpu_memory", lambda *args, **kwargs: {})
    monkeypatch.setattr(cli, "_ram", lambda: {"working_set_bytes": 1})
    monkeypatch.setattr(weather_module, "select_device", lambda _requested: torch.device("cpu"))
    monkeypatch.setattr(
        InferencePipeline,
        "from_config",
        classmethod(lambda cls, *args, **kwargs: observed),
    )

    status = cli.main(
        ["--config", str(config), "--image", str(image_path), "--image-range", "0", "65535",
         "--device", "cpu", "--batch-size", "1", "--warmup", "0", "--repeats", "1",
         "--output-dir", str(output)]
    )
    capsys.readouterr()
    expected = np.array([[[0, 128, 255], [255, 0, 0]]], np.uint8)
    assert status == 0
    assert "path-frame" in observed.received_kinds and "array" in observed.received_kinds
    assert len(observed.predictor.images) >= 2
    for pixels in observed.predictor.images:
        np.testing.assert_array_equal(pixels, expected)
    assert observed.scorer.images and all(pixels.shape == (144, 256, 3) for pixels in observed.scorer.images)
    for pixels in observed.scorer.images[1:]:
        np.testing.assert_array_equal(pixels, observed.scorer.images[0])
