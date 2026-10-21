from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest
import torch
import yaml

from car_smart_assist.inference.pipeline import InferencePipeline
from car_smart_assist.perception.visibility.dataset import list_ref_images
from car_smart_assist.perception.visibility.gate import VisibilityLevel


ROOT = Path(__file__).resolve().parents[2]
WEATHER_CASES = (
    (
        "attention",
        "configs/model/weather_attention_a.yaml",
        "artifacts/checkpoints/weather_attention_ablation/reduced_tuning/best.pt",
    ),
    (
        "pyramid",
        "configs/model/weather_pyramid.yaml",
        "artifacts/checkpoints/weather_pyramid/asl/best.pt",
    ),
    (
        "legacy",
        None,
        "artifacts/checkpoints/weather_attributes/best.pt",
    ),
)
DEVICES = ("cpu", "cuda:0")
PROTECTED = (
    "configs/model/visibility.yaml",
    "data/processed/manifests/acdc.json",
    "configs/model/weather_attention_a.yaml",
    "configs/model/weather_pyramid.yaml",
    "configs/model/weather_classifier.yaml",
    "configs/model/yolo_detection.yaml",
    "configs/model/weather_default.yaml",
    "artifacts/checkpoints/visibility/best.pt",
    "artifacts/checkpoints/weather_attention_ablation/reduced_tuning/best.pt",
    "artifacts/checkpoints/weather_pyramid/asl/best.pt",
    "artifacts/checkpoints/weather_attributes/best.pt",
    "artifacts/runs/detection/yolo11n_acdc_laf_v2/weights/best.pt",
    "artifacts/reports/detection/yolo11n_acdc_laf_v2/best_class_threshold_v1/calibration.json",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _protected_hashes() -> dict[str, str]:
    missing = [relative for relative in PROTECTED if not (ROOT / relative).is_file()]
    assert not missing, f"required local model/config/calibration assets missing: {missing}"
    return {relative: _sha256(ROOT / relative) for relative in PROTECTED}


def _model_device(model) -> torch.device:
    parameter = next(model.parameters(), None)
    assert parameter is not None, "expected a loaded model with parameters"
    return parameter.device


@pytest.mark.slow
@pytest.mark.parametrize("weather_name,weather_config,weather_checkpoint", WEATHER_CASES)
@pytest.mark.parametrize("device", DEVICES)
def test_pipeline_real_models_follow_explicit_device_and_preserve_inputs(
    weather_name, weather_config, weather_checkpoint, device, monkeypatch
):
    if device == "cuda:0" and not torch.cuda.is_available():
        pytest.skip("CUDA device coverage skipped: torch.cuda.is_available() is False; CPU cases remain required")

    expected_device = torch.device(device)
    before = _protected_hashes()
    visible_devices_before = os.environ.get("CUDA_VISIBLE_DEVICES")
    try:
        visibility_path = ROOT / "configs/model/visibility.yaml"
        visibility_cfg = yaml.safe_load(visibility_path.read_text(encoding="utf-8"))["visibility"]
        visibility_cfg_before = json.loads(json.dumps(visibility_cfg))
        build_pipeline = lambda: InferencePipeline.from_config(
            visibility_cfg,
            project_root=ROOT,
            device=device,
            weather_config=weather_config,
            weather_checkpoint=weather_checkpoint,
            require_visibility=True,
        )
        if weather_name == "legacy":
            default_route = ROOT / "configs/model/weather_default.yaml"
            original_is_file = Path.is_file
            with monkeypatch.context() as route_patch:
                route_patch.setattr(
                    Path,
                    "is_file",
                    lambda path: False if path == default_route else original_is_file(path),
                )
                pipeline = build_pipeline()
        else:
            pipeline = build_pipeline()

        assert pipeline.scorer is not None, "visibility model must be available"
        assert _model_device(pipeline.scorer.model) == expected_device
        assert pipeline.weather_predictor is not None, f"{weather_name} weather model must be available"
        assert _model_device(pipeline.weather_predictor.model) == expected_device
        assert pipeline.predictor is not None, "configured YOLO model must be available"
        yolo_device = "cpu" if device == "cpu" else "0"
        assert pipeline.predictor.device == yolo_device
        yolo_model = pipeline.predictor.model.model

        cleaning_manifest = visibility_cfg["data"]["cleaning_manifest"]
        normal_paths = list_ref_images(
            ROOT / visibility_cfg["data"]["acdc_root"],
            cleaning_manifest=cleaning_manifest,
            project_root=ROOT,
        )
        assert normal_paths, "cleaning-manifest-filtered real normal references are required"
        input_paths = normal_paths[:8]
        input_hashes = {str(path): _sha256(path) for path in input_paths}
        results = pipeline.run_batch([str(path) for path in input_paths])
        assert {str(path): _sha256(path) for path in input_paths} == input_hashes
        assert len(results) == min(8, len(normal_paths))
        live_results = [
            result for result in results
            if result.visibility is not None and result.visibility.level is not VisibilityLevel.BLIND
        ]
        assert live_results, "at least one real reference image must pass the visibility gate"
        for result in live_results:
            assert result.weather is not None, "weather branch must run on a non-BLIND frame"
            assert result.perception is not None, "YOLO branch must run on a non-BLIND frame"
            assert result.skipped == {}, f"unexpected skipped branch: {result.skipped}"
        for result in results:
            json.dumps(result.to_dict(), ensure_ascii=False, allow_nan=False)

        assert visibility_cfg == visibility_cfg_before, "pipeline construction mutated loaded visibility config"
        assert _model_device(yolo_model) == expected_device, "YOLO model parameters did not move to the requested device after prediction"
    finally:
        after = _protected_hashes()
        assert after == before, "model, config, or calibration SHA changed during inference"
        assert os.environ.get("CUDA_VISIBLE_DEVICES") == visible_devices_before, "pipeline changed CUDA visibility for subsequent calls"
