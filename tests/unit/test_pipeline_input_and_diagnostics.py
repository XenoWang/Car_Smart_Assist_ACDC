"""Batch API, fail-closed routing, CLI output protection and pause diagnostics."""

from argparse import Namespace
from types import SimpleNamespace
import json
import sys

import numpy as np
import pytest
import yaml

from _model_test_helpers import ROOT, load_module, snapshot
from car_smart_assist.advisory.generator import AdvisoryGenerator
from car_smart_assist.advisory.schema import PerceptionResult, RiskLevel
from car_smart_assist.inference.pipeline import InferencePipeline, PipelineResult


class Predictor:
    def __init__(self):
        self.images = []

    def predict(self, image):
        self.images.append(image.copy())
        return PerceptionResult(visibility_level="visible", road_obstacle_detection_available=True)


def test_debug_without_scorer_keeps_perception_in_policy_but_requests_takeover():
    predictor = Predictor()
    result = InferencePipeline(predictor=predictor).run(np.zeros((8, 8, 3), np.uint8))
    assert len(predictor.images) == 1
    assert result.perception.visibility_level == "unknown"
    assert result.advisory.source == "policy"
    assert result.advisory.should_takeover is True
    assert result.advisory.risk_level is RiskLevel.UNKNOWN
    assert "visibility_unavailable" in result.advisory.policy_details["risk"]["triggered"]


def test_batch_rejects_single_array_and_preserves_four_dimensional_batch_order():
    predictor = Predictor()
    pipeline = InferencePipeline(predictor=predictor)
    with pytest.raises(ValueError, match="run_batch"):
        pipeline.run_batch(np.zeros((8, 8, 3), np.uint8))
    assert predictor.images == []
    batch = np.stack([np.full((3, 4, 3), 11, np.uint8), np.full((3, 4, 3), 22, np.uint8)])
    results = pipeline.run_batch(batch)
    assert len(results) == 2 and len(predictor.images) == 2
    np.testing.assert_array_equal(predictor.images[0], np.full((3, 4, 3), 11, np.uint8))
    np.testing.assert_array_equal(predictor.images[1], np.full((3, 4, 3), 22, np.uint8))
    assert pipeline.run_batch([]) == []


@pytest.mark.parametrize("device", ["0", "gpu", ""])
def test_invalid_device_is_an_input_error_before_model_loading(tmp_path, device):
    from car_smart_assist.config.detection import resolve_yolo_device

    with pytest.raises(ValueError, match="device|设备"):
        InferencePipeline.from_config({}, project_root=tmp_path, device=device)
    with pytest.raises(ValueError, match="device"):
        resolve_yolo_device(device)


@pytest.mark.parametrize("state", ["missing", "disabled", "available"])
def test_configured_joint_route_never_silently_falls_back(tmp_path, monkeypatch, state):
    from car_smart_assist.config import detection
    from car_smart_assist.perception.detection_yolo import YoloDetectionPredictor

    folder = tmp_path / "configs/model"
    folder.mkdir(parents=True)
    for name in ("yolo_detection.yaml", "yolo_acdc_baseline.yaml"):
        (folder / name).write_text("configured route", encoding="utf-8")
    (tmp_path / "baseline.pt").write_bytes(b"available old detector")
    if state != "missing":
        (tmp_path / "joint.pt").write_bytes(b"available selected detector")
    monkeypatch.setattr(detection, "load_yolo_config", lambda path:
                        {"checkpoint": "joint.pt" if path.name == "yolo_detection.yaml" else "baseline.pt",
                         "enabled_for_inference": state != "disabled" if path.name == "yolo_detection.yaml" else True})
    calls, selected = [], object()
    monkeypatch.setattr(YoloDetectionPredictor, "from_config", lambda cfg, _root:
                        (calls.append(cfg["checkpoint"]), selected)[1])
    pipeline = InferencePipeline.from_config({}, project_root=tmp_path, require_visibility=False)
    if state == "available":
        assert calls == ["joint.pt"] and pipeline.predictor is selected
    else:
        assert calls == [] and pipeline.predictor is None
        expected_path = tmp_path / "joint.pt" if state == "missing" else folder / "yolo_detection.yaml"
        assert str(expected_path) in pipeline.predictor_unavailable_reason


def test_gate_failure_retains_elapsed_time_and_safe_takeover(monkeypatch):
    from car_smart_assist.inference import pipeline as module

    class BrokenScorer:
        def score_arrays(self, _images):
            raise ValueError("controlled gate failure")

    readings = iter([10.0, 10.02, 10.03, 10.04])
    monkeypatch.setattr(module, "time", SimpleNamespace(perf_counter=lambda: next(readings)))
    predictor = Predictor()
    result = InferencePipeline(scorer=BrokenScorer(), predictor=predictor).run(np.zeros((8, 8, 3), np.uint8))
    assert result.timings_ms["gate"] == pytest.approx(20.0)
    assert result.timings_ms["total"] == pytest.approx(30.0)
    assert result.advisory.should_takeover is True and predictor.images == []
    assert "perception" in result.skipped


def test_text_summary_includes_a_later_frames_failure(tmp_path, monkeypatch, capsys):
    cli = load_module("scripts/run_pipeline.py", "pipeline_summary_cli")
    (tmp_path / "scripts").mkdir()
    (tmp_path / "config.yaml").write_text("visibility: {}\n", encoding="utf-8")
    monkeypatch.setattr(cli, "__file__", str(tmp_path / "scripts/run_pipeline.py"))
    monkeypatch.setattr(cli, "parse_args", lambda: Namespace(
        config="config.yaml", checkpoint=None, device=None, weather_config=None,
        weather_loss="focal", weather_checkpoint=None, image=["a.png", "b.png"],
        synthetic=False, json=False, log_level="ERROR"))
    fallback = AdvisoryGenerator().from_unavailable("synthetic input")
    results = [PipelineResult(advisory=fallback),
               PipelineResult(advisory=fallback, skipped={"weather": "second-frame-failure"}),
               PipelineResult(advisory=fallback)]
    monkeypatch.setattr(cli.InferencePipeline, "from_config", lambda *_a, **_k:
                        SimpleNamespace(run_batch=lambda images: results))
    assert cli.main() == 0
    output = capsys.readouterr().out
    assert "b.png" in output and "second-frame-failure" in output


def test_between_model_pause_clears_the_completed_active_model(tmp_path):
    import scripts.train_pipeline as pipeline

    state = {"active_model": "weather", "stages": [
        {"name": "weather", "status": "COMPLETED"}, {"name": "visibility", "status": "PENDING"}]}
    assert pipeline.execute(tmp_path, state, SimpleNamespace(requested=True)) == 75
    saved = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert saved["status"] == "PAUSED" and saved["active_model"] is None
    assert saved["stages"][1]["status"] == "PENDING"


@pytest.mark.parametrize("independent", [False, True])
def test_candidate_val_requires_independent_output_before_models_or_report_writes(
    tmp_path, monkeypatch, capsys, independent,
):
    cli = load_module("scripts/evaluate_detection.py", "candidate_val_cli")
    (tmp_path / "scripts").mkdir()
    (tmp_path / "candidate.pt").write_bytes(b"candidate weights")
    protected = tmp_path / "reports/best_class_threshold_v1"
    protected.mkdir(parents=True)
    (protected / "val.json").write_bytes(b"protected validation evidence")
    before = snapshot(protected)
    monkeypatch.setattr(cli, "__file__", str(tmp_path / "scripts/evaluate_detection.py"))
    monkeypatch.setattr(cli, "load_yolo_config", lambda _p: {
        "checkpoint": "default.pt", "calibration_file": "reports/calibration.json", "data": "data/data.yaml"})
    args = ["evaluate_detection.py", "--config", "unused.yaml", "--checkpoint", "candidate.pt", "--split", "val"]
    if independent:
        args += ["--output-dir", "candidate-results"]
    monkeypatch.setattr(sys, "argv", args)

    class ValidatedOutputBoundary(Exception):
        pass

    def boundary(*_args, **_kwargs):
        raise ValidatedOutputBoundary

    monkeypatch.setattr(cli, "detection_data_scope", boundary)
    if independent:
        with pytest.raises(ValidatedOutputBoundary):
            cli.main()
    else:
        assert cli.main() == 2
        message = capsys.readouterr().err
        assert "--output-dir" in message and "val" in message
    assert snapshot(protected) == before


def test_unknown_training_option_is_rejected_before_creating_a_run(tmp_path, monkeypatch, capsys):
    cli = load_module("scripts/train_detection.py", "training_key_cli")
    cfg = yaml.safe_load((ROOT / "configs/model/yolo_detection.yaml").read_text(encoding="utf-8"))["yolo_detection"]
    cfg["train"]["epohcs"] = 3
    cfg.update(project="runs", name="candidate")
    (tmp_path / "scripts").mkdir()
    monkeypatch.setattr(cli, "__file__", str(tmp_path / "scripts/train_detection.py"))
    monkeypatch.setattr(cli, "load_yolo_config", lambda _p: cfg)
    monkeypatch.setattr(sys, "argv", ["train_detection.py", "--config", "unused.yaml"])
    monkeypatch.setenv("YOLO_CONFIG_DIR", str(tmp_path / "settings"))
    assert cli.main() == 2
    assert "epohcs" in capsys.readouterr().err
    assert not (tmp_path / "runs").exists()


def test_pipeline_cli_reports_invalid_device_without_traceback(tmp_path, monkeypatch, capsys, caplog):
    cli = load_module("scripts/run_pipeline.py", "pipeline_device_cli")
    (tmp_path / "scripts").mkdir()
    (tmp_path / "config.yaml").write_text("visibility: {}\n", encoding="utf-8")
    monkeypatch.setattr(cli, "__file__", str(tmp_path / "scripts/run_pipeline.py"))
    monkeypatch.setattr(sys, "argv", ["run_pipeline.py", "--config", "config.yaml", "--device", "0"])
    assert cli.main() == 2
    captured = capsys.readouterr()
    assert "设备" in caplog.text
    assert "Traceback" not in captured.out + captured.err + caplog.text
