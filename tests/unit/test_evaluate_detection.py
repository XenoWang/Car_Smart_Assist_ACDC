"""Evaluator control flow: two-pass calibration, test lock, and atomic reports."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from car_smart_assist.config.detection import YOLO_POSTPROCESS_VERSION


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "evaluate_detection.py"
SPEC = importlib.util.spec_from_file_location("evaluate_detection_under_test", SCRIPT)
assert SPEC and SPEC.loader
evaluate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluate)


class _Box:
    map50 = 0.5
    map = 0.25
    ap_class_index = np.array([0])
    all_ap = np.array([[0.5] * 10])
    px = np.array([0.03, 0.3])
    f1_curve = np.array([[0.2, 0.8]])


def _fake_backend(monkeypatch, output, calls):
    class Validator:
        pass

    monkeypatch.setitem(
        sys.modules,
        "car_smart_assist.perception.joint_yolo_backend",
        SimpleNamespace(DeploymentValidator=Validator, JointValidator=Validator),
    )

    class Model:
        names = {0: "car"}

        def val(self, **kwargs):
            validator = kwargs["validator"]
            mode = validator.postprocess_mode
            calls.append((kwargs["split"], mode, validator.locked_confidence))
            metrics = SimpleNamespace(box=_Box())
            if mode == evaluate.YOLO_LOCKED_MODE:
                metrics.operating_iou50 = {
                    "per_class": {"car": {"precision": 0.75, "recall": 0.6, "f1": 2 / 3}},
                    "macro": {"precision": 0.75, "recall": 0.6, "f1": 2 / 3},
                }
            return metrics

    def validate_export(path, verify_hashes=False, **_kwargs):
        assert verify_hashes is True

    from car_smart_assist.data import detection
    monkeypatch.setattr(detection, "validate_detection_export", validate_export)
    monkeypatch.setattr(evaluate, "load_yolo_model", lambda root, checkpoint: Model())
    monkeypatch.setattr(
        evaluate,
        "load_yolo_config",
        lambda path: {
            "checkpoint": "README.md",
            "data": "README.md",
            "calibration_file": "unused/calibration.json",
            "device": "cpu",
            "inference": {"imgsz": 32, "confidence": 0.1, "iou": 0.5, "max_det": 10},
            "train": {"batch": 1, "workers": 0},
        },
    )
    monkeypatch.setattr(evaluate, "calibration_identity", lambda *args, **kwargs: {"schema_version": 2})
    monkeypatch.setattr(evaluate, "resolve_yolo_device", lambda device: "cpu")
    original_read_text = Path.read_text

    def read_text(path, *args, **kwargs):
        if path.name == "manifest.json":
            return json.dumps({"classes": ["car"], "signature": "fixture", "images": []})
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_text)


def test_calibration_runs_proposal_then_locked_and_publishes_only_complete_payload(tmp_path, monkeypatch):
    calls = []
    _fake_backend(monkeypatch, tmp_path, calls)
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--split", "calibration", "--output-dir", str(tmp_path)])

    result = evaluate.main()

    assert result == 0
    assert [call[1] for call in calls] == [evaluate.YOLO_PROPOSAL_MODE, evaluate.YOLO_LOCKED_MODE]
    assert calls[1][2] == pytest.approx(0.3)
    calibration = tmp_path / YOLO_POSTPROCESS_VERSION / "calibration.json"
    payload = json.loads(calibration.read_text(encoding="utf-8"))
    assert payload["passes"] == [evaluate.YOLO_PROPOSAL_MODE, evaluate.YOLO_LOCKED_MODE]
    assert payload["class_confidence_thresholds"] == {"car": pytest.approx(0.3)}


def test_test_split_uses_locked_calibration_without_selecting_thresholds(tmp_path, monkeypatch):
    calls = []
    _fake_backend(monkeypatch, tmp_path, calls)
    calibration = tmp_path / YOLO_POSTPROCESS_VERSION / "calibration.json"
    calibration.parent.mkdir(parents=True)
    calibration.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(evaluate, "load_calibration", lambda *args, **kwargs: {
        "confidence_threshold": 0.42,
        "class_confidence_thresholds": {"car": 0.42},
        "class_threshold_sources": {"car": "calibration"},
    })
    monkeypatch.setattr(
        evaluate, "select_thresholds",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("test split selected thresholds")),
    )
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--split", "test", "--output-dir", str(tmp_path)])

    result = evaluate.main()

    assert result == 0
    assert [call[1] for call in calls] == [evaluate.YOLO_AP_MODE, evaluate.YOLO_LOCKED_MODE]
    assert calls[1][2] == pytest.approx(0.42)
    report = json.loads((tmp_path / YOLO_POSTPROCESS_VERSION / "test.json").read_text(encoding="utf-8"))
    assert report["threshold_source"] == "calibration"
    assert report["operating_macro_f1_iou50"] == pytest.approx(2 / 3)


def test_atomic_report_failure_preserves_existing_file(tmp_path, monkeypatch):
    path = tmp_path / "report.json"
    path.write_text('{"old": true}', encoding="utf-8")
    monkeypatch.setattr(evaluate.json, "dump", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("disk full")))

    with pytest.raises(OSError, match="disk full"):
        evaluate.atomic_json(path, {"new": True})

    assert path.read_text(encoding="utf-8") == '{"old": true}'
    assert list(tmp_path.iterdir()) == [path]
