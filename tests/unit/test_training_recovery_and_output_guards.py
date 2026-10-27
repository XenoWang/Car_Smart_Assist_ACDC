"""Training recovery, stop handling and deployment output protection."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[2]


def _load_helper(relative: str, name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_failed_stage_with_zero_checkpoints_restarts_only_that_stage(tmp_path, monkeypatch):
    pipeline_tests = _load_helper(
        "tests/unit/test_training_pipeline_control.py", "pipeline_recovery_helpers"
    )
    import scripts.train_pipeline as pipeline

    root = pipeline_tests._workspace(tmp_path, monkeypatch)
    run_dir = Path("artifacts/training/pipeline")
    work = root / run_dir / "current"
    args = [
        "--fresh", "--run-dir", str(run_dir),
        "--weather-config", "configs/model/weather.yaml",
        "--visibility-config", "configs/model/visibility.yaml",
        "--detection-config", "configs/model/detection.yaml",
    ]
    first = pipeline_tests._Spawner(work, [0, 2], no_checkpoint_indices={1})
    monkeypatch.setattr(pipeline.subprocess, "Popen", first)
    assert pipeline.main(args) == 2
    state = json.loads((work / "state.json").read_text(encoding="utf-8"))
    assert state["stages"][0]["status"] == "COMPLETED"
    assert state["stages"][1]["status"] == "FAILED"
    assert state["stages"][2]["status"] == "PENDING"
    assert all(record["sha256"] is None for record in state["stages"][1]["checkpoints"].values())

    resumed = pipeline_tests._Spawner(work, [0, 0, 0])
    monkeypatch.setattr(pipeline.subprocess, "Popen", resumed)
    assert pipeline.main(["--continue", *args[1:]]) == 0
    assert len(resumed.calls) == 3
    assert "--continue" not in resumed.calls[1][0]
    assert all("--continue" not in call[0] for call in resumed.calls)
    final = json.loads((work / "state.json").read_text(encoding="utf-8"))
    assert [stage["status"] for stage in final["stages"]] == [
        "COMPLETED", "COMPLETED", "COMPLETED", "COMPLETED"
    ]


def test_failed_detection_with_five_empty_records_restarts_detection_only(tmp_path, monkeypatch):
    pipeline_tests = _load_helper(
        "tests/unit/test_training_pipeline_control.py", "detection_recovery_pipeline_helpers"
    )
    import scripts.train_pipeline as pipeline

    root = pipeline_tests._workspace(tmp_path, monkeypatch)
    run_dir = Path("artifacts/training/pipeline")
    work = root / run_dir / "current"
    args = [
        "--fresh", "--run-dir", str(run_dir),
        "--weather-config", "configs/model/weather.yaml",
        "--visibility-config", "configs/model/visibility.yaml",
        "--detection-config", "configs/model/detection.yaml",
    ]
    first = pipeline_tests._Spawner(work, [0, 0, 0, 2], no_checkpoint_indices={3})
    monkeypatch.setattr(pipeline.subprocess, "Popen", first)
    assert pipeline.main(args) == 2
    state = json.loads((work / "state.json").read_text(encoding="utf-8"))
    detection = state["stages"][3]
    assert detection["name"] == "detection" and detection["status"] == "FAILED"
    assert len(detection["checkpoints"]) == 5
    assert all(record["sha256"] is None for record in detection["checkpoints"].values())
    protected = {
        name: {
            path: (path.read_bytes(), path.stat().st_mtime_ns)
            for path in pipeline.model_checkpoints(work, name).values()
        }
        for name in ("weather", "visibility")
    }

    resumed = pipeline_tests._Spawner(work, [0])
    monkeypatch.setattr(pipeline.subprocess, "Popen", resumed)
    assert pipeline.main(["--continue", *args[1:]]) == 0
    assert len(resumed.calls) == 1
    assert "train_detection.py" in resumed.calls[0][0][3]
    assert "--continue" not in resumed.calls[0][0]
    for name, checkpoints in protected.items():
        assert {
            path: (path.read_bytes(), path.stat().st_mtime_ns)
            for path in pipeline.model_checkpoints(work, name).values()
        } == checkpoints


def _flatten(value, prefix=""):
    if isinstance(value, dict):
        return {
            key: leaf
            for name, child in value.items()
            for key, leaf in _flatten(child, f"{prefix}.{name}" if prefix else name).items()
        }
    return {prefix: value}


def test_weather_train_recipe_matches_a85_and_uses_only_independent_output_paths():
    base = yaml.safe_load((ROOT / "configs/model/weather_attention_a.yaml").read_text(encoding="utf-8"))[
        "weather_attention"
    ]
    candidate = yaml.safe_load(
        (ROOT / "configs/model/weather_attention_train.yaml").read_text(encoding="utf-8")
    )["weather_attention"]
    before, after = _flatten(base), _flatten(candidate)
    assert before.keys() == after.keys()
    changed = {key for key in before if before[key] != after[key]}
    assert changed == {"checkpoint_root", "report_dir", "report_file"}
    default = yaml.safe_load((ROOT / "configs/model/weather_default.yaml").read_text(encoding="utf-8"))[
        "weather_default"
    ]
    assert default["config"] == "configs/model/weather_attention_a.yaml"


def test_benchmark_records_deployment_calibration_path_with_version_directory():
    import yaml as yaml_module

    from scripts.benchmark_pipeline import _artifact_paths
    from car_smart_assist.perception.yolo_postprocess import YOLO_POSTPROCESS_VERSION

    args = argparse.Namespace(
        config="configs/model/visibility.yaml", checkpoint=None,
        weather_config=None, weather_checkpoint=None, weather_loss="asl",
    )
    visibility = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))["visibility"]
    paths = _artifact_paths(args, visibility, yaml_module)
    detection = yaml.safe_load(
        (ROOT / "configs/model/yolo_detection.yaml").read_text(encoding="utf-8")
    )["yolo_detection"]
    assert paths["yolo_detection_calibration"] == (
        ROOT
        / "artifacts/reports/detection/yolo11n_acdc_laf_v2"
        / YOLO_POSTPROCESS_VERSION
        / "calibration.json"
    ).resolve()
    assert paths["yolo_detection_calibration"].parent.name == YOLO_POSTPROCESS_VERSION
    assert paths["yolo_detection_calibration"].parent.parent == (
        ROOT / Path(detection["calibration_file"]).parent
    ).resolve()


@pytest.mark.parametrize("output_dir", [None, "artifacts/reports/deployment"])
def test_candidate_calibration_rejects_deployment_output_before_model_or_file_changes(
    tmp_path, monkeypatch, output_dir
):
    import scripts.evaluate_detection as evaluate

    root = tmp_path
    (root / "scripts").mkdir()
    fake_script = root / "scripts/evaluate_detection.py"
    fake_script.write_text("", encoding="utf-8")
    monkeypatch.setattr(evaluate, "__file__", str(fake_script))
    cfg = {
        "checkpoint": "artifacts/checkpoints/deployment.pt",
        "calibration_file": "artifacts/reports/deployment/calibration.json",
    }
    from car_smart_assist.perception.yolo_postprocess import calibration_path

    default_checkpoint = root / "artifacts/checkpoints/deployment.pt"
    default_checkpoint.parent.mkdir(parents=True)
    default_checkpoint.write_bytes(b"deployment checkpoint sentinel")
    calibration = calibration_path(root, cfg)
    calibration.parent.mkdir(parents=True)
    calibration.write_bytes(b"deployment calibration sentinel")
    candidate = root / "candidate.pt"
    candidate.write_bytes(b"synthetic candidate weights")
    monkeypatch.setattr(evaluate, "load_yolo_config", lambda _path: cfg)
    monkeypatch.setattr(
        evaluate, "load_yolo_model",
        lambda *_args, **_kwargs: pytest.fail("candidate model was loaded before output guard"),
    )
    cli = ["evaluate_detection.py", "--split", "calibration", "--checkpoint", str(candidate)]
    if output_dir is not None:
        cli.extend(["--output-dir", output_dir])
    monkeypatch.setattr("sys.argv", cli)
    before = calibration.read_bytes()
    assert evaluate.main() == 2
    assert calibration.read_bytes() == before
    assert default_checkpoint.read_bytes() == b"deployment checkpoint sentinel"
    assert candidate.read_bytes() == b"synthetic candidate weights"


def test_visibility_evaluation_split_mismatch_shows_explicit_matching_checkpoint_command(
    tmp_path, monkeypatch, caplog
):
    import logging
    from types import SimpleNamespace

    import numpy as np
    import scripts.evaluate_visibility as evaluate

    root = tmp_path
    fake_script = root / "scripts/evaluate_visibility.py"
    fake_script.parent.mkdir(parents=True)
    fake_script.write_text("", encoding="utf-8")
    monkeypatch.setattr(evaluate, "__file__", str(fake_script))
    checkpoint = root / "synthetic-mismatched.pt"
    checkpoint.write_bytes(b"synthetic checkpoint placeholder")
    config = root / "configs/visibility.yaml"
    config.parent.mkdir(parents=True)
    config.write_text(yaml.safe_dump({"visibility": {
        "data": {"split_mode": "fixed", "acdc_root": "synthetic-acdc"},
        "train": {"checkpoint_dir": "checkpoints"},
        "model": {}, "scoring": {}, "report": {"output_dir": "reports", "sample_per_condition": 1},
    }}), encoding="utf-8")
    loaded = []
    monkeypatch.setattr(
        evaluate.VisibilityScorer,
        "from_checkpoint",
        lambda path, **_kwargs: loaded.append(path) or SimpleNamespace(
            input_size=(32, 32), calibration=None, scoring_cfg={},
        ),
    )
    monkeypatch.setattr(
        evaluate.VisibilityGate,
        "from_config",
        lambda _cfg: SimpleNamespace(thresholds=SimpleNamespace(info_blind=0.2)),
    )
    monkeypatch.setattr(evaluate, "list_ref_images", lambda *_args, **_kwargs: [checkpoint])
    monkeypatch.setattr(evaluate, "load_images", lambda *_args, **_kwargs: [
        np.zeros((32, 32, 3), dtype=np.uint8)
    ])

    def reject_split(*_args, **_kwargs):
        raise evaluate.SplitIdentityError("synthetic split identity mismatch")

    monkeypatch.setattr(evaluate, "eval_fit_quality", reject_split)
    for name in ("eval_clean_control", "eval_real_adverse", "eval_synthetic", "write_report"):
        monkeypatch.setattr(
            evaluate, name, lambda *_args, _name=name, **_kwargs: pytest.fail(
                f"{_name} ran after split identity rejection"
            ),
        )
    monkeypatch.setattr(
        "sys.argv",
        ["evaluate_visibility.py", "--config", "configs/visibility.yaml", "--checkpoint", str(checkpoint)],
    )
    caplog.set_level(logging.ERROR, logger=evaluate.__name__)
    assert evaluate.main() == 2
    assert loaded == [checkpoint]
    assert any("--checkpoint" in row.message for row in caplog.records)
    assert any(
        "artifacts/checkpoints/retrain_300_20261017/visibility/best.pt" in row.message
        for row in caplog.records
    )
    assert not (root / "reports").exists()


def test_stop_received_during_execute_identity_check_creates_no_attempt_or_child(
    tmp_path, monkeypatch
):
    import scripts.train_pipeline as pipeline
    from car_smart_assist.utils.stop import StopController

    work = tmp_path / "run/current"
    work.mkdir(parents=True)
    controller = StopController(tmp_path / "run/STOP", default_directory=tmp_path / "run")
    state = {
        "status": "PAUSED",
        "stages": [{
            "name": "visibility", "status": "PAUSED", "attempts": [], "checkpoints": {},
            "command": ["python", "train_visibility.py"],
        }],
    }

    def identity_check(_state, _work):
        controller.request_stop_file()

    monkeypatch.setattr(pipeline, "verify_identity", identity_check)
    monkeypatch.setattr(
        pipeline.subprocess, "Popen",
        lambda *_args, **_kwargs: pytest.fail("STOP during identity check launched a child"),
    )
    assert pipeline.execute(work, state, controller) == pipeline.STOP_EXIT_CODE
    assert state["status"] == "PAUSED"
    assert state["stages"][0]["attempts"] == []
    assert (tmp_path / "run/STOP").is_file()


@pytest.mark.parametrize("action", ["--continue", "--fresh"])
def test_refusing_unconfirmed_old_child_preserves_stop_bytes_and_mtime(
    tmp_path, monkeypatch, action
):
    pipeline_tests = _load_helper(
        "tests/unit/test_training_pipeline_control.py", "terminal_state_pipeline_helpers"
    )
    import scripts.train_pipeline as pipeline

    root = pipeline_tests._workspace(tmp_path, monkeypatch)
    run_dir = Path("artifacts/training/pipeline")
    work = root / run_dir / "current"
    work.mkdir(parents=True)
    state = {
        "stages": [{"attempts": [{"child_terminal_confirmed": False}]}],
    }
    (work / "state.json").write_text(json.dumps(state), encoding="utf-8")
    stop = root / run_dir / "STOP"
    stop.parent.mkdir(parents=True, exist_ok=True)
    stop.write_bytes(b"preserve-old-stop-marker")
    before = (stop.read_bytes(), stop.stat().st_mtime_ns)
    monkeypatch.setattr(
        pipeline.subprocess, "Popen",
        lambda *_args, **_kwargs: pytest.fail("refused queue launched a child"),
    )
    with pytest.raises(ValueError, match="terminal state is unconfirmed"):
        pipeline.main([
            action, "--run-dir", str(run_dir),
            "--weather-config", "configs/model/weather.yaml",
            "--visibility-config", "configs/model/visibility.yaml",
            "--detection-config", "configs/model/detection.yaml",
        ])
    assert (stop.read_bytes(), stop.stat().st_mtime_ns) == before
    assert not (root / run_dir / "history").exists()


def _visibility_queue_state(work: Path, journal_record):
    from scripts.train_pipeline import model_checkpoints

    checkpoints = {
        name: {"path": str(path), "sha256": None}
        for name, path in model_checkpoints(work, "visibility").items()
    }
    checkpoints["transaction_journal"] = journal_record
    return {
        "format_version": 1,
        "sources": {},
        "configs": {},
        "status": "PAUSED",
        "stages": [{
            "name": "visibility", "status": "PAUSED", "attempts": [{"old": True}],
            "checkpoints": checkpoints,
            "command": ["python", "train_visibility.py"],
        }],
    }


def test_recorded_visibility_journal_allows_continue_without_last_checkpoint(
    tmp_path, monkeypatch
):
    import scripts.train_pipeline as pipeline

    monkeypatch.setattr(pipeline, "source_identity", lambda: {})
    work = tmp_path / "run/current"
    work.mkdir(parents=True)
    journal = pipeline.model_checkpoints(work, "visibility")["last"].parent
    journal = journal / ".visibility_checkpoint_transaction/journal.json"
    journal.parent.mkdir(parents=True)
    journal.write_text('{"synthetic":"pending"}', encoding="utf-8")
    record = {"path": str(journal), "sha256": pipeline.digest(journal)}
    state = _visibility_queue_state(work, record)
    calls = []

    class Child:
        pid = 401

        def wait(self):
            return pipeline.STOP_EXIT_CODE

    def spawn(command, **_kwargs):
        calls.append(list(command))
        return Child()

    monkeypatch.setattr(pipeline.subprocess, "Popen", spawn)
    from car_smart_assist.utils.stop import StopController

    controller = StopController(tmp_path / "run/STOP", default_directory=tmp_path / "run")
    assert pipeline.execute(work, state, controller) == pipeline.STOP_EXIT_CODE
    assert len(calls) == 1 and "--continue" in calls[0]
    assert state["stages"][0]["status"] == "PAUSED"


def test_unrecorded_visibility_journal_is_rejected_and_absent_optional_null_is_allowed(
    tmp_path, monkeypatch
):
    import scripts.train_pipeline as pipeline

    monkeypatch.setattr(pipeline, "source_identity", lambda: {})
    work = tmp_path / "run/current"
    work.mkdir(parents=True)
    journal = pipeline.model_checkpoints(work, "visibility")["last"].parent
    journal = journal / ".visibility_checkpoint_transaction/journal.json"
    journal.parent.mkdir(parents=True)
    journal.write_text('{"synthetic":"unrecorded"}', encoding="utf-8")
    state = _visibility_queue_state(work, None)
    with pytest.raises(ValueError, match="Unrecorded visibility checkpoint transaction"):
        pipeline.verify_identity(state, work)

    journal.unlink()
    optional_null = {"path": str(journal), "sha256": None}
    state = _visibility_queue_state(work, optional_null)
    pipeline.verify_identity(state, work)
