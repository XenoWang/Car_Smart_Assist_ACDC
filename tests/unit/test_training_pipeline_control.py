from __future__ import annotations

import json
from pathlib import Path
import pytest
import yaml

import scripts.train_pipeline as pipeline


def _workspace(tmp_path, monkeypatch):
    root = tmp_path / "project"
    (root / "configs/model").mkdir(parents=True)
    (root / "artifacts/training").mkdir(parents=True)
    (root / "configs/model/weather.yaml").write_text(
        yaml.safe_dump({"weather_attention": {"train": {}, "checkpoint_root": "old", "report_dir": "old"}}),
        encoding="utf-8",
    )
    (root / "configs/model/visibility.yaml").write_text(
        yaml.safe_dump({"visibility": {"train": {}, "report": {}}}), encoding="utf-8"
    )
    (root / "configs/model/detection.yaml").write_text(
        yaml.safe_dump({"yolo_detection": {"project": "old", "name": "old", "checkpoint": "old", "calibration_file": "old"}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(pipeline, "ROOT", root)
    monkeypatch.setattr(pipeline.signal, "signal", lambda *_args: None)
    return root


class _Child:
    def __init__(self, command, work, code, kwargs, write_checkpoints=True):
        self.command = command
        self.work = work
        self.code = code
        self.kwargs = kwargs
        self.write_checkpoints = write_checkpoints
        self.pid = 321

    def wait(self):
        script = Path(self.command[3]).name
        preflight = "--preflight" in self.command
        stage_name = {
            "train_weather_attention.py": "weather-preflight" if preflight else "weather",
            "train_visibility.py": "visibility",
            "train_detection.py": "detection",
        }[script]
        if stage_name != "weather-preflight" and self.code in (0, 75) and self.write_checkpoints:
            for path in pipeline.model_checkpoints(self.work, stage_name).values():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes((stage_name + path.name).encode("utf-8"))
        if self.code == 2:
            Path(self.kwargs["env"]["CAR_SMART_ASSIST_STOP_FILE"]).touch()
        return self.code


class _Spawner:
    def __init__(self, work, codes, *, no_checkpoint_indices=()):
        self.work = work
        self.codes = iter(codes)
        self.calls = []
        self.no_checkpoint_indices = set(no_checkpoint_indices)

    def __call__(self, command, **kwargs):
        self.calls.append((list(command), kwargs))
        index = len(self.calls) - 1
        return _Child(command, self.work, next(self.codes), kwargs,
                      write_checkpoints=index not in self.no_checkpoint_indices)


def test_fresh_queue_archives_previous_current_and_keeps_history(tmp_path, monkeypatch):
    root = _workspace(tmp_path, monkeypatch)
    run_dir = Path("artifacts/training/pipeline")
    work = root / run_dir / "current"
    first = _Spawner(work, [0, 0, 0, 0])
    monkeypatch.setattr(pipeline.subprocess, "Popen", first)
    assert pipeline.main(["--fresh", "--run-dir", str(run_dir), "--weather-config", "configs/model/weather.yaml",
                          "--visibility-config", "configs/model/visibility.yaml", "--detection-config", "configs/model/detection.yaml"]) == 0
    original_state = (work / "state.json").read_bytes()

    second = _Spawner(work, [0, 0, 0, 0])
    monkeypatch.setattr(pipeline.subprocess, "Popen", second)
    assert pipeline.main(["--fresh", "--run-dir", str(run_dir), "--weather-config", "configs/model/weather.yaml",
                          "--visibility-config", "configs/model/visibility.yaml", "--detection-config", "configs/model/detection.yaml"]) == 0

    archived = list((root / run_dir / "history").glob("*/state.json"))
    assert len(archived) == 1
    assert archived[0].read_bytes() == original_state
    assert len(second.calls) == 4


def test_pipeline_requires_exactly_one_action(tmp_path, monkeypatch):
    _workspace(tmp_path, monkeypatch)
    with pytest.raises(SystemExit) as exc:
        pipeline.main(["--stop", "--fresh", "--run-dir", "artifacts/training/pipeline"])
    assert exc.value.code == 2


def test_continue_resumes_only_unfinished_stage_and_completed_queue_is_noop(tmp_path, monkeypatch):
    root = _workspace(tmp_path, monkeypatch)
    run_dir = Path("artifacts/training/pipeline")
    work = root / run_dir / "current"
    paused = _Spawner(work, [0, 75])
    monkeypatch.setattr(pipeline.subprocess, "Popen", paused)
    args = ["--fresh", "--run-dir", str(run_dir), "--weather-config", "configs/model/weather.yaml",
            "--visibility-config", "configs/model/visibility.yaml", "--detection-config", "configs/model/detection.yaml"]
    assert pipeline.main(args) == 75
    state = json.loads((work / "state.json").read_text(encoding="utf-8"))
    assert state["status"] == "PAUSED"
    assert state["stages"][0]["status"] == "COMPLETED"
    assert state["stages"][1]["status"] == "PAUSED"
    assert len(paused.calls) == 2

    resumed = _Spawner(work, [0, 0, 0])
    monkeypatch.setattr(pipeline.subprocess, "Popen", resumed)
    assert pipeline.main(["--continue", *args[1:]]) == 0
    assert len(resumed.calls) == 3  # Preflight and completed steps are not repeated.
    assert "--continue" in resumed.calls[0][0]
    assert all("--continue" not in call[0] for call in resumed.calls[1:])
    assert resumed.calls[0][1]["cwd"] == root
    assert resumed.calls[0][1]["env"]["CAR_SMART_ASSIST_PIPELINE_CHILD"] == "1"
    assert Path(resumed.calls[0][1]["env"]["CAR_SMART_ASSIST_STOP_FILE"]).is_absolute()
    first_attempt = json.loads((work / "state.json").read_text(encoding="utf-8"))["stages"][1]["attempts"][1]
    assert Path(first_attempt["stdout"]).is_file() and Path(first_attempt["stderr"]).is_file()

    monkeypatch.setattr(pipeline.subprocess, "Popen", lambda *a, **k: pytest.fail("completed queue launched a child"))
    assert pipeline.main(["--continue", *args[1:]]) == 0


def test_paused_before_first_checkpoint_restarts_only_untouched_step(tmp_path, monkeypatch):
    root = _workspace(tmp_path, monkeypatch)
    run_dir = Path("artifacts/training/pipeline")
    work = root / run_dir / "current"
    args = ["--fresh", "--run-dir", str(run_dir), "--weather-config", "configs/model/weather.yaml",
            "--visibility-config", "configs/model/visibility.yaml", "--detection-config", "configs/model/detection.yaml"]
    first = _Spawner(work, [0, 75], no_checkpoint_indices={1})
    monkeypatch.setattr(pipeline.subprocess, "Popen", first)
    assert pipeline.main(args) == 75
    state = json.loads((work / "state.json").read_text(encoding="utf-8"))
    assert state["stages"][1]["status"] == "PAUSED"
    assert all(record["sha256"] is None for record in state["stages"][1]["checkpoints"].values())

    second = _Spawner(work, [0, 0, 0])
    monkeypatch.setattr(pipeline.subprocess, "Popen", second)
    assert pipeline.main(["--continue", *args[1:]]) == 0
    assert len(second.calls) == 3
    assert all("--continue" not in call[0] for call in second.calls)


def test_pipeline_failure_code_is_not_masked_by_stop_marker(tmp_path, monkeypatch):
    root = _workspace(tmp_path, monkeypatch)
    run_dir = Path("artifacts/training/pipeline")
    work = root / run_dir / "current"
    spawner = _Spawner(work, [0, 2])
    monkeypatch.setattr(pipeline.subprocess, "Popen", spawner)
    args = ["--fresh", "--run-dir", str(run_dir), "--weather-config", "configs/model/weather.yaml",
            "--visibility-config", "configs/model/visibility.yaml", "--detection-config", "configs/model/detection.yaml"]
    assert pipeline.main(args) == 2
    state = json.loads((work / "state.json").read_text(encoding="utf-8"))
    assert state["status"] == "FAILED"
    assert state["termination_reason"] == "child_exit_2"
    assert len(spawner.calls) == 2


def test_continue_refuses_source_or_recipe_identity_change(tmp_path, monkeypatch):
    root = _workspace(tmp_path, monkeypatch)
    (root / "src").mkdir()
    source = root / "src" / "model.py"
    source.write_text("frozen = True\n", encoding="utf-8")
    run_dir = Path("artifacts/training/pipeline")
    work = root / run_dir / "current"
    spawner = _Spawner(work, [0, 75])
    monkeypatch.setattr(pipeline.subprocess, "Popen", spawner)
    args = ["--fresh", "--run-dir", str(run_dir), "--weather-config", "configs/model/weather.yaml",
            "--visibility-config", "configs/model/visibility.yaml", "--detection-config", "configs/model/detection.yaml"]
    assert pipeline.main(args) == 75
    recipe = root / "configs/model/weather.yaml"
    recipe.write_text(recipe.read_text(encoding="utf-8") + "\n# changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="config changed/missing"):
        pipeline.main(["--continue", *args[1:]])
    assert len(spawner.calls) == 2
    recipe.write_text(yaml.safe_dump({"weather_attention": {"train": {}, "checkpoint_root": "old", "report_dir": "old"}}), encoding="utf-8")
    source.write_text("frozen = False\n", encoding="utf-8")
    with pytest.raises(ValueError, match="source identity changed"):
        pipeline.main(["--continue", *args[1:]])
    assert len(spawner.calls) == 2


def test_continue_refuses_changed_recorded_checkpoint(tmp_path, monkeypatch):
    root = _workspace(tmp_path, monkeypatch)
    run_dir = Path("artifacts/training/pipeline")
    work = root / run_dir / "current"
    spawner = _Spawner(work, [0, 75])
    monkeypatch.setattr(pipeline.subprocess, "Popen", spawner)
    args = ["--fresh", "--run-dir", str(run_dir), "--weather-config", "configs/model/weather.yaml",
            "--visibility-config", "configs/model/visibility.yaml", "--detection-config", "configs/model/detection.yaml"]
    assert pipeline.main(args) == 75
    checkpoint = pipeline.model_checkpoints(work, "weather")["last"]
    checkpoint.write_bytes(b"changed after queue recorded the checkpoint")

    with pytest.raises(ValueError, match="Recorded checkpoint changed/missing"):
        pipeline.main(["--continue", *args[1:]])
    assert len(spawner.calls) == 2


def test_child_wait_error_requests_stop_and_retains_terminal_uncertain_state(tmp_path, monkeypatch):
    root = _workspace(tmp_path, monkeypatch)
    run_dir = Path("artifacts/training/pipeline")
    work = root / run_dir / "current"
    started = []

    class BrokenChild:
        pid = 999

        def wait(self):
            started.append("wait")
            raise OSError("wait failed")

    def spawn(command, **kwargs):
        started.append((command, kwargs))
        return BrokenChild()

    monkeypatch.setattr(pipeline.subprocess, "Popen", spawn)
    args = ["--fresh", "--run-dir", str(run_dir), "--weather-config", "configs/model/weather.yaml",
            "--visibility-config", "configs/model/visibility.yaml", "--detection-config", "configs/model/detection.yaml"]
    with pytest.raises(OSError, match="wait failed"):
        pipeline.main(args)
    assert (work.parent / "STOP").is_file()
    state = json.loads((work / "state.json").read_text(encoding="utf-8"))
    assert state["status"] == "FAILED"
    attempt = state["stages"][0]["attempts"][0]
    assert "wait failed" in attempt["error"]
    assert attempt["pid"] == 999
    assert attempt["child_terminal_confirmed"] is False
    assert "wait failed" in attempt["cleanup_error"]
    assert Path(attempt["stdout"]).is_file() and Path(attempt["stderr"]).is_file()
    with pytest.raises(ValueError, match="terminal|终态"):
        pipeline.main(["--continue", *args[1:]])
    with pytest.raises(ValueError, match="terminal|终态"):
        pipeline.main(args)  # Fresh must not archive files while a child may still own them.
