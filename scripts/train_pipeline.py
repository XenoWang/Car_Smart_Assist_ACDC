"""Serial model training with explicit stop, continue, and fresh actions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import yaml
from filelock import FileLock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from car_smart_assist.utils.stop import STOP_EXIT_CODE, StopController  # noqa: E402


def timestamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_bytes(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".pending")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if temporary.read_bytes() != payload:
            raise OSError(f"Write readback mismatch: {temporary}")
        os.replace(temporary, path)
        if path.read_bytes() != payload:
            raise OSError(f"Published readback mismatch: {path}")
    finally:
        temporary.unlink(missing_ok=True)


def save_state(work, state):
    state["updated_utc"] = timestamp()
    write_bytes(work / "state.json", (json.dumps(state, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))


def model_checkpoints(work, name):
    directory = work / "checkpoints" / name
    if name == "detection":
        directory = work / "runs" / "detection" / "weights"
    return {"last": directory / "last.pt", "best": directory / "best.pt", **(
        {"training_state": directory / "last_training_state.pt",
         "run_metadata": directory.parent / "run_metadata.json",
         "validation_summary": directory.parent / "validation_summary.json"}
        if name == "detection" else {}
    )}


def checkpoint_records(work, stage):
    if stage["name"] == "weather-preflight":
        return {}
    return {key: {"path": str(path), "sha256": digest(path) if path.is_file() else None}
            for key, path in model_checkpoints(work, stage["name"]).items()}


def source_identity():
    paths = [*(ROOT / "src").rglob("*.py"), *(ROOT / "scripts").rglob("*.py")]
    return {str(path): digest(path) for path in sorted(paths)}


def prepare(work, options):
    """Copy recipes while changing output locations only."""
    sources = {name: (ROOT / getattr(options, name + "_config")).resolve()
               for name in ("weather", "visibility", "detection")}
    configs = {name: yaml.safe_load(path.read_text(encoding="utf-8"))
               for name, path in sources.items()}
    weather = configs["weather"]["weather_attention"]
    # This queue uses the existing default attention route, not SWAD/extension experiments.
    if weather["train"].get("swad", {}).get("enabled") or "continuation" in weather["train"]:
        raise ValueError("Training pipeline requires a standard attention recipe, not a SWAD/extension recipe")
    weather["checkpoint_root"] = str(work / "checkpoints/weather")
    weather["report_dir"] = str(work / "reports/weather")
    weather["report_file"] = str(work / "reports/weather/REPORT.md")
    visibility = configs["visibility"]["visibility"]
    visibility.setdefault("train", {})["checkpoint_dir"] = str(work / "checkpoints/visibility")
    visibility.setdefault("report", {})["output_dir"] = str(work / "reports/visibility")
    detection = configs["detection"]["yolo_detection"]
    detection["project"] = str(work / "runs")
    detection["name"] = "detection"
    detection["checkpoint"] = str(work / "runs/detection/weights/best.pt")
    detection["calibration_file"] = str(work / "reports/detection/calibration.json")
    config_paths = {}
    for name, config in configs.items():
        path = work / "configs" / (name + ".yaml")
        write_bytes(path, yaml.safe_dump(config, allow_unicode=True, sort_keys=False).encode("utf-8"))
        config_paths[name] = path
    python = ROOT / ".venv/Scripts/python.exe"
    stages = []
    for name, script, config, arguments in (
        ("weather-preflight", "train_weather_attention.py", "weather", ["--preflight"]),
        ("weather", "train_weather_attention.py", "weather", []),
        ("visibility", "train_visibility.py", "visibility", []),
        ("detection", "train_detection.py", "detection", []),
    ):
        stages.append({"name": name, "status": "PENDING", "attempts": [],
                       "command": [str(python), "-X", "utf8", str(ROOT / "scripts" / script),
                                   "--config", str(config_paths[config]), *arguments]})
    state = {"format_version": 1, "status": "READY", "created_utc": timestamp(),
             "sources": source_identity(),
             "configs": {str(path): digest(path) for path in (*sources.values(), *config_paths.values())},
             "stages": stages, "publication": "not part of this training queue"}
    save_state(work, state)
    return state


def require_terminal_children(state):
    for stage in state.get("stages", []):
        for attempt in stage.get("attempts", []):
            if attempt.get("child_terminal_confirmed") is False:
                raise ValueError("Previous child terminal state is unconfirmed; preserve the run for manual inspection")


def verify_identity(state):
    require_terminal_children(state)
    if state.get("format_version") != 1:
        raise ValueError("Unsupported pipeline state")
    if state["sources"] != source_identity():
        raise ValueError("Training source identity changed; continue refused, preserve the run for review")
    for path, expected in state["configs"].items():
        if not Path(path).is_file() or digest(path) != expected:
            raise ValueError(f"Training config changed/missing; continue refused: {path}")
    for stage in state["stages"]:
        for record in stage.get("checkpoints", {}).values():
            path, expected = Path(record["path"]), record["sha256"]
            if (digest(path) if path.is_file() else None) != expected:
                raise ValueError(f"Recorded checkpoint changed/missing; continue refused: {path}")


def execute(work, state, controller):
    for stage in state["stages"]:
        if stage["status"] == "COMPLETED":
            continue
        if controller.requested:
            state.update(status="PAUSED", termination_reason="manual_stop")
            save_state(work, state)
            return STOP_EXIT_CODE
        verify_identity(state)
        command = list(stage["command"])
        if stage["name"] != "weather-preflight" and stage["attempts"]:
            records = stage.get("checkpoints", {})
            if records.get("last", {}).get("sha256") is not None:
                command.append("--continue")
            elif stage["status"] == "PAUSED" and records and all(
                record["sha256"] is None for record in records.values()
            ):
                # Exit75 before the first complete epoch has no saved progress to restore.
                # The isolated output is still empty: restart this untouched step, not the queue.
                pass
            else:
                raise ValueError(f"No paired last checkpoint for unfinished step: {stage['name']}")
        folder = work / "logs" / stage["name"] / timestamp()
        folder.mkdir(parents=True)
        attempt = {"command": command, "started_utc": timestamp(), "stdout": str(folder / "stdout.log"),
                   "stderr": str(folder / "stderr.log"), "child_terminal_confirmed": False}
        stage["attempts"].append(attempt)
        stage["status"] = "RUNNING"
        state.update(status="RUNNING", active_model=stage["name"])
        save_state(work, state)
        environment = dict(os.environ, CAR_SMART_ASSIST_STOP_FILE=str(controller.path.resolve()),
                           CAR_SMART_ASSIST_PIPELINE_CHILD="1",
                           PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1")
        process = None
        try:
            with (folder / "stdout.log").open("xb") as stdout, (folder / "stderr.log").open("xb") as stderr:
                process = subprocess.Popen(command, cwd=ROOT, env=environment,
                                           stdout=stdout, stderr=stderr, stdin=subprocess.DEVNULL,
                                           creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
                attempt["pid"] = process.pid
                save_state(work, state)
                code = process.wait()  # Wait for terminal completion, without progress polling.
        except BaseException as exc:
            attempt.update(error=f"{type(exc).__name__}: {exc}",
                           child_terminal_confirmed=process is None)
            if process is not None:
                try:
                    controller.request_stop_file()
                    attempt["cleanup_exit_code"] = process.wait()
                    attempt["child_terminal_confirmed"] = True
                except BaseException as cleanup_error:
                    attempt["cleanup_error"] = f"{type(cleanup_error).__name__}: {cleanup_error}"
            attempt["finished_utc"] = timestamp()
            stage["status"] = "FAILED"
            if attempt["child_terminal_confirmed"]:
                stage["checkpoints"] = checkpoint_records(work, stage)
            state.update(status="FAILED", termination_reason="child_wait_or_launch_error")
            save_state(work, state)
            raise
        attempt.update(exit_code=code, finished_utc=timestamp(), child_terminal_confirmed=True)
        stage["checkpoints"] = checkpoint_records(work, stage)
        if code == STOP_EXIT_CODE:
            stage["status"] = "PAUSED"
            state.update(status="PAUSED", termination_reason="manual_stop")
        elif code != 0:
            stage["status"] = "FAILED"
            state.update(status="FAILED", termination_reason=f"child_exit_{code}")
        else:
            if any(record["sha256"] is None for record in stage["checkpoints"].values()):
                stage["status"] = "FAILED"
                state.update(status="FAILED", termination_reason="missing_checkpoint")
                save_state(work, state)
                raise ValueError(f"Successful command left missing checkpoints: {stage['name']}")
            stage["status"] = "COMPLETED"
            state.update(status="PAUSED" if controller.requested else "RUNNING",
                         termination_reason="manual_stop" if controller.requested else None)
        save_state(work, state)
        if state["status"] == "PAUSED":
            return STOP_EXIT_CODE
        if state["status"] == "FAILED":
            return 2
    state.update(status="COMPLETED", active_model=None, termination_reason="budget_completed")
    save_state(work, state)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--stop", action="store_true", help="Request epoch-end pause; never launch training")
    modes.add_argument("--continue", dest="continue_run", action="store_true", help="Resume the saved queue and its last checkpoint")
    modes.add_argument("--fresh", action="store_true", help="Archive the previous run and train from configured initialization")
    parser.add_argument("--run-dir", default="artifacts/training/model_pipeline")
    parser.add_argument("--weather-config", default="configs/model/weather_attention_a.yaml")
    parser.add_argument("--visibility-config", default="configs/model/visibility.yaml")
    parser.add_argument("--detection-config", default="configs/model/yolo_detection.yaml")
    options = parser.parse_args(argv)
    directory = (ROOT / options.run_dir).resolve()
    base = (ROOT / "artifacts/training").resolve()
    protected = base / "full_800_20261022"
    if not directory.is_relative_to(base) or directory == base or directory.is_relative_to(protected):
        raise ValueError("Use a dedicated artifacts/training subdirectory; the paused 800-epoch run is protected")
    controller = StopController(directory / "STOP", default_directory=directory)
    if options.stop:
        controller.request_stop_file()
        print(f"STOP requested: {controller.path}; wait for the active epoch to save and exit")
        return 0
    directory.mkdir(parents=True, exist_ok=True)
    work = directory / "current"
    if not work.resolve().is_relative_to(directory) or work.is_symlink():
        raise ValueError("Pipeline current directory must remain inside its run directory")
    with FileLock(str(directory / "pipeline.lock"), timeout=0), controller:
        if options.fresh:
            if work.exists():
                if not (work / "state.json").is_file():
                    raise ValueError("Existing current directory has no pipeline state; refusing to archive unrelated files")
                previous = json.loads((work / "state.json").read_text(encoding="utf-8"))
                require_terminal_children(previous)
                archived = directory / "history" / timestamp()
                archived.parent.mkdir(parents=True, exist_ok=True)
                os.replace(work, archived)
            controller.resume()
            state = prepare(work, options)
        else:
            path = work / "state.json"
            if not path.is_file():
                raise ValueError("No saved training queue; use --fresh explicitly to start one")
            state = json.loads(path.read_text(encoding="utf-8"))
            verify_identity(state)
            for name in ("weather", "visibility", "detection"):
                controller.watch_directory(model_checkpoints(work, name)["last"].parent)
            controller.resume()
        def request_stop(_signum, _frame):
            controller.request_stop_file()
        for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
            signum = getattr(signal, name, None)
            if signum is not None:
                signal.signal(signum, request_stop)
        print(f"Training queue: {work / 'state.json'}; STOP: {controller.path}", flush=True)
        return execute(work, state, controller)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (Exception, KeyboardInterrupt) as exc:
        print(f"Training pipeline stopped: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(130 if isinstance(exc, KeyboardInterrupt) else 2)
