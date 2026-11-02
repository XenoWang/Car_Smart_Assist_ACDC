"""Persistent pauses survive rejected restarts; only accepted old pauses clear."""
from contextlib import contextmanager
from dataclasses import replace
import os
import sys
import time

import pytest

from _model_test_helpers import ROOT, load_module
from car_smart_assist.utils.stop import STOP_EXIT_CODE, StopController


def marker_state(path):
    return path.read_bytes(), path.stat().st_mtime_ns


@pytest.mark.parametrize("marker", ["default", "explicit"])
def test_unaccepted_restart_preserves_original_pause(tmp_path, marker):
    default = tmp_path / "run"
    default.mkdir()
    path = default / "STOP" if marker == "default" else tmp_path / "custom-STOP"
    path.write_bytes(b"paused by user\n")
    before = marker_state(path)
    controller = StopController(path if marker == "explicit" else None, default_directory=default)
    controller.prepare_resume()
    controller.require_running()
    # A rejected restore never reaches acceptance, including fresh validation.
    assert marker_state(path) == before
    controller.accept_resume()
    assert not path.exists()
    assert not controller.requested


@pytest.mark.parametrize("arrival", ["before_prepare", "after_prepare", "replace_old"])
def test_new_pause_during_restart_is_never_cleared(tmp_path, arrival):
    path = tmp_path / "STOP"
    if arrival == "replace_old":
        path.write_bytes(b"old pause")
    controller = StopController(default_directory=tmp_path)
    if arrival != "before_prepare":
        controller.prepare_resume()
    path.write_bytes(b"new request during startup")
    if arrival == "before_prepare":
        controller.prepare_resume()
    with pytest.raises(SystemExit) as stopped:
        controller.accept_resume()
        controller.require_running()
    assert stopped.value.code == STOP_EXIT_CODE
    assert path.read_bytes() == b"new request during startup"


def test_pause_created_during_config_loading_is_not_an_old_pause(tmp_path):
    started = time.time_ns()
    path = tmp_path / "STOP"
    path.write_bytes(b"new config-time stop")
    os.utime(path, ns=(started + 1_000_000, started + 1_000_000))
    controller = StopController(default_directory=tmp_path, started_ns=started)
    controller.prepare_resume()
    with pytest.raises(SystemExit) as stopped:
        controller.require_running()
    assert stopped.value.code == STOP_EXIT_CODE
    assert path.read_bytes() == b"new config-time stop"


def test_child_restart_keeps_parent_pause(tmp_path, monkeypatch):
    parent, child = tmp_path / "parent-STOP", tmp_path / "child"
    child.mkdir()
    parent.write_bytes(b"parent pause")
    (child / "STOP").write_bytes(b"old child pause")
    monkeypatch.setenv("CAR_SMART_ASSIST_STOP_FILE", str(parent))
    monkeypatch.setenv("CAR_SMART_ASSIST_PIPELINE_CHILD", "1")
    controller = StopController(default_directory=child)
    controller.prepare_resume()
    with pytest.raises(SystemExit) as stopped:
        controller.accept_resume()
    assert stopped.value.code == STOP_EXIT_CODE
    assert parent.read_bytes() == b"parent pause"
    assert (child / "STOP").read_bytes() == b"old child pause"


@pytest.mark.parametrize("kind", ["visibility", "weather_attention", "detection"])
@pytest.mark.parametrize("mode,failure", [("continue", "missing"), ("continue", "invalid"), ("fresh", "invalid")])
def test_cli_rejected_continue_preserves_pause_bytes_and_time(tmp_path, monkeypatch, capsys, caplog, kind, mode, failure):
    module = load_module(f"scripts/train_{kind}.py", f"rejected_restart_{kind}")
    monkeypatch.setattr(module, "__file__", str(tmp_path / "scripts" / f"train_{kind}.py"))
    directory = tmp_path / ("runs/demo/weights" if kind == "detection" else "run")
    directory.mkdir(parents=True)
    marker = directory / "STOP"
    marker.write_bytes(b"persistent user pause\n")
    before = marker_state(marker)
    if failure == "invalid":
        (directory / "last.pt").write_bytes(b"invalid paired checkpoint")
    if kind == "visibility":
        monkeypatch.setattr(module, "load_visibility_config", lambda _: {"train": {"checkpoint_dir": "run"}})
        def reject_trainer(*args, **kwargs):
            raise ValueError("checkpoint rejected by controlled fixture")
        monkeypatch.setattr(module, "VisibilityTrainer", reject_trainer)
    elif kind == "weather_attention":
        from car_smart_assist.config.weather_attention import load_weather_attention_config
        cfg = replace(load_weather_attention_config(ROOT / "configs/model/weather_attention_a.yaml"),
                      checkpoint_root="run", report_dir="reports", report_file="reports/REPORT.md")
        monkeypatch.setattr(module, "load_weather_attention_config", lambda _: cfg)
        def reject_data(*args):
            raise ValueError("dataset rejected by controlled fixture")
        monkeypatch.setattr(module, "prepare_attention_datasets", reject_data)
    else:
        monkeypatch.setattr(module, "load_yolo_config", lambda _: {
            "name": "demo", "project": "runs", "train": {"epochs": 2, "seed": 42, "close_mosaic": 0},
            "data": "controlled", "model": "initial.pt", "checkpoint": "unused.pt",
            "inference": {"imgsz": 640, "confidence": .25, "iou": .5, "max_det": 100}})
        @contextmanager
        def data_scope(*args, **kwargs):
            yield "data.yaml", {"signature": "synthetic", "seed": 42, "classes": [{"name": "car", "source_id": 26}]}, {}
        monkeypatch.setattr(module, "detection_data_scope", data_scope)
        original_plan = module.resolve_training_plan
        if failure == "invalid":
            def reject_plan(*args, **kwargs):
                raise ValueError("paired checkpoint rejected by controlled fixture")
            monkeypatch.setattr(module, "resolve_training_plan", reject_plan)
        else:
            monkeypatch.setattr(module, "resolve_training_plan", original_plan)
    monkeypatch.setattr(sys, "argv", ["train", "--config", "unused.yaml", f"--{mode}"])
    assert module.main() == 2
    assert marker_state(marker) == before
    output = capsys.readouterr()
    combined = output.err + output.out + caplog.text
    if failure == "missing":
        assert "last" in combined
    else:
        assert "rejected by controlled fixture" in combined


def test_weather_accepted_restore_clears_old_pause_before_real_updates(tmp_path, monkeypatch):
    from test_training_stop_models import _attention_setup
    from car_smart_assist.utils.stop import StopController
    training, cfg, datasets, split, batches = _attention_setup(tmp_path, monkeypatch)
    training.train_attention(cfg, tmp_path, datasets, split, fresh=True)
    marker = tmp_path / "checkpoints/STOP"
    marker.write_bytes(b"old pause after completed stage")
    before_batches = len(batches)
    controller = StopController(default_directory=marker.parent)
    controller.prepare_resume()
    _, result = training.train_attention(cfg, tmp_path, datasets, split,
                                        stop_controller=controller, continue_only=True)
    assert not marker.exists()
    assert len(batches) == before_batches
    assert result["training_complete"] is True
    assert result["manual_stop"] is False


def test_yolo_accepts_old_pause_only_after_restore_validation(tmp_path):
    from test_yolo_training_state import trainer, SELECTION_CONTRACT, TRAINING_SEGMENT
    from car_smart_assist.perception.yolo_training import attach_training_state_callbacks
    from types import SimpleNamespace
    path = tmp_path / "STOP"
    path.write_bytes(b"old pause")
    controller = StopController(default_directory=tmp_path)
    controller.prepare_resume()
    callbacks = {}
    model = SimpleNamespace(add_callback=lambda event, callback: callbacks.update({event: callback}))
    target = trainer(tmp_path)
    # Build a schema-valid fixture; persisted STOP identity is the independent oracle.
    import torch
    attach_training_state_callbacks(model, "synthetic", None,
                                    selection_contract=SELECTION_CONTRACT,
                                    training_segment=TRAINING_SEGMENT, stop_controller=controller)
    callbacks["on_model_save"](target)
    state = torch.load(tmp_path / "last_training_state.pt", map_location="cpu", weights_only=True)
    target.start_epoch += 1
    attach_training_state_callbacks(model, "synthetic", state,
                                    selection_contract=SELECTION_CONTRACT,
                                    training_segment=TRAINING_SEGMENT, stop_controller=controller)
    before = marker_state(path)
    with pytest.raises(ValueError, match="Resume epoch"):
        callbacks["on_train_start"](target)
    assert marker_state(path) == before
    target.start_epoch -= 1
    attach_training_state_callbacks(model, "synthetic", None,
                                    selection_contract=SELECTION_CONTRACT,
                                    training_segment=TRAINING_SEGMENT, stop_controller=controller)
    callbacks["on_train_start"](target)
    assert not path.exists()
    assert not controller.requested


@pytest.fixture(autouse=True)
def isolated_random_state():
    import random
    import numpy as np
    import torch
    state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        yield
    finally:
        random.setstate(state[0])
        np.random.set_state(state[1])
        torch.set_rng_state(state[2])
        torch.set_num_threads(threads)

def test_visibility_accepted_completed_restore_clears_pause_without_new_epoch(tmp_path, monkeypatch):
    from test_visibility_best_identity import _config, _fit_one_epoch
    from car_smart_assist.perception.visibility.trainer import VisibilityTrainer
    original = _fit_one_epoch(tmp_path, monkeypatch)
    path = tmp_path / "checkpoints/STOP"
    path.write_bytes(b"old paused stage")
    controller = StopController(default_directory=path.parent)
    controller.prepare_resume()
    resumed = VisibilityTrainer(_config(tmp_path), project_root=tmp_path, resume="auto")
    monkeypatch.setattr(resumed, "_build_datasets", original._build_datasets)
    history = resumed.fit(stop_controller=controller, continue_only=True)
    assert not path.exists()
    assert history.training_complete is True
    assert history.manual_stop is False
    assert len(history.train_loss) == 1
