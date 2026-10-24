"""Eight explicit stop scenarios for the three main pipeline models.

1-2 weather attention: stop during first epoch, stop during epoch 5 of 15.
3-4 visibility AE: stop during first epoch, stop during epoch 5 of 15.
5-6 joint YOLO: stop during first epoch, stop during epoch 5 of 15.
7-8 pipeline: stop between weather/visibility and between visibility/detection.

Model scenarios use real (tiny) training loops; pipeline boundaries use the
existing controlled child-process fixture so no long training is launched.
Run with ``pytest -v -s`` to see each scenario line in the terminal.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import random
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from car_smart_assist.utils.stop import StopController

ROOT = Path(__file__).resolve().parents[2]
EPOCHS = 15
MID_STOP_EPOCH = 5


def _load(relative: str, name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Weather attention
# ---------------------------------------------------------------------------


def _attention_stop(tmp_path, monkeypatch, *, stop_after_batch: int, name: str):
    helper = _load("tests/unit/test_training_stop_models.py", f"stop8_attention_{name}")
    root = tmp_path / name
    root.mkdir(parents=True, exist_ok=True)
    controller = StopController(root / "STOP", default_directory=root)
    training, cfg, datasets, split, _ = helper._attention_setup(
        root, monkeypatch, stop_controller=controller
    )
    cfg = replace(cfg, train={**cfg.train, "epochs": EPOCHS, "schedule_epochs": EPOCHS,
                              "early_stopping": {"enabled": False}})

    def selection_for_stop_scenarios(*_args):
        return dict.fromkeys(cfg.attributes, 0.5), {
            "score": 0.5,
            "per_attribute": {name: {"source_balanced_ap": 0.5} for name in cfg.attributes},
        }

    # The original fixture has only three validation scores; this case can reach epoch 5.
    monkeypatch.setattr(training, "balanced_thresholds", selection_for_stop_scenarios)
    # Make the tiny training split two batches per epoch so "first epoch not
    # complete" is distinguishable from "last batch of the epoch".
    train_dataset = datasets["train"]
    train_dataset.samples = [SimpleNamespace() for _ in range(4)]
    train_dataset.targets = np.zeros((4, 4), dtype=np.float32)
    train_dataset.mask = np.ones((4, 4), dtype=np.float32)
    monkeypatch.setattr(training, "attention_sampling_weights", lambda *_a: torch.ones(4))
    original_batch_images = training.batch_images
    count = {"batches": 0}

    def batch_images(dataset, indices, device, options):
        values = original_batch_images(dataset, indices, device, options)
        count["batches"] += 1
        if count["batches"] == stop_after_batch:
            controller.request()
        return values

    monkeypatch.setattr(training, "batch_images", batch_images)
    _, result = training.train_attention(
        cfg, root, datasets, split, fresh=True, stop_controller=controller
    )
    last = torch.load(root / "checkpoints/last.pt", map_location="cpu", weights_only=True)
    return result, last, count["batches"]


def test_weather_stop_before_first_epoch_completes(tmp_path, monkeypatch):
    print("\n[SCENARIO 1] weather attention: STOP during first epoch (before it finishes)", flush=True)
    result, last, batches = _attention_stop(
        tmp_path, monkeypatch, stop_after_batch=1, name="weather_first"
    )
    assert result["manual_stop"] is True
    assert result["termination_reason"] == "manual_stop"
    assert result["trained_through_epoch"] == 1
    assert result["calibrated"] is False
    assert last["epoch"] == 1 and last["planned_epochs"] == EPOCHS
    assert len(last["history"]) == 1
    assert batches == 2  # Both batches of epoch 1 still complete before the stop is honored.
    print(f"[RESULT 1] paused at epoch {last['epoch']}, batches={batches}, calibrated=False", flush=True)


def test_weather_stop_mid_training_completes_current_epoch(tmp_path, monkeypatch):
    print(f"\n[SCENARIO 2] weather attention: STOP during epoch {MID_STOP_EPOCH} of {EPOCHS}", flush=True)
    result, last, batches = _attention_stop(
        tmp_path, monkeypatch, stop_after_batch=(MID_STOP_EPOCH - 1) * 2 + 1, name="weather_mid"
    )
    assert result["manual_stop"] is True
    assert result["trained_through_epoch"] == MID_STOP_EPOCH
    assert result["calibrated"] is False
    assert last["epoch"] == MID_STOP_EPOCH and last["planned_epochs"] == EPOCHS
    assert len(last["history"]) == MID_STOP_EPOCH
    assert batches == MID_STOP_EPOCH * 2  # Finish both batches of the requested epoch.
    print(f"[RESULT 2] paused at epoch {last['epoch']}, batches={batches}, history={len(last['history'])}", flush=True)


# ---------------------------------------------------------------------------
# Visibility AE
# ---------------------------------------------------------------------------


def _ae_stop(tmp_path, monkeypatch, *, stop_after_batch: int, name: str):
    from car_smart_assist.perception.visibility.trainer import VisibilityTrainer

    directory = tmp_path / name
    config = {
        "device": "cpu",
        "model": {"input_size": (16, 16), "base_channels": 2, "latent_dim": 4},
        "data": {"num_workers": 0},
        "train": {
            "epochs": EPOCHS, "batch_size": 2, "lr": 1e-3, "weight_decay": 0.0,
            "optimizer": "adamw", "amp": False, "dtype": "float32",
            "augment": False, "denoise_sigma": 0.0, "grad_clip_norm": 1.0,
            "seed": 23,
            "early_stopping": {"enabled": False, "patience": 4, "min_delta": 1e-5},
            "checkpoint_dir": str(directory),
        },
    }
    samples = torch.linspace(0, 1, 4 * 3 * 16 * 16).reshape(4, 3, 16, 16)

    class Rows:
        def __len__(self):
            return len(samples)

        def __getitem__(self, index):
            return samples[index]

    trainer = VisibilityTrainer(config, project_root=tmp_path, resume="none")
    dataset = Rows()
    monkeypatch.setattr(trainer, "_build_datasets", lambda: (dataset, dataset, dataset, []))
    values = iter(1.0 + 0.1 * epoch for epoch in range(EPOCHS))
    calibrations = []
    monkeypatch.setattr(trainer, "_eval_loss", lambda _loader: next(values))
    monkeypatch.setattr(trainer, "_calibrate", lambda _loader: calibrations.append(True))
    controller = StopController(directory / "STOP", default_directory=directory)
    count = {"batches": 0}

    def hook(module, _inputs, _output):
        if module.training:
            count["batches"] += 1
            if count["batches"] == stop_after_batch:
                controller.request()

    handle = trainer.model.register_forward_hook(hook)
    random.seed(23);
    np.random.seed(23);
    torch.manual_seed(23)
    history = trainer.fit(stop_controller=controller)
    handle.remove()
    last = torch.load(directory / "last.pt", map_location="cpu", weights_only=False)
    return history, last, calibrations, count["batches"]


def test_visibility_stop_before_first_epoch_completes(tmp_path, monkeypatch):
    print("\n[SCENARIO 3] visibility AE: STOP during first epoch (before it finishes)", flush=True)
    history, last, calibrations, batches = _ae_stop(
        tmp_path, monkeypatch, stop_after_batch=1, name="visibility_first"
    )
    assert history.manual_stop is True
    assert history.termination_reason == "manual_stop"
    assert len(history.train_loss) == 1
    assert calibrations == []
    assert last["epoch"] == 1 and last["planned_epochs"] == EPOCHS
    assert batches == 2
    print(f"[RESULT 3] paused at epoch {last['epoch']}, batches={batches}, calibration_calls=0", flush=True)


def test_visibility_stop_mid_training_completes_current_epoch(tmp_path, monkeypatch):
    print(f"\n[SCENARIO 4] visibility AE: STOP during epoch {MID_STOP_EPOCH} of {EPOCHS}", flush=True)
    history, last, calibrations, batches = _ae_stop(
        tmp_path, monkeypatch, stop_after_batch=(MID_STOP_EPOCH - 1) * 2 + 1, name="visibility_mid"
    )
    assert history.manual_stop is True
    assert history.termination_reason == "manual_stop"
    assert len(history.train_loss) == MID_STOP_EPOCH
    assert calibrations == []
    assert last["epoch"] == MID_STOP_EPOCH and last["planned_epochs"] == EPOCHS
    assert batches == MID_STOP_EPOCH * 2
    print(f"[RESULT 4] paused at epoch {last['epoch']}, batches={batches}, calibration_calls=0", flush=True)


# ---------------------------------------------------------------------------
# Joint YOLO (real tiny Ultralytics CPU run)
# ---------------------------------------------------------------------------


def _yolo_stop(tmp_path, monkeypatch, *, stop_epoch: int, name: str):
    from car_smart_assist.perception.detection_yolo import configure_yolo_environment

    monkeypatch.setenv("YOLO_CONFIG_DIR", str(tmp_path / "artifacts/ultralytics_settings"))
    configure_yolo_environment(tmp_path)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    previous_threads = torch.get_num_threads()
    previous_rng = torch.get_rng_state()
    previous_python, previous_numpy = random.getstate(), np.random.get_state()
    torch.set_num_threads(1)
    try:
        import ultralytics
        from ultralytics import YOLO
        from ultralytics.data import utils as data_utils
        from ultralytics.utils import downloads

        monkeypatch.setattr(data_utils, "check_font", lambda *_a, **_k: None)
        monkeypatch.setattr(
            downloads, "safe_download", lambda *_a, **_k: pytest.fail("stop scenario attempted a download")
        )
        assert ultralytics.__version__ == "8.3.162"
        model_yaml = Path(ultralytics.__file__).parent / "cfg/models/11/yolo11.yaml"
        assert model_yaml.is_file()

        from car_smart_assist.perception.yolo_training import (
            attach_training_state_callbacks,
            detection_trainer_for_resume,
            load_training_state,
        )
        import scripts.train_detection as detection_cli

        manifest = {
            "signature": f"stop8-{name}",
            "seed": 41,
            "classes": [{"name": "object", "source_id": 0}],
        }
        train_args = {
            "epochs": EPOCHS,
            "seed": 41,
            "imgsz": 64,
            "batch": 2,
            "nbs": 2,
            "workers": 0,
            "close_mosaic": 0,
            "resume_extra_epochs": None,
            "amp": False,
            "pretrained": False,
            "optimizer": "AdamW",
            "lr0": 0.001,
            "warmup_epochs": 0,
            "patience": 20,
            "deterministic": False,
            "save": True,
            "val": True,
            "plots": False,
            "cache": False,
        }
        cfg = {
            "model": str(model_yaml),
            "checkpoint": str(tmp_path / "not-a-checkpoint.pt"),
            "train": dict(train_args),
            "inference": {"imgsz": 64, "confidence": 0.25, "iou": 0.5, "max_det": 10},
        }
        run = tmp_path / "runs" / name
        plan = detection_cli.resolve_training_plan(tmp_path, cfg, manifest, run)
        run.mkdir(parents=True)
        metadata = {
            "dataset_signature": manifest["signature"],
            "seed": manifest["seed"],
            "classes": manifest["classes"],
            "train_args": plan["train"],
            "training_segment": plan["training_segment"],
            "model_selection_contract": plan["model_selection_contract"],
            "epochs_before_run": 0,
            "segment_resume_checkpoint_sha256": None,
        }
        (run / "run_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

        def write_dataset(dataset_root):
            # Four images with batch=2 -> two batches per epoch, so the first
            # epoch is genuinely incomplete when the first batch requests stop.
            for split in ("train", "val"):
                image_dir = dataset_root / "images" / split
                label_dir = dataset_root / "labels" / split
                image_dir.mkdir(parents=True)
                label_dir.mkdir(parents=True)
                for index in range(4):
                    pixels = np.full((64, 64, 3), 40 + 20 * index, dtype=np.uint8)
                    pixels[20:44, 20:44] = (180, 90, 30)
                    Image.fromarray(pixels).save(image_dir / f"{split}_{index}.jpg")
                    (label_dir / f"{split}_{index}.txt").write_text(
                        "0 0.5 0.5 0.5 0.5\n", encoding="utf-8"
                    )
            path = dataset_root / "dataset.yaml"
            path.write_text(
                f"path: {dataset_root.as_posix()}\n"
                "train: images/train\n"
                "val: images/val\n"
                "names:\n"
                "  0: object\n",
                encoding="utf-8",
            )
            return path

        data_yaml = write_dataset(tmp_path / "dataset" / name)
        controller = StopController(default_directory=run / "weights")
        model = YOLO(str(model_yaml))
        trainer_type = detection_trainer_for_resume(None, plan["model_selection_contract"])
        attach_training_state_callbacks(
            model, manifest["signature"], None,
            selection_contract=plan["model_selection_contract"],
            training_segment=plan["training_segment"],
            stop_controller=controller,
        )
        validated_epochs, saved_epochs, batch_epochs = [], [], []
        requested = {"value": False}

        def request_during_target_epoch(trainer):
            batch_epochs.append(trainer.epoch)
            if trainer.epoch == stop_epoch and not requested["value"]:
                requested["value"] = True
                controller.request()

        model.add_callback("on_train_batch_end", request_during_target_epoch)
        model.add_callback("on_val_end", lambda _validator: validated_epochs.append(model.trainer.epoch))
        model.add_callback("on_model_save", lambda trainer: saved_epochs.append(trainer.epoch))
        model.train(
            data=str(data_yaml),
            project=str(run.parent),
            name=run.name,
            exist_ok=True,
            device="cpu",
            resume=False,
            trainer=trainer_type,
            **plan["train"],
        )
        weights = run / "weights"
        state = load_training_state(
            weights / "last_training_state.pt",
            weights / "last.pt",
            manifest["signature"],
            plan["model_selection_contract"],
        )
        return state, validated_epochs, saved_epochs, batch_epochs
    finally:
        torch.set_num_threads(previous_threads)
        torch.set_rng_state(previous_rng)
        random.setstate(previous_python)
        np.random.set_state(previous_numpy)


def test_yolo_stop_before_first_epoch_completes(tmp_path, monkeypatch):
    print("\n[SCENARIO 5] joint YOLO: STOP during first epoch (before it finishes)", flush=True)
    state, validated, saved, batch_epochs = _yolo_stop(
        tmp_path, monkeypatch, stop_epoch=0, name="yolo_first"
    )
    assert state["epoch"] == 0
    assert state["training_segment"]["target_epochs"] == EPOCHS
    assert state["manual_stop"] is True
    assert state["training_complete"] is False
    assert state["termination_reason"] == "manual_stop"
    assert validated == [0] and saved == [0]
    assert batch_epochs.count(0) == 2 and 1 not in batch_epochs
    print(f"[RESULT 5] paused at epoch {state['epoch']}, validated={validated}, saved={saved}, batch_epochs={batch_epochs}", flush=True)


def test_yolo_stop_mid_training_completes_current_epoch(tmp_path, monkeypatch):
    print(f"\n[SCENARIO 6] joint YOLO: STOP during epoch {MID_STOP_EPOCH} of {EPOCHS}", flush=True)
    state, validated, saved, batch_epochs = _yolo_stop(
        tmp_path, monkeypatch, stop_epoch=MID_STOP_EPOCH - 1, name="yolo_mid"
    )
    assert state["epoch"] == MID_STOP_EPOCH - 1
    assert state["training_segment"]["target_epochs"] == EPOCHS
    assert state["manual_stop"] is True
    assert state["training_complete"] is False
    assert state["termination_reason"] == "manual_stop"
    assert validated == list(range(MID_STOP_EPOCH)) and saved == list(range(MID_STOP_EPOCH))
    assert all(batch_epochs.count(epoch) == 2 for epoch in range(MID_STOP_EPOCH))
    assert MID_STOP_EPOCH not in batch_epochs
    print(f"[RESULT 6] paused at epoch {state['epoch']}, validated={validated}, saved={saved}, batch_epochs={batch_epochs}", flush=True)


# ---------------------------------------------------------------------------
# Pipeline transitions: stop after a model completes, before the next starts
# ---------------------------------------------------------------------------


def _pipeline_transition_stop(tmp_path, monkeypatch, *, stop_after_call: int, name: str):
    helpers = _load("tests/unit/test_training_pipeline_control.py", f"stop8_pipeline_{name}")
    import scripts.train_pipeline as pipeline

    root = helpers._workspace(tmp_path, monkeypatch)
    run_dir = Path("artifacts/training/pipeline")
    work = root / run_dir / "current"

    class TransitionChild:
        def __init__(self, command, code, kwargs):
            self.command = command
            self.code = code
            self.kwargs = kwargs
            self.pid = 4321

        def wait(self):
            script = Path(self.command[3]).name
            preflight = "--preflight" in self.command
            stage = {
                "train_weather_attention.py": "weather-preflight" if preflight else "weather",
                "train_visibility.py": "visibility",
                "train_detection.py": "detection",
            }[script]
            if stage != "weather-preflight" and self.code == 0:
                for path in pipeline.model_checkpoints(self.work, stage).values():
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes((stage + path.name).encode("utf-8"))
            if self.index == stop_after_call:
                Path(self.kwargs["env"]["CAR_SMART_ASSIST_STOP_FILE"]).touch()
            return self.code

    class Spawner:
        def __init__(self):
            self.calls = []

        def __call__(self, command, **kwargs):
            self.calls.append((list(command), kwargs))
            child = TransitionChild(command, 0, kwargs)
            child.work = work
            child.index = len(self.calls) - 1
            return child

    spawner = Spawner()
    monkeypatch.setattr(pipeline.subprocess, "Popen", spawner)
    args = [
        "--fresh", "--run-dir", str(run_dir),
        "--weather-config", "configs/model/weather.yaml",
        "--visibility-config", "configs/model/visibility.yaml",
        "--detection-config", "configs/model/detection.yaml",
    ]
    code = pipeline.main(args)
    state = json.loads((work / "state.json").read_text(encoding="utf-8"))
    return code, state, spawner.calls


def test_pipeline_stop_between_weather_and_visibility(tmp_path, monkeypatch):
    print("\n[SCENARIO 7] pipeline: STOP after weather completes, before visibility starts", flush=True)
    # Calls: 0 preflight, 1 weather, 2 visibility, 3 detection.
    code, state, calls = _pipeline_transition_stop(
        tmp_path, monkeypatch, stop_after_call=1, name="weather_to_visibility"
    )
    assert code == 75
    assert len(calls) == 2
    assert state["status"] == "PAUSED"
    assert state["stages"][0]["status"] == "COMPLETED"  # preflight
    assert state["stages"][1]["status"] == "COMPLETED"  # weather
    assert state["stages"][2]["status"] == "PENDING"    # visibility not started
    assert state["stages"][3]["status"] == "PENDING"    # detection not started
    print(f"[RESULT 7] calls={len(calls)}, weather=COMPLETED, visibility=PENDING, pipeline=PAUSED", flush=True)


def test_pipeline_stop_between_visibility_and_detection(tmp_path, monkeypatch):
    print("\n[SCENARIO 8] pipeline: STOP after visibility completes, before detection starts", flush=True)
    code, state, calls = _pipeline_transition_stop(
        tmp_path, monkeypatch, stop_after_call=2, name="visibility_to_detection"
    )
    assert code == 75
    assert len(calls) == 3
    assert state["status"] == "PAUSED"
    assert state["stages"][0]["status"] == "COMPLETED"
    assert state["stages"][1]["status"] == "COMPLETED"
    assert state["stages"][2]["status"] == "COMPLETED"
    assert state["stages"][3]["status"] == "PENDING"    # detection not started
    print(f"[RESULT 8] calls={len(calls)}, visibility=COMPLETED, detection=PENDING, pipeline=PAUSED", flush=True)
