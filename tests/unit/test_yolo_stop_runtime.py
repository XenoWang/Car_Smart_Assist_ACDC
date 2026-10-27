"""Real, tiny CPU Ultralytics stop/resume ordering and paired-state check."""

import copy
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
import torch


def _assert_tree_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        assert torch.equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_tree_equal(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            _assert_tree_equal(left, right)
    else:
        assert actual == expected


def _write_tiny_detection_dataset(root):
    for split in ("train", "val"):
        image_dir = root / "images" / split
        label_dir = root / "labels" / split
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        for index in range(2):
            pixels = np.full((64, 64, 3), 40 + 50 * index, dtype=np.uint8)
            pixels[20:44, 20:44] = (180, 90, 30)
            Image.fromarray(pixels).save(image_dir / f"{split}_{index}.jpg")
            (label_dir / f"{split}_{index}.txt").write_text(
                "0 0.5 0.5 0.5 0.5\n", encoding="utf-8"
            )
    data = root / "dataset.yaml"
    data.write_text(
        f"path: {root.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: object\n",
        encoding="utf-8",
    )
    return data


def test_real_ultralytics_stop_saves_epoch_then_continue_restores_and_uses_saved_target(
    tmp_path, monkeypatch
):
    vendor_environment = ("CUDA_VISIBLE_DEVICES", "CUBLAS_WORKSPACE_CONFIG", "PYTHONHASHSEED")
    previous_environment = {key: os.environ.get(key) for key in vendor_environment}
    # Configure project-local settings before importing Ultralytics.
    from car_smart_assist.perception.detection_yolo import configure_yolo_environment

    monkeypatch.setenv("YOLO_CONFIG_DIR", str(tmp_path / "artifacts/ultralytics_settings"))
    configure_yolo_environment(tmp_path)
    # Entry snapshots are the independent state oracle; cleanup that restores only
    # thread count would leave Python/NumPy/Torch RNG or deterministic flags changed.
    # Preserve live CUDA RNG before the availability stub and any manual_seed path.
    previous_cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    previous_threads = torch.get_num_threads()
    previous_rng = torch.get_rng_state()
    previous_python_rng, previous_numpy_rng = random.getstate(), np.random.get_state()
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    previous_cudnn_benchmark = torch.backends.cudnn.benchmark
    previous_cudnn_deterministic = torch.backends.cudnn.deterministic
    torch.set_num_threads(1)
    try:
        import ultralytics
        from ultralytics import YOLO
        from ultralytics.data import utils as data_utils
        from ultralytics.utils import downloads
        from ultralytics.engine import trainer as trainer_module, validator as validator_module
        from ultralytics.utils.torch_utils import select_device

        def select_cpu_without_hiding_cuda(device, *args, **kwargs):
            # The vendor string CPU selector can cache zero visible GPUs in a
            # fresh pytest process. Its resolved-device path preserves visibility
            # while retaining the real CPU trainer, validator and model updates.
            if str(device) == "cpu":
                device = torch.device("cpu")
            return select_device(device, *args, **kwargs)

        monkeypatch.setattr(trainer_module, "select_device", select_cpu_without_hiding_cuda)
        monkeypatch.setattr(validator_module, "select_device", select_cpu_without_hiding_cuda)

        # Plotting is disabled; font acquisition is unrelated to the training contract.
        monkeypatch.setattr(data_utils, "check_font", lambda *_a, **_k: None)
        monkeypatch.setattr(downloads, "safe_download", lambda *_a, **_k: pytest.fail("Runtime fixture attempted a download"))
        assert ultralytics.__version__ == "8.3.162"
        model_yaml = Path(ultralytics.__file__).parent / "cfg/models/11/yolo11.yaml"
        assert model_yaml.is_file(), "The installed local YOLO11 architecture YAML is required"

        from car_smart_assist.perception.yolo_training import (
            attach_training_state_callbacks,
            detection_trainer_for_resume,
            load_training_state,
        )
        from car_smart_assist.utils.stop import StopController
        import scripts.train_detection as detection_cli

        manifest = {
            "signature": "tiny-local-ultralytics-stop-fixture-v1",
            "seed": 41,
            "classes": [{"name": "object", "source_id": 0}],
        }
        train_args = {
            "epochs": 2,
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
        run = tmp_path / "runs" / "tiny_stop"
        initial_plan = detection_cli.resolve_training_plan(tmp_path, cfg, manifest, run)
        assert initial_plan["mode"] == "fresh"
        run.mkdir(parents=True)
        metadata = {
            "dataset_signature": manifest["signature"],
            "seed": manifest["seed"],
            "classes": manifest["classes"],
            "train_args": initial_plan["train"],
            "training_segment": initial_plan["training_segment"],
            "model_selection_contract": initial_plan["model_selection_contract"],
            "epochs_before_run": 0,
            "segment_resume_checkpoint_sha256": None,
        }
        (run / "run_metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        data_yaml = _write_tiny_detection_dataset(tmp_path / "dataset")

        controller = StopController(default_directory=run / "weights")
        model = YOLO(str(model_yaml))
        trainer_type = detection_trainer_for_resume(
            None, initial_plan["model_selection_contract"]
        )
        attach_training_state_callbacks(
            model,
            manifest["signature"],
            None,
            selection_contract=initial_plan["model_selection_contract"],
            training_segment=initial_plan["training_segment"],
            stop_controller=controller,
        )
        validated_epochs = []
        saved_epochs = []

        def request_after_first_training_batch(trainer):
            if not controller.requested:
                controller.request()

        model.add_callback("on_train_batch_end", request_after_first_training_batch)
        model.add_callback("on_val_end", lambda _validator: validated_epochs.append(model.trainer.epoch))
        model.add_callback("on_model_save", lambda trainer: saved_epochs.append(trainer.epoch))
        first_results = model.train(
            data=str(data_yaml),
            project=str(run.parent),
            name=run.name,
            exist_ok=True,
            device="cpu",
            resume=False,
            trainer=trainer_type,
            **initial_plan["train"],
        )

        assert os.environ.get("CUDA_VISIBLE_DEVICES") == previous_environment["CUDA_VISIBLE_DEVICES"]

        weights = run / "weights"
        last = weights / "last.pt"
        best = weights / "best.pt"
        state_path = weights / "last_training_state.pt"
        stopped_state = load_training_state(
            state_path, last, manifest["signature"], initial_plan["model_selection_contract"]
        )
        assert validated_epochs == [0]
        assert saved_epochs == [0]
        assert first_results.results_dict
        assert stopped_state["epoch"] == 0
        assert stopped_state["manual_stop"] is True
        assert stopped_state["training_complete"] is False
        assert stopped_state["termination_reason"] == "manual_stop"
        assert stopped_state["training_segment"]["target_epochs"] == 2
        assert stopped_state["checkpoint_sha256"] == hashlib.sha256(last.read_bytes()).hexdigest()
        assert stopped_state["best_checkpoint_sha256"] == hashlib.sha256(best.read_bytes()).hexdigest()

        continuation = detection_cli.resolve_training_plan(
            tmp_path, cfg, manifest, run, continue_only=True
        )
        assert continuation["mode"] == "resume"
        assert continuation["no_op"] is False
        assert continuation["resume_state"]["epoch"] == 0
        assert continuation["training_segment"]["target_epochs"] == 2
        assert continuation["train"]["epochs"] == 2

        # Keep the independent disk-loaded oracle: CPU Optimizer.load_state_dict may
        # share state tensors with the supplied mapping and update that mapping later.
        _assert_tree_equal(continuation["resume_state"], stopped_state)
        resumed_controller = StopController(default_directory=weights)
        resumed_controller.resume()
        assert not resumed_controller.requested
        resumed_model = YOLO(str(last))
        resumed_trainer_type = detection_trainer_for_resume(
            continuation["resume_state"], continuation["model_selection_contract"]
        )
        attach_training_state_callbacks(
            resumed_model,
            manifest["signature"],
            continuation["resume_state"],
            selection_contract=continuation["model_selection_contract"],
            training_segment=continuation["training_segment"],
            stop_controller=resumed_controller,
        )
        resumed_epochs = []
        restored = {}

        def capture_restored_state(trainer):
            restored["model"] = {
                key: value.detach().cpu().clone()
                for key, value in trainer.model.state_dict().items()
            }
            restored["optimizer"] = copy.deepcopy(trainer.optimizer.state_dict())
            restored["scheduler"] = copy.deepcopy(trainer.scheduler.state_dict())
            restored["scaler"] = copy.deepcopy(trainer.scaler.state_dict())
            restored["ema"] = {
                key: value.detach().cpu().clone()
                for key, value in trainer.ema.ema.state_dict().items()
            }
            restored["ema_updates"] = trainer.ema.updates
            restored["torch_rng"] = torch.get_rng_state().clone()

        resumed_model.add_callback(
            "on_train_epoch_start", lambda trainer: resumed_epochs.append(trainer.epoch)
        )
        resumed_model.add_callback("on_train_start", capture_restored_state)
        resumed_results = resumed_model.train(
            data=str(data_yaml),
            project=str(run.parent),
            name=run.name,
            exist_ok=True,
            device="cpu",
            resume=True,
            trainer=resumed_trainer_type,
            **continuation["train"],
        )

        assert os.environ.get("CUDA_VISIBLE_DEVICES") == previous_environment["CUDA_VISIBLE_DEVICES"]
        assert resumed_epochs == [1]
        assert resumed_results.results_dict
        _assert_tree_equal(restored["model"], stopped_state["model"])
        _assert_tree_equal(restored["optimizer"], stopped_state["optimizer"])
        _assert_tree_equal(restored["scheduler"], stopped_state["scheduler"])
        _assert_tree_equal(restored["scaler"], stopped_state["scaler"])
        _assert_tree_equal(restored["ema"], stopped_state["ema"])
        assert restored["ema_updates"] == stopped_state["ema_updates"]
        assert torch.equal(restored["torch_rng"], stopped_state["torch_rng"])

        completed_state = load_training_state(
            state_path, last, manifest["signature"], continuation["model_selection_contract"]
        )
        assert completed_state["epoch"] == 1
        assert completed_state["manual_stop"] is False
        assert completed_state["training_complete"] is True
        assert completed_state["termination_reason"] == "budget_completed"
        assert completed_state["training_segment"]["target_epochs"] == 2
        assert completed_state["checkpoint_sha256"] == hashlib.sha256(last.read_bytes()).hexdigest()
        assert completed_state["best_checkpoint_sha256"] == hashlib.sha256(best.read_bytes()).hexdigest()
    finally:
        try:
            random.setstate(previous_python_rng)
            np.random.set_state(previous_numpy_rng)
            torch.set_rng_state(previous_rng)
            if previous_cuda_rng is not None:
                torch.cuda.set_rng_state_all(previous_cuda_rng)
            torch.use_deterministic_algorithms(previous_deterministic, warn_only=previous_warn_only)
            torch.backends.cudnn.benchmark = previous_cudnn_benchmark
            torch.backends.cudnn.deterministic = previous_cudnn_deterministic
            torch.set_num_threads(previous_threads)
        finally:
            for key, value in previous_environment.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
    assert random.getstate() == previous_python_rng
    restored_numpy_rng = np.random.get_state()
    assert restored_numpy_rng[0] == previous_numpy_rng[0]
    np.testing.assert_array_equal(restored_numpy_rng[1], previous_numpy_rng[1])
    assert restored_numpy_rng[2:] == previous_numpy_rng[2:]
    assert torch.equal(torch.get_rng_state(), previous_rng)
    assert torch.get_num_threads() == previous_threads
    assert torch.are_deterministic_algorithms_enabled() is previous_deterministic
    assert torch.is_deterministic_algorithms_warn_only_enabled() is previous_warn_only
    assert torch.backends.cudnn.benchmark is previous_cudnn_benchmark
    assert torch.backends.cudnn.deterministic is previous_cudnn_deterministic
    assert {key: os.environ.get(key) for key in vendor_environment} == previous_environment
    if previous_cuda_rng is not None:
        restored_cuda_rng = torch.cuda.get_rng_state_all()
        assert len(restored_cuda_rng) == len(previous_cuda_rng)
        assert all(
            torch.equal(actual, expected)
            for actual, expected in zip(restored_cuda_rng, previous_cuda_rng, strict=True)
        )
        print(f"CUDA_RNG_RESTORED_DEVICES={len(restored_cuda_rng)}")
