from __future__ import annotations

import random
import signal

import numpy as np
import pytest
import torch

from car_smart_assist.utils.stop import STOP_EXIT_CODE, StopController


@pytest.fixture(autouse=True)
def _isolated_cpu_random_state():
    threads = torch.get_num_threads()
    state = torch.get_rng_state()
    python_state, numpy_state = random.getstate(), np.random.get_state()
    torch.set_num_threads(2)
    try:
        yield
    finally:
        torch.set_num_threads(threads)
        torch.set_rng_state(state)
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def test_stop_file_and_nested_marker_latch_until_explicit_resume(tmp_path):
    shared = tmp_path / "pipeline" / "STOP"
    child = tmp_path / "model" / "STOP"
    controller = StopController(shared, default_directory=tmp_path / "pipeline")
    controller.watch_directory(child.parent)

    child.parent.mkdir(parents=True)
    child.touch()
    assert controller.requested
    child.unlink()
    assert controller.requested  # Once observed, a marker remains latched.

    controller.resume()
    assert not controller.requested
    assert not shared.exists()
    assert not child.exists()


def test_pipeline_child_continue_cannot_delete_a_new_shared_stop(tmp_path, monkeypatch):
    shared = tmp_path / "pipeline" / "STOP"
    monkeypatch.setenv("CAR_SMART_ASSIST_STOP_FILE", str(shared))
    monkeypatch.setenv("CAR_SMART_ASSIST_PIPELINE_CHILD", "1")
    controller = StopController(shared, default_directory=tmp_path / "pipeline")
    controller.watch_directory(tmp_path / "model")
    shared.parent.mkdir(parents=True)
    shared.touch()  # Simulates a parent request arriving during child startup.

    controller.resume()

    assert shared.is_file()
    assert controller.requested
    with pytest.raises(SystemExit) as stopped:
        controller.require_running()
    assert stopped.value.code == STOP_EXIT_CODE


def test_nested_signal_handler_restoration(tmp_path, monkeypatch):
    calls = []
    handlers = {getattr(signal, name): object() for name in ("SIGINT", "SIGTERM", "SIGBREAK")
                if hasattr(signal, name)}
    monkeypatch.setattr(signal, "signal", lambda signum, handler: calls.append((signum, handler)) or handlers.get(signum))
    controller = StopController(tmp_path / "STOP", default_directory=tmp_path)

    with controller:
        installed_count = len(calls)
        with controller:
            assert len(calls) == installed_count
            controller._signal(None, None)
            assert controller.requested
        assert len(calls) == installed_count

    restored = calls[installed_count:]
    assert {signum for signum, _ in restored} == set(handlers)
    assert all(handler is handlers[signum] for signum, handler in restored)


def _assert_nested_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        assert torch.equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_nested_equal(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            _assert_nested_equal(left, right)
    elif hasattr(expected, "shape") and hasattr(expected, "dtype"):
        assert np.array_equal(actual, expected)
    else:
        assert actual == expected


def test_fresh_ae_archives_and_removes_old_active_checkpoint_pair(tmp_path):
    from car_smart_assist.perception.visibility.trainer import archive_visibility_checkpoints

    directory = tmp_path / "checkpoints"
    directory.mkdir()
    old = {name: (name + " old bytes").encode() for name in ("last.pt", "best.pt")}
    for name, data in old.items():
        (directory / name).write_bytes(data)

    archive_visibility_checkpoints(directory)

    archived = list((directory / "history").glob("*/last.pt"))
    assert len(archived) == 1
    folder = archived[0].parent
    assert {name: (folder / name).read_bytes() for name in old} == old
    assert not (directory / "last.pt").exists()
    assert not (directory / "best.pt").exists()


def test_stop_cli_writes_marker_without_starting_training_or_archiving(tmp_path, monkeypatch):
    import scripts.train_pipeline as pipeline

    run_dir = tmp_path / "artifacts/training/run"
    current = run_dir / "current"
    current.mkdir(parents=True)
    state = current / "state.json"
    state.write_text("sentinel", encoding="utf-8")
    monkeypatch.setattr(pipeline, "ROOT", tmp_path)
    monkeypatch.setattr(pipeline.subprocess, "Popen", lambda *a, **k: pytest.fail("stop launched a child"))

    assert pipeline.main(["--stop", "--run-dir", "artifacts/training/run"]) == 0
    assert (run_dir / "STOP").is_file()
    assert state.read_text(encoding="utf-8") == "sentinel"
    assert not (run_dir / "history").exists()


def test_ae_stop_finishes_epoch_skips_calibration_and_continue_matches_full_trajectory(
    tmp_path, monkeypatch
):
    from car_smart_assist.perception.visibility.scorer import CalibrationStats
    from car_smart_assist.perception.visibility.trainer import VisibilityTrainer

    def config(directory):
        return {
            "device": "cpu",
            "model": {"input_size": (16, 16), "base_channels": 2, "latent_dim": 4},
            "data": {"num_workers": 0},
            "train": {
                "epochs": 3, "batch_size": 2, "lr": 1e-3, "weight_decay": 0.0,
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

    def make_trainer(directory):
        trainer = VisibilityTrainer(config(directory), project_root=tmp_path, resume="none")
        dataset = Rows()
        monkeypatch.setattr(trainer, "_build_datasets", lambda: (dataset, dataset, dataset, []))
        return trainer

    # The uninterrupted reference sees validation losses 1.0, 1.1, 1.2.
    random.seed(23); np.random.seed(23); torch.manual_seed(23)
    reference = make_trainer(tmp_path / "reference")
    reference_values = iter([1.0, 1.1, 1.2])
    monkeypatch.setattr(reference, "_eval_loss", lambda _loader: next(reference_values))
    monkeypatch.setattr(reference, "_calibrate", lambda _loader: CalibrationStats(0.01, 0.002, 0.02, 0.03, 4))
    reference.fit()
    expected = torch.load(tmp_path / "reference/last.pt", map_location="cpu", weights_only=False)

    random.seed(23); np.random.seed(23); torch.manual_seed(23)
    paused = make_trainer(tmp_path / "resumed")
    controller = StopController(tmp_path / "resumed/STOP", default_directory=tmp_path / "resumed")
    values = iter([1.0, 1.1, 1.2])
    calibrations = []
    monkeypatch.setattr(paused, "_eval_loss", lambda _loader: next(values))
    monkeypatch.setattr(paused, "_calibrate", lambda _loader: calibrations.append(True))
    training_batches = []

    def request_during_first_batch(module, _inputs, _output):
        if module.training:
            training_batches.append(True)
            if len(training_batches) == 1:
                controller.request()

    hook = paused.model.register_forward_hook(request_during_first_batch)
    history = paused.fit(stop_controller=controller)
    hook.remove()
    assert len(training_batches) == 2  # Both batches finish despite the first-batch request.
    assert history.manual_stop and history.termination_reason == "manual_stop"
    assert len(history.train_loss) == 1  # Request is honored only after full epoch save.
    assert calibrations == []
    first_last = torch.load(tmp_path / "resumed/last.pt", map_location="cpu", weights_only=False)
    first_best = torch.load(tmp_path / "resumed/best.pt", map_location="cpu", weights_only=False)
    assert first_last["epoch"] == 1 and first_last["planned_epochs"] == 3

    continued = VisibilityTrainer(config(tmp_path / "resumed"), project_root=tmp_path,
                                  resume=tmp_path / "resumed/last.pt")
    dataset = Rows()
    monkeypatch.setattr(continued, "_build_datasets", lambda: (dataset, dataset, dataset, []))
    rest = iter([1.1, 1.2])
    monkeypatch.setattr(continued, "_eval_loss", lambda _loader: next(rest))
    monkeypatch.setattr(continued, "_calibrate", lambda _loader: CalibrationStats(0.01, 0.002, 0.02, 0.03, 4))
    completed = continued.fit(continue_only=True)
    actual = torch.load(tmp_path / "resumed/last.pt", map_location="cpu", weights_only=False)

    assert completed.termination_reason == "budget_completed"
    assert actual["epoch"] == 3 and actual["planned_epochs"] == 3
    # Calibration may add metadata; nonimprovement must preserve the winning model and epoch.
    final_best = torch.load(tmp_path / "resumed/best.pt", map_location="cpu", weights_only=False)
    _assert_nested_equal(final_best["model_state"], first_best["model_state"])
    assert final_best["epoch"] == first_best["epoch"] == 1
    assert final_best["best_model_sha256"] == first_best["best_model_sha256"]
    for key in ("model_state", "optimizer_state", "scheduler_state", "rng_state"):
        _assert_nested_equal(actual[key], expected[key])


def test_legacy_cnn_continue_refuses_new_format_last_without_torch_rng(tmp_path):
    from dataclasses import replace

    from PIL import Image

    from car_smart_assist.config.weather import load_weather_config
    from car_smart_assist.perception.weather_training import train_weather

    original = load_weather_config()
    cfg = replace(
        original,
        image_size=(32, 48), channels=(4, 8, 8), dropout=0, device="cpu",
        train={**original.train, "acdc_root": "acdc", "cache_dir": "cache", "epochs": 1,
               "batch_size": 2, "num_workers": 0, "patience": 3,
               "use_pixel_accurate": False},
        checkpoint="checkpoints/weather_attributes/best.pt",
    )
    for attr_index, condition in enumerate(cfg.attributes):
        for split, count in (("train", 2), ("val", 1)):
            sequence = f"{condition}_{split}_sequence"
            directory = tmp_path / "acdc/rgb_anon" / condition / split / sequence
            directory.mkdir(parents=True)
            for index in range(count):
                image = np.full((40, 60, 3), 20 + attr_index * 50 + index, dtype=np.uint8)
                Image.fromarray(image).save(directory / f"{sequence}_frame_{index:06d}_rgb_anon.png")

    train_weather(cfg, tmp_path)
    last = tmp_path / "checkpoints/weather_attributes/last.pt"
    payload = torch.load(last, map_location="cpu", weights_only=False)
    assert payload["format_version"] == 2 and "planned_epochs" in payload
    payload.pop("torch_rng")
    torch.save(payload, last)
    before = last.read_bytes()

    with pytest.raises(ValueError, match="RNG|随机状态"):
        train_weather(cfg, tmp_path, continue_only=True)
    assert last.read_bytes() == before


def _attention_compare_inputs(tmp_path):
    from types import SimpleNamespace

    cfg = SimpleNamespace(
        device="cpu", image_size=(8, 8), attributes=("fog",), checkpoint_root="checkpoints",
        train={"epochs": 1}, data={"cache_dir": "cache", "seed": 1},
        evaluation={"attention_references": {}, "asl_config": "asl.yaml", "asl_checkpoint": "asl.pt",
                    "old_config": "old.yaml", "old_checkpoint": "old.pt", "batch_size": 1},
    )
    datasets = {"test": SimpleNamespace(samples=[]), "calibration": SimpleNamespace(samples=[])}
    from dataclasses import dataclass

    @dataclass
    class OldConfig:
        device: str
        image_size: tuple
        attributes: tuple
        decision_thresholds: dict

    cfg_old = OldConfig(device="cpu", image_size=(8, 8), attributes=("fog",), decision_thresholds={"fog": 0.5})
    cfg_asl = SimpleNamespace(image_size=(8, 8))
    split = {"signature": "split", "manifest": "manifest", "splits": {}}
    trained = {"selected_kind": "attention_asl", "candidates": {}, "thresholds": {"fog": 0.5},
               "baseline_hashes": {}, "best_epoch": 1}
    return cfg, datasets, cfg_old, cfg_asl, split, trained


def test_attention_asl_test_stop_prevents_following_calibration(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import car_smart_assist.perception.weather_attention_training as training

    cfg, datasets, _old_cfg, cfg_asl, split, trained = _attention_compare_inputs(tmp_path)
    controller = StopController(tmp_path / "STOP", default_directory=tmp_path)
    calls = []
    monkeypatch.setattr(training, "select_device", lambda _device: torch.device("cpu"))
    monkeypatch.setattr(training, "load_weather_pyramid_config", lambda _path: cfg_asl)
    monkeypatch.setattr(training.WeatherPyramidPredictor, "from_checkpoint",
                        lambda *_a, **_k: SimpleNamespace(model=object(), thresholds={"fog": 0.5}))

    def probabilities(_model, dataset, *_args):
        calls.append(dataset)
        if len(calls) == 2:
            controller.request()  # Request arrives during the pyramid ASL test pass.
        return np.zeros((1, 1), dtype=np.float32)

    monkeypatch.setattr(training, "attention_probabilities", probabilities)
    monkeypatch.setattr(training, "balanced_thresholds", lambda *_a: pytest.fail("calibration ran after stop"))
    with pytest.raises(SystemExit) as stopped:
        training.compare_attention(cfg, tmp_path, datasets, split, object(), trained,
                                   stop_controller=controller)
    assert stopped.value.code == STOP_EXIT_CODE
    assert len(calls) == 2


def test_attention_old_model_test_stop_prevents_its_calibration(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import car_smart_assist.perception.weather_attention_training as training

    cfg, datasets, cfg_old, cfg_asl, split, trained = _attention_compare_inputs(tmp_path)
    controller = StopController(tmp_path / "STOP", default_directory=tmp_path)
    calls = {"calibration": 0, "old": 0}
    monkeypatch.setattr(training, "select_device", lambda _device: torch.device("cpu"))
    monkeypatch.setattr(training, "load_weather_pyramid_config", lambda _path: cfg_asl)
    monkeypatch.setattr(training.WeatherPyramidPredictor, "from_checkpoint",
                        lambda *_a, **_k: SimpleNamespace(model=object(), thresholds={"fog": 0.5}))
    monkeypatch.setattr(training, "load_weather_config", lambda _path: cfg_old)
    monkeypatch.setattr(training.WeatherPredictor, "from_checkpoint",
                        lambda *_a, **_k: SimpleNamespace(thresholds={"fog": 0.5}))
    monkeypatch.setattr(training, "WeatherDataset",
                        lambda samples, *_a: SimpleNamespace(samples=samples, is_test=samples is datasets["test"].samples))
    monkeypatch.setattr(training, "attention_probabilities", lambda *_a: np.zeros((1, 1), dtype=np.float32))

    def thresholds(dataset, *_args):
        if dataset is datasets["calibration"]:
            calls["calibration"] += 1
        return {"fog": 0.5}, {}

    def old_probabilities(_model, dataset, _batch_size):
        calls["old"] += 1
        if calls["old"] == 1:
            controller.request()  # Request arrives during the old model's test pass.
        return np.zeros((1, 1), dtype=np.float32)

    monkeypatch.setattr(training, "balanced_thresholds", thresholds)
    monkeypatch.setattr(training, "_old_probabilities", old_probabilities)
    monkeypatch.setattr(training, "detailed_metrics", lambda *_a: {})
    with pytest.raises(SystemExit) as stopped:
        training.compare_attention(cfg, tmp_path, datasets, split, object(), trained,
                                   stop_controller=controller)
    assert stopped.value.code == STOP_EXIT_CODE
    assert calls["calibration"] == 1  # ASL calibration only; old-model calibration never starts.
    assert calls["old"] == 1



@pytest.mark.parametrize("clear", [False, True])
def test_yolo_fresh_archive_clears_stale_pair_but_normal_archive_retains_it(tmp_path, clear):
    from scripts.train_detection import archive_checkpoints

    run = tmp_path / "run"
    paths = ("weights/last.pt", "weights/best.pt", "weights/last_training_state.pt",
             "run_metadata.json", "validation_summary.json", "args.yaml", "results.csv")
    for name in paths:
        path = run / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(("previous run " + name).encode())
    originals = {name: (run / name).read_bytes() for name in paths}
    archive_checkpoints(run, clear=clear)
    for name, expected in originals.items():
        assert (run / "history/round_0001" / name).read_bytes() == expected
        assert (run / name).exists() is (not clear)
        if not clear:
            assert (run / name).read_bytes() == expected


@pytest.mark.parametrize(
    "module_name,config_name,key,guard_name",
    [("train_visibility", "visibility.yaml", "visibility", "VisibilityTrainer"),
     ("train_weather", "weather_classifier.yaml", "weather_classifier", "train_weather"),
     ("train_weather_attention", "weather_attention.yaml", "weather_attention", "prepare_attention_datasets"),
     ("train_detection", "yolo_detection.yaml", "yolo_detection", "validate_detection_export")],
)
def test_individual_stop_action_never_starts_model_or_reads_dataset(
    tmp_path, monkeypatch, module_name, config_name, key, guard_name
):
    import importlib
    import sys
    from pathlib import Path
    import yaml

    module = importlib.import_module("scripts." + module_name)
    root = Path(__file__).resolve().parents[2]
    raw = yaml.safe_load((root / "configs/model" / config_name).read_text(encoding="utf-8"))
    output = tmp_path / "project/output"
    config = raw[key]
    if key == "visibility":
        config["train"]["checkpoint_dir"] = str(output)
    elif key == "weather_classifier":
        config["checkpoint"] = str(output / "best.pt")
    elif key == "weather_attention":
        config["checkpoint_root"] = str(output)
    else:
        config["project"] = str(output)
        config["name"] = "model"
        output = output / "model/weights"
    cfg = tmp_path / "recipe.yaml"
    cfg.write_text(yaml.safe_dump(raw), encoding="utf-8")
    monkeypatch.setattr(module, "__file__", str(tmp_path / "project/scripts" / (module_name + ".py")))
    monkeypatch.setattr(sys, "argv", [module_name, "--config", str(cfg), "--stop"])
    monkeypatch.setattr(module, guard_name, lambda *_a, **_k: pytest.fail("stop performed training/data work"))
    assert module.main() == 0
    assert (output / "STOP").is_file()
    assert not (output / "history").exists()
