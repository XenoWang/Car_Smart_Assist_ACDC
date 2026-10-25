from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from car_smart_assist.data.detection import _materialize_image, validate_detection_export
from car_smart_assist.perception.visibility.trainer import VisibilityTrainer

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _isolated_cpu_state(monkeypatch):
    threads = torch.get_num_threads()
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state().clone()
    torch.set_num_threads(2)
    try:
        yield
    finally:
        # Undo monkeypatches first so RNG restores call the real APIs.
        monkeypatch.undo()
        torch.set_num_threads(threads)
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)


def _helper(path, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _RecordingGradScaler:
    """Delegates to real torch.amp.GradScaler while recording actual state loads."""
    def __init__(self, delegate, loads):
        self.delegate = delegate
        self.loads = loads

    def __getattr__(self, name):
        return getattr(self.delegate, name)

    def load_state_dict(self, state):
        self.loads.append(copy.deepcopy(state))
        return self.delegate.load_state_dict(state)


@pytest.mark.parametrize("initial_scaler_enabled", [False, True], ids=["disabled-empty", "enabled-state"])
def test_attention_resume_restores_only_nonempty_real_scaler_state(
    tmp_path, monkeypatch, initial_scaler_enabled
):
    training, cfg, datasets, split, _ = _helper(
        "tests/unit/test_training_stop_models.py", "p2_attention_fixture"
    )._attention_setup(tmp_path, monkeypatch)
    real_scaler = torch.amp.GradScaler
    scaler_loads = []
    constructions = 0

    def cpu_scaler_factory(_device, *, enabled):
        nonlocal constructions
        constructions += 1
        use_enabled = bool(enabled) or (constructions == 1 and initial_scaler_enabled) or constructions > 1
        return _RecordingGradScaler(real_scaler("cpu", enabled=use_enabled), scaler_loads)

    monkeypatch.setattr(training.torch.amp, "GradScaler", cpu_scaler_factory)
    writer = training.write_json

    def interrupt_after_checkpoint(path, value):
        if path.name == "progress.json" and value.get("epoch") == 2:
            raise RuntimeError("synthetic interruption after checkpoint")
        return writer(path, value)

    monkeypatch.setattr(training, "write_json", interrupt_after_checkpoint)
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        training.train_attention(cfg, tmp_path, datasets, split, fresh=True)
    monkeypatch.setattr(training, "write_json", writer)

    checkpoint = tmp_path / "checkpoints/last.pt"
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    scaler_state = saved["scaler_state"]
    if initial_scaler_enabled:
        assert set(scaler_state) == {
            "scale", "growth_factor", "backoff_factor", "growth_interval", "_growth_tracker"
        }
    else:
        assert scaler_state == {}

    training.train_attention(cfg, tmp_path, datasets, split, fresh=False)
    if scaler_state:
        assert scaler_loads == [scaler_state]
    else:
        assert scaler_loads == []


def _visibility_checkpoint(tmp_path, monkeypatch, *, epochs=1):
    helper = _helper("tests/unit/test_visibility_retrain_training.py", "p2_visibility_fixture")
    trainer = helper._trainer_with_controlled_validation(
        tmp_path, monkeypatch, [1.0, 0.8][:epochs], epochs=epochs
    )
    trainer.fit()
    return helper, tmp_path / "checkpoints/last.pt"


def _set_cuda_layout(monkeypatch, current_uuids, captured=None):
    import torch.cuda
    monkeypatch.setattr(torch.cuda, "device_count", lambda: len(current_uuids))
    monkeypatch.setattr(
        torch.cuda, "get_device_properties",
        lambda index: SimpleNamespace(uuid=current_uuids[index]),
    )
    monkeypatch.setattr(
        torch.cuda, "set_rng_state_all",
        lambda states: captured.extend(item.clone() for item in states) if captured is not None
        else pytest.fail("rejected CUDA layout attempted RNG restore"),
    )


@pytest.mark.parametrize(
    "case,saved_ids,saved_count,current_ids",
    [
        ("missing UUID", None, 1, ["gpu-a"]),
        ("fewer visible devices", ["gpu-a", "gpu-b"], 2, ["gpu-a"]),
        ("more visible devices", ["gpu-a"], 1, ["gpu-a", "gpu-b"]),
        ("changed device order", ["gpu-a", "gpu-b"], 2, ["gpu-b", "gpu-a"]),
        ("changed device UUID", ["gpu-a", "gpu-b"], 2, ["gpu-a", "gpu-c"]),
    ],
)
def test_visibility_cuda_layout_rejection_precedes_all_state_mutation(
    tmp_path, monkeypatch, case, saved_ids, saved_count, current_ids
):
    helper, checkpoint = _visibility_checkpoint(tmp_path, monkeypatch)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["rng_state"]["cuda"] = [
        torch.tensor([index + 1, index + 2], dtype=torch.uint8) for index in range(saved_count)
    ]
    if saved_ids is None:
        payload["rng_state"].pop("cuda_device_ids", None)
    else:
        payload["rng_state"]["cuda_device_ids"] = saved_ids
    torch.save(payload, checkpoint)

    oracle = torch.load(checkpoint, map_location="cpu", weights_only=False)
    resumed = helper.VisibilityTrainer(
        helper._config(tmp_path, epochs=1), project_root=tmp_path, resume=checkpoint
    )
    resumed.device = torch.device("cuda:0")
    optimizer = torch.optim.AdamW(resumed.model.parameters())
    before_model = {key: value.clone() for key, value in resumed.model.state_dict().items()}
    before_python = random.getstate()
    before_numpy = np.random.get_state()
    before_torch = torch.get_rng_state().clone()
    _set_cuda_layout(monkeypatch, current_ids)

    def unexpected(*_args, **_kwargs):
        pytest.fail(f"{case}: state mutation occurred before CUDA layout rejection")

    monkeypatch.setattr(resumed, "_load_verified_best", unexpected)
    monkeypatch.setattr(resumed.model, "load_state_dict", unexpected)
    monkeypatch.setattr(optimizer, "load_state_dict", unexpected)
    import car_smart_assist.perception.visibility.trainer as training
    monkeypatch.setattr(training.random, "setstate", unexpected)
    monkeypatch.setattr(training.np.random, "set_state", unexpected)
    monkeypatch.setattr(torch, "set_rng_state", unexpected)

    match = "缺少CUDA设备身份" if saved_ids is None else "CUDA设备数量"
    with pytest.raises(ValueError, match=match):
        resumed._restore(optimizer, None)
    assert oracle["rng_state"]["cuda"]
    assert all(torch.equal(resumed.model.state_dict()[key], value)
               for key, value in before_model.items())
    assert not optimizer.state
    assert random.getstate() == before_python
    after_numpy = np.random.get_state()
    assert before_numpy[0] == after_numpy[0] and np.array_equal(before_numpy[1], after_numpy[1])
    assert before_numpy[2:] == after_numpy[2:]
    assert torch.equal(torch.get_rng_state(), before_torch)


def _assert_nested_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        assert torch.equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key, value in expected.items():
            _assert_nested_equal(actual[key], value)
    elif isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            _assert_nested_equal(left, right)
    else:
        assert actual == expected


def test_visibility_matching_cuda_layout_restores_full_state_with_injected_identity(
    tmp_path, monkeypatch
):
    helper, checkpoint = _visibility_checkpoint(tmp_path, monkeypatch)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    ids = ["mock-gpu-a", "mock-gpu-b"]
    cuda_states = [torch.tensor([3, 4], dtype=torch.uint8), torch.tensor([5, 6], dtype=torch.uint8)]
    payload["rng_state"]["cuda"] = cuda_states
    payload["rng_state"]["cuda_device_ids"] = ids
    torch.save(payload, checkpoint)
    oracle = torch.load(checkpoint, map_location="cpu", weights_only=False)

    resumed = helper.VisibilityTrainer(
        helper._config(tmp_path, epochs=1), project_root=tmp_path, resume=checkpoint
    )
    resumed.device = torch.device("cuda:0")
    optimizer = torch.optim.AdamW(resumed.model.parameters())
    restored_cuda = []
    _set_cuda_layout(monkeypatch, ids, restored_cuda)
    resumed._restore(optimizer, None)

    for key, expected in oracle["model_state"].items():
        assert torch.equal(resumed.model.state_dict()[key], expected)
    _assert_nested_equal(optimizer.state_dict()["state"], oracle["optimizer_state"]["state"])
    _assert_nested_equal(random.getstate(), oracle["rng_state"]["python"])
    actual_numpy = np.random.get_state()
    expected_numpy = oracle["rng_state"]["numpy"]
    assert actual_numpy[0] == expected_numpy[0]
    assert np.array_equal(actual_numpy[1], expected_numpy[1])
    assert actual_numpy[2:] == expected_numpy[2:]
    assert torch.equal(torch.get_rng_state(), oracle["rng_state"]["torch"])
    assert torch.equal(resumed.loader_generator.get_state(), oracle["rng_state"]["loader"])
    _assert_nested_equal(restored_cuda, cuda_states)


def _tiny_visibility_trainer(tmp_path, monkeypatch, *, epochs=1):
    helper = _helper("tests/unit/test_visibility_retrain_training.py", "p2_visibility_fit_fixture")
    trainer = helper._trainer_with_controlled_validation(
        tmp_path, monkeypatch, [0.5] * epochs, epochs=epochs
    )
    return trainer


@pytest.mark.parametrize("use_cuda", [False, True], ids=["cpu", "injected-cuda"])
def test_visibility_payload_records_only_training_device_cuda_identity(
    tmp_path, monkeypatch, use_cuda
):
    trainer = _tiny_visibility_trainer(tmp_path, monkeypatch)
    expected_ids = ["mock-gpu-a", "mock-gpu-b"]
    expected_states = [torch.tensor([1, 2], dtype=torch.uint8),
                       torch.tensor([3, 4], dtype=torch.uint8)]
    if use_cuda:
        trainer.device = torch.device("cuda:0")
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
        monkeypatch.setattr(torch.cuda, "get_device_properties",
                            lambda index: SimpleNamespace(uuid=expected_ids[index]))
        monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: expected_states)
    else:
        def unexpected(*_args, **_kwargs):
            pytest.fail("CPU payload queried CUDA state or device identity")

        for name in ("device_count", "get_device_properties", "get_rng_state_all"):
            monkeypatch.setattr(torch.cuda, name, unexpected)
    rng = trainer._make_payload(0, None, None, None)["rng_state"]
    assert rng["cuda_device_ids"] == (expected_ids if use_cuda else [])
    _assert_nested_equal(rng["cuda"], expected_states if use_cuda else [])


@pytest.mark.gpu
def test_visibility_fp16_overflow_skips_optimizer_step_and_reduces_scale(tmp_path, monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("requires a real CUDA device for FP16 GradScaler fit")
    trainer = _tiny_visibility_trainer(tmp_path, monkeypatch, epochs=1)
    trainer.device = torch.device("cuda:0")
    trainer.model.to(trainer.device)
    parameter = next(item for item in trainer.model.parameters() if item.requires_grad)
    before = {name: value.detach().clone() for name, value in trainer.model.named_parameters()}
    hook_calls = []

    def overflow_gradient(gradient):
        hook_calls.append(True)
        return torch.full_like(gradient, float("inf"))

    parameter.register_hook(overflow_gradient)
    trainer.cfg["train"]["amp"] = True
    trainer.cfg["train"]["dtype"] = "float16"
    trainer.fit()
    assert hook_calls
    assert trainer.history.train_loss and np.isfinite(trainer.history.train_loss[0])
    assert trainer.scaler.is_enabled()
    assert trainer.scaler.get_scale() < 65536
    parameters_after = dict(trainer.model.named_parameters())
    assert all(torch.equal(parameters_after[name], value) for name, value in before.items())
    last = torch.load(tmp_path / "checkpoints/last.pt", map_location="cpu", weights_only=False)
    assert last["optimizer_state"]["state"] == {}


def test_visibility_cpu_non_amp_nonfinite_gradient_still_raises(tmp_path, monkeypatch):
    trainer = _tiny_visibility_trainer(tmp_path, monkeypatch, epochs=1)
    parameter = next(item for item in trainer.model.parameters() if item.requires_grad)
    parameter.register_hook(lambda gradient: torch.full_like(gradient, float("inf")))
    with pytest.raises(RuntimeError, match="non-finite"):
        trainer.fit()
    assert not (tmp_path / "checkpoints/last.pt").exists()




@pytest.mark.parametrize("mode", ["hardlink", "copy"])
def test_materialize_rejects_changed_source_before_new_destination(tmp_path, mode):
    source, destination = tmp_path / "source.png", tmp_path / "out/frame.png"
    source.write_bytes(b"changed after digest")
    expected = hashlib.sha256(b"approved bytes").hexdigest()
    with pytest.raises(ValueError, match="Source image content changed"):
        _materialize_image(source, destination, mode, expected)
    assert not destination.exists()


def test_materialize_rejects_modified_existing_hardlink_samefile(tmp_path):
    source, destination = tmp_path / "source.png", tmp_path / "out/frame.png"
    source.write_bytes(b"approved bytes")
    destination.parent.mkdir(parents=True)
    destination.hardlink_to(source)
    expected = hashlib.sha256(source.read_bytes()).hexdigest()
    source.write_bytes(b"changed through shared inode")
    assert source.samefile(destination)
    with pytest.raises(ValueError, match="Existing exported image differs"):
        _materialize_image(source, destination, "hardlink", expected)


@pytest.mark.parametrize("mode", ["hardlink", "copy"])
def test_materialize_accepts_verified_content_and_is_idempotent(tmp_path, mode):
    source, destination = tmp_path / "source.png", tmp_path / "out/frame.png"
    source.write_bytes(b"verified image bytes")
    expected = hashlib.sha256(source.read_bytes()).hexdigest()
    assert _materialize_image(source, destination, mode, expected) in {"hardlink", "copy"}
    assert _materialize_image(source, destination, mode, expected) == "existing"
    assert destination.read_bytes() == source.read_bytes()


def _write_base_export(root, *, corrupt=False):
    base = root / "data/processed/base"
    image = base / "images/train/frame.png"
    label = base / "labels/train/frame.txt"
    source = root / "data/raw/acdc/frame.png"
    source.parent.mkdir(parents=True, exist_ok=True)
    image.parent.mkdir(parents=True)
    label.parent.mkdir(parents=True)
    Image.new("RGB", (4, 4), (10, 20, 30)).save(source)
    image_bytes = source.read_bytes()
    image.write_bytes(b"changed base image" if corrupt else image_bytes)
    label.write_text("", encoding="utf-8")
    (base / "train.txt").write_text("./images/train/frame.png\n", encoding="utf-8")
    digest = hashlib.sha256(image_bytes).hexdigest()
    manifest = {
        "seed": 7,
        "signature": "synthetic-base-signature",
        "classes": [{"name": "car"}],
        "images": [{
            "split": "train", "group": "base-group", "sha256": digest,
            "image": "images/train/frame.png", "label": "labels/train/frame.txt",
            "source_relative": "frame.png", "source": str(source), "boxes": [],
        }],
    }
    (base / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return base


def _joint_config():
    return {
        "group_strategy": "location",
        "source_weights": {"acdc": 0.8, "lost_and_found": 0.2},
        "classification_loss_weights": {"acdc": 1.0, "lost_and_found": 1.0},
        "acdc_manifest": "data/processed/base/manifest.json",
        "seed": 7,
        "output_root": "data/processed/joint",
        "report_root": "artifacts/reports/joint",
        "lost_and_found_root": "data/raw/lost_and_found",
        "cleaning_manifest": "data/cleaning.json",
        "development_ratios": {"train": 0.7, "val": 0.15, "calibration": 0.15},
        "integrity_workers": 1,
        "image_mode": "copy",
        "additional_class": {"name": "obstacle"},
        "obstacle_train_id": 2,
        "valid_roi_ids": [1, 2],
    }


def _write_laf_fixture(root):
    dataset = root / "data/raw/lost_and_found"
    for index in range(3):
        location = f"location-{index}"
        relative = Path("train") / location
        stem = f"frame-{index}"
        image = dataset / "leftImg8bit" / relative / f"{stem}_leftImg8bit.png"
        mask = dataset / "gtCoarse" / relative / f"{stem}_gtCoarse_labelTrainIds.png"
        instances = mask.with_name(f"{stem}_gtCoarse_instanceIds.png")
        for path in (image, mask, instances):
            path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (4, 4), (20 + index, 30, 40)).save(image)
        Image.fromarray(np.ones((4, 4), dtype=np.uint8)).save(mask)
        Image.fromarray(np.zeros((4, 4), dtype=np.uint16)).save(instances)
    (root / "data").mkdir(exist_ok=True)
    (root / "data/cleaning.json").write_text(json.dumps({"invalid": []}), encoding="utf-8")


def test_joint_prepare_rejects_changed_base_before_publishing_output(tmp_path):
    from car_smart_assist.data.detection_joint import prepare_joint_detection

    _write_base_export(tmp_path, corrupt=True)
    cfg = _joint_config()
    with pytest.raises(ValueError, match="Exported image content changed"):
        prepare_joint_detection(tmp_path, cfg)
    assert not (tmp_path / cfg["output_root"]).exists()


def test_joint_prepare_hash_checks_base_and_final_and_is_idempotent(tmp_path, monkeypatch):
    import car_smart_assist.data.detection_joint as joint

    _write_base_export(tmp_path)
    _write_laf_fixture(tmp_path)
    calls = []
    original_validate = joint.validate_detection_export

    def record_hash_validation(output, verify_hashes=False):
        calls.append(verify_hashes)
        return original_validate(output, verify_hashes=verify_hashes)

    monkeypatch.setattr(joint, "validate_detection_export", record_hash_validation)
    cfg = _joint_config()
    first = joint.prepare_joint_detection(tmp_path, cfg)
    second = joint.prepare_joint_detection(tmp_path, cfg)
    assert calls == [True, True, True, True]
    assert first["signature"] == second["signature"]
    output = (tmp_path / cfg["output_root"]).resolve()
    exported = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    image = output / exported["images"][0]["image"]
    image.write_bytes(b"altered after joint export")
    with pytest.raises(ValueError, match="Exported image content changed"):
        original_validate(output, verify_hashes=True)


@pytest.mark.parametrize("request_kind", ["file", "signal"])
def test_continue_stop_during_identity_check_is_preserved(tmp_path, monkeypatch, request_kind):
    helpers = _helper("tests/unit/test_training_pipeline_control.py", "p2_pipeline_control_helpers")
    import scripts.train_pipeline as pipeline

    root = helpers._workspace(tmp_path, monkeypatch)
    run_dir = Path("artifacts/training/pipeline")
    work = root / run_dir / "current"
    spawner = helpers._Spawner(work, [75])
    monkeypatch.setattr(pipeline.subprocess, "Popen", spawner)
    handlers = {}
    monkeypatch.setattr(pipeline.signal, "signal", lambda key, handler: handlers.__setitem__(key, handler))
    args = ["--fresh", "--run-dir", str(run_dir), "--weather-config", "configs/model/weather.yaml",
            "--visibility-config", "configs/model/visibility.yaml",
            "--detection-config", "configs/model/detection.yaml"]
    assert pipeline.main(args) == pipeline.STOP_EXIT_CODE
    calls = len(spawner.calls)
    stop = root / run_dir / "STOP"
    verify = pipeline.verify_identity

    def request_during_verify(state):
        verify(state)
        if request_kind == "file":
            stop.touch()
        else:
            handlers[pipeline.signal.SIGINT](pipeline.signal.SIGINT, None)

    monkeypatch.setattr(pipeline, "verify_identity", request_during_verify)
    assert pipeline.main(["--continue", *args[1:]]) == pipeline.STOP_EXIT_CODE
    assert stop.is_file() and len(spawner.calls) == calls


@pytest.mark.parametrize("request_kind", ["file", "signal"])
def test_fresh_stop_during_archive_survives_and_prevents_child_start(
    tmp_path, monkeypatch, request_kind
):
    helpers = _helper("tests/unit/test_training_pipeline_control.py", "p2_pipeline_archive_helpers")
    import scripts.train_pipeline as pipeline

    root = helpers._workspace(tmp_path, monkeypatch)
    run_dir = Path("artifacts/training/pipeline")
    work = root / run_dir / "current"
    spawner = helpers._Spawner(work, [0, 0, 0, 0])
    monkeypatch.setattr(pipeline.subprocess, "Popen", spawner)
    handlers = {}
    monkeypatch.setattr(pipeline.signal, "signal", lambda key, handler: handlers.__setitem__(key, handler))
    args = ["--fresh", "--run-dir", str(run_dir), "--weather-config", "configs/model/weather.yaml",
            "--visibility-config", "configs/model/visibility.yaml",
            "--detection-config", "configs/model/detection.yaml"]
    assert pipeline.main(args) == 0
    calls = len(spawner.calls)
    stop = root / run_dir / "STOP"
    real_replace = pipeline.os.replace

    def archive_then_stop(source, destination):
        real_replace(source, destination)
        if Path(source) == work:
            if request_kind == "file":
                stop.touch()
            else:
                handlers[pipeline.signal.SIGINT](pipeline.signal.SIGINT, None)

    monkeypatch.setattr(pipeline.os, "replace", archive_then_stop)
    with pytest.raises(SystemExit) as stopped:
        pipeline.main(args)
    assert stopped.value.code == pipeline.STOP_EXIT_CODE
    assert stop.is_file()
    assert len(list((root / run_dir / "history").glob("*/state.json"))) == 1
    assert len(spawner.calls) == calls
