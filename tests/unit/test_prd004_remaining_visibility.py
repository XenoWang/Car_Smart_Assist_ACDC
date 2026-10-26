from __future__ import annotations

import json
import importlib.util
import logging
from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
from threading import Event, Lock

import numpy as np
import pytest
import torch
from PIL import Image
from filelock import FileLock as RealFileLock

from car_smart_assist.config.visibility import validate_input_size
from car_smart_assist.perception.visibility import dataset as visibility_dataset
from car_smart_assist.perception.visibility.dataset import ensure_cache, list_ref_images
from car_smart_assist.perception.visibility.trainer import VisibilityTrainer, _load_visibility_checkpoint


def _write_ref(path: Path, color: tuple[int, int, int]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color).save(path)
    return path


def test_relocated_nonempty_invalid_manifest_warns_without_guessing_or_dropping_suspect(
    tmp_path, caplog
):
    candidate = _write_ref(
        tmp_path / "new-root/data/raw/acdc/rgb_anon/rain/train_ref/seq/a_rgb_ref_anon.png",
        (10, 20, 30),
    )
    old_invalid = tmp_path / "old-root/data/raw/acdc/rgb_anon/rain/train_ref/seq/old_rgb_ref_anon.png"
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"invalid": [str(old_invalid)]}), encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger="car_smart_assist.perception.visibility.dataset"):
        found = list_ref_images(
            tmp_path / "new-root/data/raw/acdc",
            cleaning_manifest=manifest,
            project_root=tmp_path,
        )

    assert found == [candidate]
    assert "非空 invalid 清单没有匹配" in caplog.text
    assert "重新运行" in caplog.text


def test_matching_invalid_manifest_excludes_only_invalid_and_preserves_suspect(tmp_path):
    invalid = _write_ref(
        tmp_path / "data/raw/acdc/rgb_anon/rain/train_ref/seq/invalid_rgb_ref_anon.png",
        (1, 2, 3),
    )
    suspect = _write_ref(
        tmp_path / "data/raw/acdc/rgb_anon/night/train_ref/seq/suspect_rgb_ref_anon.png",
        (4, 5, 6),
    )
    manifest = tmp_path / "data/processed/manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps({"invalid": [invalid.relative_to(tmp_path).as_posix()]}), encoding="utf-8"
    )

    found = list_ref_images(
        tmp_path / "data/raw/acdc",
        cleaning_manifest="data/processed/manifest.json",
        project_root=tmp_path,
    )

    assert found == [suspect]
    assert invalid.is_file()
    assert suspect.is_file()


def test_shared_cache_build_is_serialized_and_second_caller_reuses_result(tmp_path, monkeypatch):
    source = _write_ref(tmp_path / "source.png", (31, 41, 59))
    cache = tmp_path / "cache/images.npy"
    entered_builder = Event()
    release_builder = Event()
    second_acquire_attempted = Event()
    acquire_count_lock = Lock()
    builds = 0
    lock_acquires = 0
    original_builder = visibility_dataset._build_cache

    class ObservedFileLock:
        def __init__(self, path):
            self._lock = RealFileLock(path)

        def __enter__(self):
            nonlocal lock_acquires
            with acquire_count_lock:
                lock_acquires += 1
                if lock_acquires == 2:
                    second_acquire_attempted.set()
            self._lock.acquire()
            return self

        def __exit__(self, exc_type, exc, traceback):
            self._lock.release()

    monkeypatch.setattr(visibility_dataset, "FileLock", ObservedFileLock)

    def gated_builder(paths, size, target):
        nonlocal builds
        builds += 1
        entered_builder.set()
        assert release_builder.wait(timeout=10)
        original_builder(paths, size, target)

    monkeypatch.setattr(visibility_dataset, "_build_cache", gated_builder)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(ensure_cache, [source], (4, 4), cache)
        assert entered_builder.wait(timeout=10)
        second = pool.submit(ensure_cache, [source], (4, 4), cache)
        observed_contender = second_acquire_attempted.wait(timeout=10)
        release_builder.set()
        assert observed_contender  # FileLock acquisition is an explicit AUDIT-V004-19 contract.
        first_result = first.result(timeout=10)
        second_result = second.result(timeout=10)

    assert first_result == second_result == cache
    assert lock_acquires == 2
    # Behavioral oracle: after serialized publication, only one cache build occurred.
    assert builds == 1
    cached = np.load(cache, allow_pickle=False)
    assert cached.shape == (1, 4, 4, 3)
    assert cached.dtype == np.uint8
    assert tuple(cached[0, 0, 0]) == (31, 41, 59)


def test_cache_publish_permission_error_preserves_previous_cache_and_gives_action(
    tmp_path, monkeypatch
):
    source = _write_ref(tmp_path / "source.png", (12, 34, 56))
    cache = tmp_path / "cache/images.npy"
    ensure_cache([source], (4, 4), cache)
    manifest = cache.with_suffix(cache.suffix + ".json")
    old_cache = cache.read_bytes()
    old_manifest = manifest.read_bytes()
    original_replace = Path.replace

    def deny_cache_replace(path, target):
        if Path(target) == cache:
            raise PermissionError("synthetic Windows mapped-file denial")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", deny_cache_replace)
    with pytest.raises(ValueError, match="结束占用进程.*独立的 data.cache_path"):
        ensure_cache([source], (6, 6), cache)

    assert cache.read_bytes() == old_cache
    assert manifest.read_bytes() == old_manifest


@pytest.mark.parametrize("declared", [[8, 8], (8, 8)])
def test_input_size_contract_accepts_match_and_rejects_conflict(declared):
    validate_input_size({"input_size": declared}, (8, 8))
    with pytest.raises(ValueError, match="与有效模型尺寸"):
        validate_input_size({"input_size": declared}, (16, 16))


def test_trainer_rejects_data_size_that_disagrees_with_model_before_training(tmp_path):
    cfg = {
        "device": "cpu",
        "model": {"input_size": [16, 16], "base_channels": 2, "latent_dim": 4},
        "data": {"input_size": [8, 8]},
        "train": {"checkpoint_dir": str(tmp_path / "checkpoints")},
    }
    with pytest.raises(ValueError, match="与有效模型尺寸"):
        VisibilityTrainer(cfg, project_root=tmp_path, resume="none")


def _write_synthetic_resume_checkpoint(path: Path, model_cfg: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 3,
            "best_model_sha256": "synthetic-independent-checkpoint-identity",
            "model_cfg": model_cfg,
        },
        path,
    )


def test_resume_checkpoint_model_size_is_effective_oracle_for_data_size(tmp_path):
    checkpoint_model_cfg = {"input_size": [16, 16], "base_channels": 2, "latent_dim": 4}
    checkpoint = tmp_path / "checkpoint.pt"
    _write_synthetic_resume_checkpoint(checkpoint, checkpoint_model_cfg)
    matching_cfg = {
        "device": "cpu",
        "model": {"input_size": [8, 8], "base_channels": 2, "latent_dim": 4},
        "data": {"input_size": [16, 16]},
        "train": {"checkpoint_dir": str(tmp_path / "checkpoints")},
    }

    trainer = VisibilityTrainer(matching_cfg, project_root=tmp_path, resume=checkpoint)

    assert trainer.model_cfg["input_size"] == [16, 16]


def test_resume_checkpoint_data_size_conflict_is_rejected_from_checkpoint_metadata(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    _write_synthetic_resume_checkpoint(
        checkpoint,
        {"input_size": [16, 16], "base_channels": 2, "latent_dim": 4},
    )
    cfg = {
        "device": "cpu",
        "model": {"input_size": [8, 8], "base_channels": 2, "latent_dim": 4},
        "data": {"input_size": [8, 8]},
        "train": {"checkpoint_dir": str(tmp_path / "checkpoints")},
    }

    with pytest.raises(ValueError, match="data.input_size=.*有效模型尺寸"):
        VisibilityTrainer(cfg, project_root=tmp_path, resume=checkpoint)


def test_corrupt_checkpoint_is_converted_to_actionable_value_error(tmp_path):
    checkpoint = tmp_path / "broken.pt"
    checkpoint.write_bytes(b"not a torch checkpoint")

    with pytest.raises(ValueError, match="无法读取能见度检查点") as exc_info:
        _load_visibility_checkpoint(checkpoint)

    assert str(checkpoint) in str(exc_info.value)


@pytest.mark.parametrize(
    ("failure_location", "failure"),
    [
        ("fit", FileNotFoundError("synthetic missing cleaning manifest")),
        ("fit", OSError("synthetic empty reference set")),
    ],
    ids=["missing-manifest", "no-reference-images"],
)
def test_train_cli_turns_expected_input_failures_into_exit_two_without_traceback(
    tmp_path, monkeypatch, capsys, failure_location, failure
):
    script = Path(__file__).resolve().parents[2] / "scripts" / "train_visibility.py"
    spec = importlib.util.spec_from_file_location("train_visibility_cli_under_test", script)
    assert spec is not None and spec.loader is not None
    cli = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = cli
    spec.loader.exec_module(cli)
    project = tmp_path / "project"
    (project / "scripts").mkdir(parents=True)
    config_path = project / "configs/model/visibility.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        "visibility:\n  device: cpu\n  model: {input_size: [8, 8], base_channels: 2, latent_dim: 4}\n"
        "  train: {checkpoint_dir: artifacts/checkpoints/visibility}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(cli, "__file__", str(project / "scripts/train_visibility.py"))
    monkeypatch.setattr(
        cli,
        "parse_args",
        lambda: Namespace(
            config="configs/model/visibility.yaml",
            epochs=None,
            batch_size=None,
            device=None,
            log_level="INFO",
            stop_file=None,
            stop=False,
            fresh=False,
            continue_training=False,
            resume="missing.pt",
        ),
    )

    class FailingTrainer:
        resume_path = None

        def __init__(self, *args, **kwargs):
            pass

        def fit(self, **kwargs):
            raise failure

    monkeypatch.setattr(cli, "VisibilityTrainer", FailingTrainer)
    assert cli.main() == 2
    captured = capsys.readouterr()
    assert "Traceback" not in captured.out + captured.err


@pytest.mark.parametrize("checkpoint_exists", [False, True], ids=["missing", "corrupt-pt"])
def test_train_cli_handles_missing_or_corrupt_resume_with_real_trainer(
    tmp_path, monkeypatch, capsys, checkpoint_exists
):
    script = Path(__file__).resolve().parents[2] / "scripts" / "train_visibility.py"
    spec = importlib.util.spec_from_file_location("train_visibility_real_resume_test", script)
    assert spec is not None and spec.loader is not None
    cli = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = cli
    spec.loader.exec_module(cli)
    project = tmp_path / "project"
    (project / "scripts").mkdir(parents=True)
    config_path = project / "configs/model/visibility.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        "visibility:\n  device: cpu\n  model: {input_size: [32, 32], base_channels: 2, latent_dim: 4}\n"
        "  train: {checkpoint_dir: artifacts/checkpoints/visibility}\n",
        encoding="utf-8",
    )
    checkpoint = project / "artifacts/checkpoints/broken.pt"
    checkpoint.parent.mkdir(parents=True)
    corrupt_bytes = b"synthetic truncated torch checkpoint bytes"
    if checkpoint_exists:
        checkpoint.write_bytes(corrupt_bytes)
    monkeypatch.setattr(cli, "__file__", str(project / "scripts/train_visibility.py"))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(script),
            "--config",
            "configs/model/visibility.yaml",
            "--device",
            "cpu",
            "--resume",
            str(checkpoint),
        ],
    )

    assert cli.main() == 2
    captured = capsys.readouterr()
    assert "Traceback" not in captured.out + captured.err
    if checkpoint_exists:
        assert checkpoint.read_bytes() == corrupt_bytes
    else:
        assert not checkpoint.exists()


def test_train_cli_preserves_checkpoint_when_fit_is_interrupted(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[2] / "scripts" / "train_visibility.py"
    spec = importlib.util.spec_from_file_location("train_visibility_interrupt_test", script)
    assert spec is not None and spec.loader is not None
    cli = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = cli
    spec.loader.exec_module(cli)
    project = tmp_path / "project"
    (project / "scripts").mkdir(parents=True)
    config_path = project / "configs/model/visibility.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        "visibility:\n  device: cpu\n  model: {input_size: [8, 8], base_channels: 2, latent_dim: 4}\n"
        "  train: {checkpoint_dir: artifacts/checkpoints/visibility}\n",
        encoding="utf-8",
    )
    checkpoint = project / "artifacts/checkpoints/visibility/last.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"preserved synthetic checkpoint bytes")
    prior_bytes = checkpoint.read_bytes()
    monkeypatch.setattr(cli, "__file__", str(project / "scripts/train_visibility.py"))
    monkeypatch.setattr(
        cli,
        "parse_args",
        lambda: Namespace(
            config="configs/model/visibility.yaml",
            epochs=None,
            batch_size=None,
            device=None,
            log_level="INFO",
            stop_file=None,
            stop=False,
            fresh=False,
            continue_training=False,
            resume="auto",
        ),
    )

    class InterruptedTrainer:
        resume_path = None

        def __init__(self, *args, **kwargs):
            pass

        def fit(self, **kwargs):
            raise KeyboardInterrupt

    monkeypatch.setattr(cli, "VisibilityTrainer", InterruptedTrainer)
    with pytest.raises(KeyboardInterrupt):
        cli.main()

    assert checkpoint.read_bytes() == prior_bytes


def test_evaluate_cli_resolves_explicit_relative_checkpoint_under_project_root(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[2] / "scripts" / "evaluate_visibility.py"
    spec = importlib.util.spec_from_file_location("evaluate_visibility_cli_under_test", script)
    assert spec is not None and spec.loader is not None
    cli = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = cli
    spec.loader.exec_module(cli)
    project = tmp_path / "project"
    (project / "scripts").mkdir(parents=True)
    config_path = project / "configs/model/visibility.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        "visibility:\n  device: cpu\n  model: {input_size: [16, 16]}\n"
        "  data: {input_size: [8, 8]}\n  train: {checkpoint_dir: artifacts/checkpoints/visibility}\n",
        encoding="utf-8",
    )
    checkpoint = project / "artifacts/model.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"synthetic checkpoint placeholder")
    external_cwd = tmp_path / "external-cwd"
    same_name = external_cwd / "artifacts/model.pt"
    same_name.parent.mkdir(parents=True)
    same_name.write_bytes(b"different synthetic CWD file")
    monkeypatch.chdir(external_cwd)
    observed = []

    class SyntheticScorer:
        input_size = (16, 16)

        @classmethod
        def from_checkpoint(cls, path, **kwargs):
            observed.append(Path(path))
            return cls()

    monkeypatch.setattr(cli, "__file__", str(project / "scripts/evaluate_visibility.py"))
    monkeypatch.setattr(cli, "VisibilityScorer", SyntheticScorer)
    monkeypatch.setattr(cli, "resolve_device", lambda _device: "cpu")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(script),
            "--config",
            "configs/model/visibility.yaml",
            "--checkpoint",
            "artifacts/model.pt",
            "--device",
            "cpu",
        ],
    )

    assert cli.main() == 2  # The configured size conflict stops before scoring or data access.
    assert observed == [checkpoint]


def test_inference_pipeline_rejects_data_size_conflict(tmp_path):
    from car_smart_assist.inference.pipeline import InferencePipeline

    with pytest.raises(ValueError, match="与有效模型尺寸"):
        InferencePipeline.from_config(
            {
                "device": "cpu",
                "model": {"input_size": [16, 16]},
                "data": {"input_size": [8, 8]},
                "train": {"checkpoint_dir": "absent"},
            },
            project_root=tmp_path,
            require_visibility=False,
            weather_config={},
        )

