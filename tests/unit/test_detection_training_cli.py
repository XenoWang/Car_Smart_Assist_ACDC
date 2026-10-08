"""Default resume, explicit fresh, compatibility and history preservation."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path

import pytest
import torch

spec = importlib.util.spec_from_file_location(
    "detection_training_cli", Path(__file__).resolve().parents[2] / "scripts/train_detection.py"
)
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


def setup_run(root, *, epoch=-1, state_version=None):
    run = root / "artifacts/runs/detection/demo"
    weights = run / "weights"
    weights.mkdir(parents=True)
    cfg = {
        "model": "initial.pt",
        "checkpoint": "artifacts/runs/detection/demo/weights/best.pt",
        "train": {"epochs": 2, "seed": 42, "close_mosaic": 5, "resume_extra_epochs": None},
    }
    manifest = {"signature": "dataset", "seed": 42, "classes": [{"name": "car", "source_id": 26}]}
    metadata = {
        "dataset_signature": manifest["signature"],
        "seed": 42,
        "classes": manifest["classes"],
        "train_args": {"epochs": 2, "seed": 42, "close_mosaic": 2},
    }
    (run / "run_metadata.json").write_text(json.dumps(metadata))
    torch.save({"epoch": epoch, "optimizer": {} if epoch >= 0 else None}, weights / "last.pt")
    torch.save({"epoch": -1}, weights / "best.pt")
    if epoch < 0:
        (run / "validation_summary.json").write_text(
            json.dumps(
                {
                    "epochs_completed": 2,
                    "validation_metrics": {"fitness": 0.2},
                }
            )
        )
    if state_version is not None:
        torch.save(
            {
                "format_version": state_version,
                "dataset_signature": manifest["signature"],
                "checkpoint_sha256": hashlib.sha256((weights / "last.pt").read_bytes()).hexdigest(),
                "epoch": epoch if epoch >= 0 else 1,
                "best_fitness": 0.2,
            },
            weights / "last_training_state.pt",
        )
    return cfg, manifest, run


def test_no_checkpoint_and_explicit_resume(tmp_path):
    cfg = {
        "model": "initial.pt",
        "checkpoint": "missing.pt",
        "train": {"epochs": 2, "seed": 42, "close_mosaic": 5},
    }
    manifest = {"seed": 42}
    plan = cli.resolve_training_plan(tmp_path, cfg, manifest, tmp_path / "new", epochs=1)
    assert plan["mode"] == "fresh"
    assert plan["train"]["close_mosaic"] == 1
    with pytest.raises(FileNotFoundError):
        cli.resolve_training_plan(tmp_path, cfg, manifest, tmp_path / "new", require_resume=True)


def test_legacy_completed_model_automatically_reuses_best(tmp_path):
    cfg, manifest, run = setup_run(tmp_path)
    plan = cli.resolve_training_plan(tmp_path, cfg, manifest, run)
    assert plan["mode"] == "finetune"
    assert plan["checkpoint"] == run / "weights/best.pt"
    assert plan["best_fitness"] == 0.2
    assert plan["epochs_before_run"] == 2


def test_project_settings_configured_before_checkpoint_import(tmp_path, monkeypatch):
    cfg, manifest, run = setup_run(tmp_path)
    monkeypatch.delenv("YOLO_CONFIG_DIR", raising=False)
    original_load = torch.load

    def checked_load(*args, **kwargs):
        assert os.environ["YOLO_CONFIG_DIR"] == str(tmp_path / "artifacts/ultralytics_settings")
        return original_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", checked_load)
    assert cli.resolve_training_plan(tmp_path, cfg, manifest, run)["mode"] == "finetune"


def test_interrupted_run_defaults_to_full_resume(tmp_path):
    cfg, manifest, run = setup_run(tmp_path, epoch=0, state_version=1)
    plan = cli.resolve_training_plan(tmp_path, cfg, manifest, run)
    assert plan["mode"] == "resume"
    assert plan["train"]["epochs"] == 2
    assert "resume_extra_epochs" not in plan["train"]


@pytest.mark.parametrize("extra,override,expected", [(None, None, 4), (1, None, 3), (None, 1, 3)])
def test_completed_full_state_adds_epochs(tmp_path, extra, override, expected):
    cfg, manifest, run = setup_run(tmp_path, state_version=2)
    cfg["train"]["resume_extra_epochs"] = extra
    plan = cli.resolve_training_plan(tmp_path, cfg, manifest, run, epochs=override)
    assert plan["mode"] == "resume"
    assert plan["train"]["epochs"] == expected


def test_fresh_ignores_existing_model_and_changed_dataset(tmp_path):
    cfg, manifest, run = setup_run(tmp_path)
    manifest["signature"] = "changed"
    with pytest.raises(ValueError, match="changed"):
        cli.resolve_training_plan(tmp_path, cfg, manifest, run)
    plan = cli.resolve_training_plan(tmp_path, cfg, manifest, run, fresh=True)
    assert plan["mode"] == "fresh"
    assert plan["checkpoint"] == tmp_path / "initial.pt"
    assert plan["epochs_before_run"] == 0


def test_archive_retains_original_files_and_separate_snapshots(tmp_path):
    _, _, run = setup_run(tmp_path, state_version=2)
    original = (run / "weights/best.pt").read_bytes()
    cli.archive_checkpoints(run)
    cli.archive_checkpoints(run)
    assert (run / "weights/best.pt").read_bytes() == original
    for number in (1, 2):
        assert (run / f"history/round_{number:04d}/weights/best.pt").read_bytes() == original
