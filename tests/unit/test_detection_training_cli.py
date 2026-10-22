"""默认续跑、显式从头开始、兼容性和历史记录保留。"""

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
        "inference": {"imgsz": 640, "confidence": 0.25, "iou": 0.5, "max_det": 100},
    }
    manifest = {"signature": "dataset", "seed": 42, "classes": [{"name": "car", "source_id": 26}]}
    selection_contract = cli.yolo_model_selection_contract(cfg, manifest)
    metadata = {
        "dataset_signature": manifest["signature"],
        "seed": 42,
        "classes": manifest["classes"],
        "train_args": {"epochs": 2, "seed": 42, "close_mosaic": 2},
        "model_selection_contract": selection_contract,
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
                    "model_selection_contract": selection_contract,
                }
            )
        )
    if state_version is not None:
        torch.save(
            {
                "format_version": state_version,
                "dataset_signature": manifest["signature"],
                "model_selection_contract": selection_contract,
                "checkpoint_sha256": hashlib.sha256((weights / "last.pt").read_bytes()).hexdigest(),
                "epoch": epoch if epoch >= 0 else 1,
                "best_fitness": 0.2,
                "best_checkpoint_sha256": hashlib.sha256((weights / "best.pt").read_bytes()).hexdigest(),
                "training_segment": {
                    "start_epoch": 0, "target_epochs": 2,
                    "resume_extra_epochs": None, "epochs_override": None,
                    "early_stopping_policy": "preserve_history",
                },
                "early_stopper": {
                    "best_fitness": 0.2, "best_epoch": 1, "patience": 3,
                    "possible_stop": False,
                },
            },
            weights / "last_training_state.pt",
        )
    return cfg, manifest, run


def test_no_checkpoint_and_explicit_resume(tmp_path):
    cfg = {
        "model": "initial.pt",
        "checkpoint": "missing.pt",
        "train": {"epochs": 2, "seed": 42, "close_mosaic": 5},
        "inference": {"imgsz": 640, "confidence": 0.25, "iou": 0.5, "max_det": 100},
    }
    manifest = {"seed": 42}
    plan = cli.resolve_training_plan(tmp_path, cfg, manifest, tmp_path / "new", epochs=1)
    assert plan["mode"] == "fresh"
    assert plan["train"]["close_mosaic"] == 1
    with pytest.raises(FileNotFoundError):
        cli.resolve_training_plan(tmp_path, cfg, manifest, tmp_path / "new", require_resume=True)


def test_matching_completed_model_without_full_state_is_rejected(tmp_path):
    cfg, manifest, run = setup_run(tmp_path)
    before = {path: path.read_bytes() for path in (run / "weights").iterdir()}
    with pytest.raises(ValueError, match="resume state is missing"):
        cli.resolve_training_plan(tmp_path, cfg, manifest, run)
    assert {path: path.read_bytes() for path in before} == before


def test_new_directory_finetunes_old_best_without_old_state_or_fitness(tmp_path, monkeypatch):
    cfg, manifest, source_run = setup_run(tmp_path, state_version=3)
    metadata_path = source_run / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata.pop("model_selection_contract")
    metadata_path.write_text(json.dumps(metadata))
    summary_path = source_run / "validation_summary.json"
    summary = json.loads(summary_path.read_text())
    summary.pop("model_selection_contract")
    summary_path.write_text(json.dumps(summary))
    state_path = source_run / "weights/last_training_state.pt"
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    state.pop("model_selection_contract")
    torch.save(state, state_path)

    def forbidden_load(*args, **kwargs):
        pytest.fail("new-directory fine-tuning must not load the old resume state")

    monkeypatch.setattr(torch, "load", forbidden_load)
    new_run = tmp_path / "artifacts/runs/detection/new-name"
    plan = cli.resolve_training_plan(tmp_path, cfg, manifest, new_run)
    assert plan["mode"] == "finetune"
    assert plan["checkpoint"] == source_run / "weights/best.pt"
    assert plan["resume_state"] is None
    assert plan["best_fitness"] is None
    assert plan["epochs_before_run"] == 2


def test_project_settings_configured_before_checkpoint_import(tmp_path, monkeypatch):
    cfg, manifest, run = setup_run(tmp_path, state_version=4)
    monkeypatch.delenv("YOLO_CONFIG_DIR", raising=False)
    original_load = torch.load

    def checked_load(*args, **kwargs):
        # The tensor-only FP32 state cannot import Ultralytics; the model payload can.
        if kwargs.get("weights_only") is False:
            assert os.environ["YOLO_CONFIG_DIR"] == str(tmp_path / "artifacts/ultralytics_settings")
        return original_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", checked_load)
    assert cli.resolve_training_plan(tmp_path, cfg, manifest, run)["mode"] == "resume"


def test_interrupted_run_defaults_to_full_resume(tmp_path):
    cfg, manifest, run = setup_run(tmp_path, epoch=0, state_version=4)
    plan = cli.resolve_training_plan(tmp_path, cfg, manifest, run)
    assert plan["mode"] == "resume"
    assert plan["train"]["epochs"] == 2
    assert "resume_extra_epochs" not in plan["train"]


@pytest.mark.parametrize("extra,override,expected", [(None, None, 4), (1, None, 3), (None, 1, 3)])
def test_completed_full_state_adds_epochs(tmp_path, extra, override, expected):
    cfg, manifest, run = setup_run(tmp_path, state_version=4)
    cfg["train"]["resume_extra_epochs"] = extra
    plan = cli.resolve_training_plan(tmp_path, cfg, manifest, run, epochs=override)
    assert plan["mode"] == "resume"
    assert plan["train"]["epochs"] == expected


def test_fresh_ignores_existing_model_and_changed_dataset(tmp_path):
    cfg, manifest, run = setup_run(tmp_path)
    metadata_path = run / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata.pop("model_selection_contract")
    metadata_path.write_text(json.dumps(metadata))
    manifest["signature"] = "changed"
    with pytest.raises(ValueError, match="changed"):
        cli.resolve_training_plan(tmp_path, cfg, manifest, run)
    plan = cli.resolve_training_plan(tmp_path, cfg, manifest, run, fresh=True)
    assert plan["mode"] == "fresh"
    assert plan["checkpoint"] == tmp_path / "initial.pt"
    assert plan["epochs_before_run"] == 0


@pytest.mark.parametrize("record_kind", ["metadata", "summary", "state"])
def test_old_or_changed_selection_contract_is_rejected_before_checkpoint_load(
    tmp_path, monkeypatch, record_kind
):
    cfg, manifest, run = setup_run(
        tmp_path,
        epoch=-1 if record_kind == "summary" else (0 if record_kind == "state" else -1),
        state_version=3 if record_kind == "state" else None,
    )
    metadata_path = run / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    if record_kind == "metadata":
        metadata.pop("model_selection_contract")
        metadata_path.write_text(json.dumps(metadata))
    elif record_kind == "summary":
        summary_path = run / "validation_summary.json"
        summary = json.loads(summary_path.read_text())
        summary["model_selection_contract"] = {"imgsz": 641}
        summary_path.write_text(json.dumps(summary))
    else:
        state_path = run / "weights/last_training_state.pt"
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        state["model_selection_contract"] = {"imgsz": 641}
        torch.save(state, state_path)

    if record_kind != "state":
        def forbidden_load(*args, **kwargs):
            pytest.fail("checkpoint deserialization must not occur before contract rejection")

        monkeypatch.setattr(torch, "load", forbidden_load)
    observed_files = (
        run / "run_metadata.json",
        run / "validation_summary.json",
        run / "weights/last.pt",
        run / "weights/best.pt",
        run / "weights/last_training_state.pt",
    )
    before = {path: path.read_bytes() for path in observed_files if path.is_file()}
    with pytest.raises(ValueError, match="contract|missing or changed"):
        cli.resolve_training_plan(tmp_path, cfg, manifest, run)
    assert {path: path.read_bytes() for path in before} == before
    assert not (run / "history").exists()


def test_archive_retains_original_files_and_separate_snapshots(tmp_path):
    _, _, run = setup_run(tmp_path, state_version=3)
    original = (run / "weights/best.pt").read_bytes()
    cli.archive_checkpoints(run)
    cli.archive_checkpoints(run)
    assert (run / "weights/best.pt").read_bytes() == original
    for number in (1, 2):
        assert (run / f"history/round_{number:04d}/weights/best.pt").read_bytes() == original
def test_interrupted_extra_segment_keeps_original_target_after_second_resume(tmp_path, monkeypatch):
    cfg, manifest, run = setup_run(tmp_path, epoch=11, state_version=3)
    weights = run / "weights"
    best, last = weights / "best.pt", weights / "last.pt"
    segment = {
        "start_epoch": 10, "target_epochs": 15, "resume_extra_epochs": 5,
        "epochs_override": None, "early_stopping_policy": "preserve_history",
    }
    state_path = weights / "last_training_state.pt"
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    state.update(
        format_version=4,
        checkpoint_sha256=hashlib.sha256(last.read_bytes()).hexdigest(),
        best_checkpoint_sha256=hashlib.sha256(best.read_bytes()).hexdigest(),
        training_segment=segment,
        early_stopper={"best_fitness": 0.2, "best_epoch": 8, "patience": 4,
                       "possible_stop": False},
    )
    torch.save(state, state_path)
    metadata_path = run / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["train_args"] = {**metadata["train_args"], "epochs": 15}
    metadata["training_segment"] = segment
    metadata["segment_resume_checkpoint_sha256"] = hashlib.sha256(last.read_bytes()).hexdigest()
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    cfg["train"]["resume_extra_epochs"] = 5
    plan = cli.resolve_training_plan(tmp_path, cfg, manifest, run)
    assert plan["mode"] == "resume"
    assert plan["training_segment"]["target_epochs"] == 15
    assert plan["train"]["epochs"] == 15


def test_old_format3_complete_resume_is_explicitly_rejected(tmp_path):
    cfg, manifest, run = setup_run(tmp_path, epoch=0, state_version=3)
    with pytest.raises(ValueError, match="history is missing or incompatible"):
        cli.resolve_training_plan(tmp_path, cfg, manifest, run)
