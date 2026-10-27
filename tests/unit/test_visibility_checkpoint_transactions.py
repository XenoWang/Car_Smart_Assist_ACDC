"""Synthetic crash-recovery coverage for visibility checkpoint save transactions."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[2]


def _load_helper(relative: str, name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _publish_interrupted_pair(directory: Path, monkeypatch, fail_name: str, updates=None):
    import car_smart_assist.perception.visibility.trainer as training

    directory.mkdir(parents=True, exist_ok=True)
    original_replace = training.os.replace
    transaction = directory / ".visibility_checkpoint_transaction"

    def fail_during_publication(source, destination):
        source, destination = Path(source), Path(destination)
        if (
            source.parent == transaction
            and source.name == fail_name
            and destination.parent == directory
        ):
            raise OSError(f"synthetic interruption publishing {fail_name}")
        return original_replace(source, destination)

    monkeypatch.setattr(training.os, "replace", fail_during_publication)
    with pytest.raises(OSError, match=f"synthetic interruption publishing {fail_name}"):
        training._publish_checkpoint_transaction(
            directory,
            updates or {
                "best.pt": {"payload": "new-best"},
                "last.pt": {"payload": "new-last"},
            },
        )
    monkeypatch.setattr(training.os, "replace", original_replace)
    return transaction


def _write_old_pair(directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    originals = {"best.pt": b"old-best-bytes", "last.pt": b"old-last-bytes"}
    for name, content in originals.items():
        (directory / name).write_bytes(content)
    return originals


def test_best_published_last_missing_recovers_existing_pair(tmp_path, monkeypatch):
    import torch

    from car_smart_assist.perception.visibility.trainer import (
        _publish_checkpoint_transaction,
        recover_visibility_checkpoint_transaction,
    )

    directory = tmp_path / "checkpoints"
    originals = _write_old_pair(directory)
    _publish_interrupted_pair(directory, monkeypatch, "last.pt")
    assert torch.load(directory / "best.pt", map_location="cpu", weights_only=False) == {
        "payload": "new-best"
    }
    assert (directory / "last.pt").read_bytes() == originals["last.pt"]

    assert recover_visibility_checkpoint_transaction(directory)
    assert torch.load(directory / "best.pt", map_location="cpu", weights_only=False) == {
        "payload": "new-best"
    }
    assert torch.load(directory / "last.pt", map_location="cpu", weights_only=False) == {
        "payload": "new-last"
    }
    assert not (directory / ".visibility_checkpoint_transaction").exists()
    # A subsequent single-file publication still preserves the best identity pair.
    _publish_checkpoint_transaction(directory, {"last.pt": {"payload": "last-only"}})
    assert torch.load(directory / "best.pt", map_location="cpu", weights_only=False) == {
        "payload": "new-best"
    }
    assert torch.load(directory / "last.pt", map_location="cpu", weights_only=False) == {
        "payload": "last-only"
    }


def test_first_save_with_no_old_last_recovers_after_best_publish(tmp_path, monkeypatch):
    import torch

    from car_smart_assist.perception.visibility.trainer import recover_visibility_checkpoint_transaction

    directory = tmp_path / "checkpoints"
    directory.mkdir()
    _publish_interrupted_pair(directory, monkeypatch, "last.pt")
    assert (directory / "best.pt").is_file()
    assert not (directory / "last.pt").exists()
    assert recover_visibility_checkpoint_transaction(directory)
    assert torch.load(directory / "best.pt", map_location="cpu", weights_only=False) == {
        "payload": "new-best"
    }
    assert torch.load(directory / "last.pt", map_location="cpu", weights_only=False) == {
        "payload": "new-last"
    }


@pytest.mark.parametrize("tamper", ["candidate", "backup", "journal", "formal"])
def test_transaction_tampering_is_rejected_without_overwriting_formal_pair(
    tmp_path, monkeypatch, tamper
):
    import car_smart_assist.perception.visibility.trainer as training

    directory = tmp_path / "checkpoints"
    originals = _write_old_pair(directory)
    transaction = _publish_interrupted_pair(directory, monkeypatch, "best.pt")
    if tamper == "candidate":
        target = transaction / "best.pt"
    elif tamper == "backup":
        target = transaction / "best.pt.before"
    elif tamper == "journal":
        target = transaction / "journal.json"
    else:
        target = directory / "best.pt"
    target.write_bytes(target.read_bytes() + b"tampered")
    formal_before_recovery = {
        name: (directory / name).read_bytes() for name in ("best.pt", "last.pt")
    }

    with pytest.raises(ValueError, match="事务|检查点|摘要|journal|ownership"):
        training.recover_visibility_checkpoint_transaction(directory)
    assert {
        name: (directory / name).read_bytes() for name in ("best.pt", "last.pt")
    } == formal_before_recovery
    if tamper != "formal":
        assert formal_before_recovery == originals
    assert transaction.exists()


def test_fresh_archives_exact_original_pair_and_pending_transaction_bytes(tmp_path, monkeypatch):
    import car_smart_assist.perception.visibility.trainer as training

    directory = tmp_path / "checkpoints"
    originals = _write_old_pair(directory)
    transaction = _publish_interrupted_pair(directory, monkeypatch, "best.pt")
    transaction_originals = {
        path.name: path.read_bytes() for path in transaction.iterdir() if path.is_file()
    }
    training.archive_visibility_checkpoints(directory)
    archives = list((directory / "history").glob("*"))
    assert len(archives) == 1
    archived = archives[0]
    assert {name: (archived / name).read_bytes() for name in originals} == originals
    archived_transaction = archived / ".visibility_checkpoint_transaction"
    assert {
        path.name: path.read_bytes() for path in archived_transaction.iterdir()
    } == transaction_originals
    assert not (directory / "best.pt").exists()
    assert not (directory / "last.pt").exists()
    assert not transaction.exists()


def test_recovery_removes_only_a_confirmed_empty_fixed_transaction_directory(tmp_path):
    from car_smart_assist.perception.visibility.trainer import recover_visibility_checkpoint_transaction

    directory = tmp_path / "checkpoints"
    transaction = directory / ".visibility_checkpoint_transaction"
    transaction.mkdir(parents=True)
    assert recover_visibility_checkpoint_transaction(directory) is False
    assert not transaction.exists()


def test_fresh_archive_removes_only_an_empty_fixed_transaction_directory(tmp_path):
    from car_smart_assist.perception.visibility.trainer import archive_visibility_checkpoints

    directory = tmp_path / "checkpoints"
    transaction = directory / ".visibility_checkpoint_transaction"
    transaction.mkdir(parents=True)
    archive_visibility_checkpoints(directory)
    assert not transaction.exists()
    assert not (directory / "history").exists()


def test_recovery_finishes_cleanup_when_all_after_and_owner_was_removed(tmp_path, monkeypatch):
    import json

    import car_smart_assist.perception.visibility.trainer as training

    directory = tmp_path / "checkpoints"
    original_state = training._visibility_transaction_state
    interrupted = False

    def remove_owner_after_pair_publish(path):
        nonlocal interrupted
        result = original_state(path)
        _, transaction, files, owner = result
        if not interrupted and owner is not None and "journal.json" in files:
            journal = json.loads(files["journal.json"].read_text(encoding="utf-8"))
            current = {
                name: training._checkpoint_file_sha256(directory / name)
                for name in ("best.pt", "last.pt")
            }
            if current == journal["after_sha256"]:
                files["owner.json"].unlink()
                interrupted = True
                raise OSError("synthetic interruption after owner cleanup")
        return result

    monkeypatch.setattr(training, "_visibility_transaction_state", remove_owner_after_pair_publish)
    with pytest.raises(OSError, match="after owner cleanup"):
        training._publish_checkpoint_transaction(
            directory,
            {"best.pt": {"payload": "published-best"}, "last.pt": {"payload": "published-last"}},
        )
    assert interrupted
    transaction = directory / ".visibility_checkpoint_transaction"
    assert (transaction / "journal.json").is_file()
    assert not (transaction / "owner.json").exists()
    monkeypatch.setattr(training, "_visibility_transaction_state", original_state)

    assert training.recover_visibility_checkpoint_transaction(directory)
    assert not transaction.exists()


def test_direct_trainer_auto_recovers_pending_pair_before_continue(tmp_path, monkeypatch):
    import copy
    import torch

    import car_smart_assist.perception.visibility.trainer as training

    helper = _load_helper(
        "tests/unit/test_visibility_retrain_training.py", "visibility_auto_resume_fixture"
    )
    first = helper._trainer_with_controlled_validation(tmp_path, monkeypatch, [1.0], epochs=1)
    first.fit()
    directory = tmp_path / "checkpoints"
    payload = torch.load(directory / "last.pt", map_location="cpu", weights_only=False)
    changed = copy.deepcopy(payload)
    changed["termination_reason"] = "synthetic_pending_publication"
    _publish_interrupted_pair(
        directory,
        monkeypatch,
        "last.pt",
        updates={"best.pt": changed, "last.pt": changed},
    )

    resumed = training.VisibilityTrainer(
        helper._config(tmp_path, epochs=1), project_root=tmp_path, resume="auto"
    )
    monkeypatch.setattr(resumed, "_fit", lambda *_args, **_kwargs: "fit-skipped")
    assert resumed.fit() == "fit-skipped"
    recovered = torch.load(directory / "last.pt", map_location="cpu", weights_only=False)
    assert recovered["termination_reason"] == "synthetic_pending_publication"
    assert (directory / ".visibility_checkpoint_transaction").exists() is False


def test_visibility_cli_continue_recovers_pair_before_checking_last(tmp_path, monkeypatch):
    import json

    import scripts.train_visibility as cli
    import car_smart_assist.perception.visibility.trainer as training

    root = tmp_path
    fake_script = root / "scripts/train_visibility.py"
    fake_script.parent.mkdir(parents=True)
    fake_script.write_text("", encoding="utf-8")
    monkeypatch.setattr(cli, "__file__", str(fake_script))
    config = root / "configs/visibility.yaml"
    config.parent.mkdir(parents=True)
    config.write_text(
        yaml.safe_dump({"visibility": {
            "train": {"checkpoint_dir": "artifacts/checkpoints/visibility"},
            "report": {"output_dir": "artifacts/reports/visibility"},
        }}),
        encoding="utf-8",
    )
    directory = root / "artifacts/checkpoints/visibility"
    directory.mkdir(parents=True)
    # These candidates need only be readable checkpoint bytes for CLI recovery;
    # the trainer constructor is replaced so no model is loaded or calibration run.
    _publish_interrupted_pair(directory, monkeypatch, "last.pt")
    assert (directory / "best.pt").is_file()
    assert not (directory / "last.pt").exists()
    seen = {}
    recover = cli.recover_visibility_checkpoint_transaction

    def observe_recovery(path):
        seen["last_missing_before_recovery"] = not (Path(path) / "last.pt").exists()
        result = recover(path)
        seen["last_exists_after_recovery"] = (Path(path) / "last.pt").is_file()
        return result

    monkeypatch.setattr(cli, "recover_visibility_checkpoint_transaction", observe_recovery)

    class History:
        manual_stop = False

        def to_dict(self):
            return {"status": "synthetic"}

    class StubTrainer:
        def __init__(self, cfg, *, project_root, resume):
            seen["cfg"] = cfg
            seen["root"] = project_root
            seen["resume"] = resume
            self.resume_path = resume
            self.calibration = None
            self.seed = 1
            self.split_signature = "synthetic"
            self.split_stats = {}

        def fit(self, *, stop_controller, continue_only):
            seen["continue_only"] = continue_only
            seen["last_exists_during_fit"] = seen["resume"].is_file()
            return History()

    monkeypatch.setattr(cli, "VisibilityTrainer", StubTrainer)
    monkeypatch.setattr("sys.argv", ["train_visibility.py", "--config", "configs/visibility.yaml", "--continue"])
    assert cli.main() == 0
    assert seen["last_missing_before_recovery"] is True
    assert seen["last_exists_after_recovery"] is True
    assert seen["continue_only"] is True
    assert seen["last_exists_during_fit"] is True
    assert seen["resume"] == directory / "last.pt"
    assert training.recover_visibility_checkpoint_transaction(directory) is False
    assert json.loads((root / "artifacts/reports/visibility/train_history.json").read_text(
        encoding="utf-8"
    ))["history"]["status"] == "synthetic"


@pytest.mark.parametrize("stop_during_calibration", [False, True])
def test_epoch_calibration_and_pause_saves_use_transaction_publisher(
    tmp_path, monkeypatch, stop_during_calibration
):
    import car_smart_assist.perception.visibility.trainer as training
    from car_smart_assist.perception.visibility.scorer import CalibrationStats
    from car_smart_assist.utils.stop import StopController

    helper = _load_helper(
        "tests/unit/test_visibility_retrain_training.py", "visibility_trainer_fixture"
    )
    trainer = helper._trainer_with_controlled_validation(tmp_path, monkeypatch, [1.0], epochs=1)
    updates_seen = []
    publish = training._publish_checkpoint_transaction

    def record_updates(directory, updates):
        updates_seen.append(set(updates))
        return publish(directory, updates)

    monkeypatch.setattr(training, "_publish_checkpoint_transaction", record_updates)
    controller = StopController(default_directory=trainer.ckpt_dir)
    if stop_during_calibration:
        def calibrate(_loader):
            result = CalibrationStats(mean=0.01, std=0.002, p95=0.02, p99=0.03, n=4)
            controller.request_stop_file()
            return result

        monkeypatch.setattr(trainer, "_calibrate", calibrate)
    trainer.fit(stop_controller=controller)
    assert updates_seen[0] == {"best.pt", "last.pt"}
    if stop_during_calibration:
        assert updates_seen[-1] == {"last.pt"}
    else:
        assert {"best.pt"} in updates_seen
        assert updates_seen[-1] == {"best.pt"}
    assert (trainer.ckpt_dir / "best.pt").is_file()
    assert (trainer.ckpt_dir / "last.pt").is_file()
