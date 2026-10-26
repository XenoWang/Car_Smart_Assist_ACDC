from __future__ import annotations

import pytest
import torch

from car_smart_assist.perception.visibility.scorer import CalibrationStats
from car_smart_assist.perception.visibility.trainer import (
    VisibilityTrainer,
    _atomic_save,
    _model_state_sha256,
)


def _config(tmp_path, *, epochs=1, seed=19):
    return {
        "device": "cpu",
        "model": {"input_size": (32, 32), "base_channels": 4, "latent_dim": 8},
        "data": {"num_workers": 0},
        "train": {
            "epochs": epochs, "batch_size": 2, "lr": 1e-3, "weight_decay": 0.0,
            "optimizer": "adamw", "amp": False, "dtype": "float32", "augment": False,
            "denoise_sigma": 0.0, "grad_clip_norm": 1.0, "seed": seed,
            "early_stopping": {"enabled": False, "patience": 2, "min_delta": 1e-5},
            "checkpoint_dir": str(tmp_path / "checkpoints"),
        },
    }


def _fit_one_epoch(tmp_path, monkeypatch):
    cfg = _config(tmp_path)
    trainer = VisibilityTrainer(cfg, project_root=tmp_path, resume="none")
    samples = torch.linspace(0, 1, 4 * 3 * 32 * 32).reshape(4, 3, 32, 32)

    class TensorRows:
        def __len__(self):
            return len(samples)

        def __getitem__(self, index):
            return samples[index]

    dataset = TensorRows()
    monkeypatch.setattr(trainer, "_build_datasets", lambda: (dataset, dataset, dataset, []))
    monkeypatch.setattr(trainer, "_eval_loss", lambda _loader: 0.25)
    monkeypatch.setattr(
        trainer, "_calibrate",
        lambda _loader: CalibrationStats(mean=0.01, std=0.002, p95=0.02, p99=0.03, n=4),
    )
    import car_smart_assist.perception.visibility.trainer as training

    save_order = []
    original_save = training._atomic_save

    def track_save(payload, path):
        save_order.append(path.name)
        return original_save(payload, path)

    monkeypatch.setattr(training, "_atomic_save", track_save)
    trainer.fit()
    trainer._checkpoint_save_order = save_order
    return trainer


def test_model_identity_binds_parameters_and_integer_buffers_not_metadata():
    state = {
        "weight": torch.tensor([1.0, 2.0]),
        "counter": torch.tensor([3], dtype=torch.int64),
    }
    first = _model_state_sha256(state)

    changed_weight = {key: value.clone() for key, value in state.items()}
    changed_weight["weight"][0] += 1
    assert _model_state_sha256(changed_weight) != first
    changed_buffer = {key: value.clone() for key, value in state.items()}
    changed_buffer["counter"][0] += 1
    assert _model_state_sha256(changed_buffer) != first


def test_calibration_and_history_metadata_do_not_change_checkpoint_weight_identity(tmp_path):
    trainer = VisibilityTrainer(_config(tmp_path), project_root=tmp_path, resume="none")
    state_identity = _model_state_sha256(trainer.model.state_dict())
    trainer._best_model_sha256 = state_identity
    trainer.calibration = CalibrationStats(mean=0.01, std=0.002, p95=0.02, p99=0.03, n=4)
    trainer.history.train_loss = [1.0]
    before = trainer._make_payload(1, None, None, 0.5)
    trainer.calibration = CalibrationStats(mean=0.05, std=0.01, p95=0.1, p99=0.2, n=9)
    trainer.history.train_loss.append(0.4)
    after = trainer._make_payload(2, None, None, 0.4)
    assert before["best_model_sha256"] == after["best_model_sha256"] == state_identity


def test_training_writes_format3_best_and_last_bound_to_best_state(tmp_path, monkeypatch):
    trainer = _fit_one_epoch(tmp_path, monkeypatch)
    best = torch.load(tmp_path / "checkpoints/best.pt", map_location="cpu", weights_only=False)
    last = torch.load(tmp_path / "checkpoints/last.pt", map_location="cpu", weights_only=False)
    assert best["format_version"] == last["format_version"] == 3
    assert best["best_model_sha256"] == last["best_model_sha256"]
    assert best["best_model_sha256"] == _model_state_sha256(best["model_state"])
    assert best["best_epoch"] == last["best_epoch"] == 1
    assert trainer.history.best_epoch == 1
    assert trainer._checkpoint_save_order.index("best.pt") < trainer._checkpoint_save_order.index("last.pt")


def test_matching_format3_best_allows_restore(tmp_path, monkeypatch):
    original = _fit_one_epoch(tmp_path, monkeypatch)
    checkpoint_path = tmp_path / "checkpoints/last.pt"
    resumed = VisibilityTrainer(
        _config(tmp_path, epochs=2), project_root=tmp_path, resume=checkpoint_path
    )
    resumed.split_signature = original.split_signature
    optimizer = torch.optim.AdamW(resumed.model.parameters(), lr=1e-3)
    epoch = resumed._restore(optimizer, None)
    assert epoch == 1
    best = torch.load(tmp_path / "checkpoints/best.pt", map_location="cpu", weights_only=False)
    assert resumed._best_model_sha256 == _model_state_sha256(best["model_state"])


def test_tampered_best_state_is_rejected_and_last_is_unchanged(tmp_path, monkeypatch):
    original = _fit_one_epoch(tmp_path, monkeypatch)
    checkpoint = tmp_path / "checkpoints/last.pt"
    last_before = checkpoint.read_bytes()
    best_path = tmp_path / "checkpoints/best.pt"
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    first_key = next(iter(best["model_state"]))
    best["model_state"][first_key] = best["model_state"][first_key].clone()
    best["model_state"][first_key].view(-1)[0] += 1
    _atomic_save(best, best_path)

    resumed = VisibilityTrainer(_config(tmp_path, epochs=2), project_root=tmp_path, resume=checkpoint)
    resumed.split_signature = original.split_signature
    optimizer = torch.optim.AdamW(resumed.model.parameters(), lr=1e-3)
    with pytest.raises(ValueError, match="权重身份或指标不匹配"):
        resumed._restore(optimizer, None)
    assert checkpoint.read_bytes() == last_before


@pytest.mark.parametrize("field", [
    "split_signature", "seed", "model_cfg", "best_epoch", "best_val_loss",
    "current_val_loss", "epoch", "best_model_sha256",
])
def test_best_identity_metadata_mismatch_is_rejected(tmp_path, monkeypatch, field):
    _fit_one_epoch(tmp_path, monkeypatch)
    checkpoint = tmp_path / "checkpoints/last.pt"
    best_path = tmp_path / "checkpoints/best.pt"
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    value = best[field]
    if field == "model_cfg":
        best[field] = {**value, "latent_dim": value["latent_dim"] + 1}
    elif field == "best_model_sha256":
        best[field] = "0" * 64
    elif field == "split_signature":
        best[field] = "different-split"
    else:
        best[field] = value + 1
    _atomic_save(best, best_path)

    reference = torch.load(checkpoint, map_location="cpu", weights_only=False)
    trainer = VisibilityTrainer(_config(tmp_path, epochs=2), project_root=tmp_path, resume=checkpoint)
    with pytest.raises(ValueError, match="不匹配"):
        trainer._load_verified_best(reference)


def test_external_last_cannot_resume_without_best_in_destination_directory(tmp_path, monkeypatch):
    original = _fit_one_epoch(tmp_path / "source", monkeypatch)
    source_last = tmp_path / "source/checkpoints/last.pt"
    source_bytes = source_last.read_bytes()
    destination = tmp_path / "empty-destination"
    resumed = VisibilityTrainer(
        _config(destination, epochs=2), project_root=destination, resume=source_last
    )
    resumed.split_signature = original.split_signature
    optimizer = torch.optim.AdamW(resumed.model.parameters(), lr=1e-3)
    with pytest.raises(ValueError, match="历史best缺失"):
        resumed._restore(optimizer, None)
    assert source_last.read_bytes() == source_bytes
    assert not (destination / "checkpoints/best.pt").exists()


def test_missing_historical_best_is_rejected_without_replacing_last(tmp_path, monkeypatch):
    _fit_one_epoch(tmp_path, monkeypatch)
    checkpoint = tmp_path / "checkpoints/last.pt"
    last_before = checkpoint.read_bytes()
    (tmp_path / "checkpoints/best.pt").unlink()
    resumed = VisibilityTrainer(_config(tmp_path, epochs=2), project_root=tmp_path, resume=checkpoint)
    with pytest.raises(ValueError, match="历史best缺失"):
        resumed._load_verified_best(torch.load(checkpoint, map_location="cpu", weights_only=False))
    assert checkpoint.read_bytes() == last_before
    assert not (tmp_path / "checkpoints/best.pt").exists()


def test_format2_resume_is_rejected_while_legacy_weight_inference_remains_available(tmp_path, monkeypatch):
    _fit_one_epoch(tmp_path, monkeypatch)
    checkpoint = tmp_path / "checkpoints/last.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["format_version"] = 2
    payload.pop("best_model_sha256", None)
    _atomic_save(payload, checkpoint)
    before = checkpoint.read_bytes()

    with pytest.raises(ValueError, match="没有可核验的历史best身份"):
        VisibilityTrainer(_config(tmp_path, epochs=2), project_root=tmp_path, resume=checkpoint)
    assert checkpoint.read_bytes() == before

    from car_smart_assist.perception.visibility.scorer import VisibilityScorer

    scorer = VisibilityScorer.from_checkpoint(checkpoint, cfg=payload["model_cfg"])
    assert scorer.input_size == tuple(payload["model_cfg"]["input_size"])
    assert scorer.split_signature == payload.get("split_signature")
