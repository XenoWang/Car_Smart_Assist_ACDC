from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
import torch

from car_smart_assist.perception.visibility.scorer import CalibrationStats
from car_smart_assist.perception.visibility.trainer import VisibilityTrainer, _seed_worker


def _config(tmp_path, *, epochs=5, seed=42, patience=2, min_delta=1e-5):
    return {
        "device": "cpu",
        "model": {"input_size": (32, 32), "base_channels": 4, "latent_dim": 8},
        "data": {"num_workers": 0},
        "train": {
            "epochs": epochs,
            "batch_size": 2,
            "lr": 1e-3,
            "weight_decay": 0.0,
            "optimizer": "adamw",
            "amp": False,
            "dtype": "float32",
            "augment": False,
            "denoise_sigma": 0.0,
            "grad_clip_norm": 1.0,
            "seed": seed,
            "early_stopping": {
                "enabled": True,
                "patience": patience,
                "min_delta": min_delta,
            },
            "checkpoint_dir": str(tmp_path / "checkpoints"),
        },
    }


def _trainer_with_controlled_validation(tmp_path, monkeypatch, losses, *, epochs=5):
    trainer = VisibilityTrainer(_config(tmp_path, epochs=epochs), project_root=tmp_path, resume="none")
    samples = torch.linspace(0, 1, 4 * 3 * 32 * 32).reshape(4, 3, 32, 32)
    class TensorRows:
        def __len__(self):
            return len(samples)

        def __getitem__(self, index):
            return samples[index]

    dataset = TensorRows()
    monkeypatch.setattr(trainer, "_build_datasets", lambda: (dataset, dataset, dataset, []))
    values = iter(losses)
    monkeypatch.setattr(trainer, "_eval_loss", lambda _loader: next(values))
    monkeypatch.setattr(
        trainer,
        "_calibrate",
        lambda _loader: CalibrationStats(mean=0.01, std=0.002, p95=0.02, p99=0.03, n=4),
    )
    return trainer


def test_fresh_ae_initialization_and_loader_order_follow_fixed_seed(tmp_path):
    cfg = _config(tmp_path, seed=42)
    first = VisibilityTrainer(cfg, project_root=tmp_path, resume="none")
    first_weights = {name: value.clone() for name, value in first.model.state_dict().items()}
    first_python = random.random()
    first_numpy = np.random.random()
    first_order = torch.randperm(16, generator=first.loader_generator)

    second = VisibilityTrainer(cfg, project_root=tmp_path, resume="none")
    second_weights = second.model.state_dict()
    second_python = random.random()
    second_numpy = np.random.random()
    second_order = torch.randperm(16, generator=second.loader_generator)

    assert all(torch.equal(first_weights[name], second_weights[name]) for name in first_weights)
    assert first_python == second_python
    assert first_numpy == second_numpy
    assert torch.equal(first_order, second_order)
    assert first.resume_path is None and second.resume_path is None


def test_worker_seed_initializes_python_and_numpy_from_torch_worker_seed():
    torch.manual_seed(9321)
    _seed_worker(0)
    first = (random.random(), np.random.random())

    torch.manual_seed(9321)
    _seed_worker(0)
    second = (random.random(), np.random.random())

    assert first == second


def test_recording_family_algorithm_version_is_bound_into_split_signature(
    tmp_path, monkeypatch
):
    import car_smart_assist.perception.visibility.dataset as dataset
    import car_smart_assist.perception.visibility.trainer as training

    cfg = _config(tmp_path)
    cfg["data"].update(
        {
            "acdc_root": "acdc",
            "split_by_sequence": True,
            "train_ratio": 0.8,
            "val_ratio": 0.05,
            "calib_ratio": 0.05,
            "test_ratio": 0.1,
            "split_seed": 0,
            "split_mode": "fixed",
        }
    )
    acdc_root = tmp_path / "acdc"
    paths = []
    for index in range(100):
        sequence = f"seq{index // 5:02d}"
        path = (
            acdc_root
            / "rgb_anon/fog/train_ref"
            / sequence
            / f"frame_{index:03d}_rgb_ref_anon.png"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")
        paths.append(path)

    monkeypatch.setattr(dataset, "list_ref_images", lambda *_args, **_kwargs: paths)
    monkeypatch.setattr(dataset, "sequence_of", lambda path: Path(path).parent.name)
    monkeypatch.setattr(dataset, "ensure_cache", lambda *_args: None)
    monkeypatch.setattr(
        training,
        "VisibilityImageDataset",
        lambda _paths, _size, *, indices, **_kwargs: [int(index) for index in indices],
    )
    monkeypatch.setattr(training, "training_config", lambda _cfg: {"augment": False})

    captured_payload = {}
    original_dumps = training.json.dumps

    def capture_split_payload(payload, *args, **kwargs):
        if isinstance(payload, dict) and "split_algorithm" in payload:
            captured_payload.update(payload)
        return original_dumps(payload, *args, **kwargs)

    monkeypatch.setattr(training.json, "dumps", capture_split_payload)
    trainer = VisibilityTrainer(cfg, project_root=tmp_path, resume="none")

    trainer._build_datasets()

    assert captured_payload["split_algorithm"] == "recording-family-greedy-nonempty-sse-v1"
    assert len(captured_payload["groups"]) == 100
    assert captured_payload["train"] and captured_payload["val"]
    assert captured_payload["calib"] and captured_payload["test"]


def test_last_checkpoint_restores_python_numpy_torch_and_loader_rng(tmp_path, monkeypatch):
    trainer = _trainer_with_controlled_validation(
        tmp_path, monkeypatch, [1.0, 0.8], epochs=2
    )
    trainer.fit()
    checkpoint = tmp_path / "checkpoints/last.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    rng = payload["rng_state"]
    assert set(rng) == {"python", "numpy", "torch", "cuda", "loader"}

    expected_python_state = random.getstate()
    expected_numpy_state = np.random.get_state()
    expected_torch_state = torch.get_rng_state()
    random.setstate(rng["python"])
    expected_python = random.random()
    np.random.set_state(rng["numpy"])
    expected_numpy = np.random.random()
    torch.set_rng_state(rng["torch"])
    expected_torch = torch.rand(1)
    expected_loader_generator = torch.Generator()
    expected_loader_generator.set_state(rng["loader"])
    expected_loader = torch.randperm(16, generator=expected_loader_generator)
    random.setstate(expected_python_state)
    np.random.set_state(expected_numpy_state)
    torch.set_rng_state(expected_torch_state)

    resumed = VisibilityTrainer(
        _config(tmp_path, epochs=3), project_root=tmp_path, resume=checkpoint
    )
    optimizer = torch.optim.AdamW(resumed.model.parameters())
    resumed._restore(optimizer, None)

    assert random.random() == expected_python
    assert np.random.random() == expected_numpy
    assert torch.equal(torch.rand(1), expected_torch)
    assert torch.equal(
        torch.randperm(16, generator=resumed.loader_generator), expected_loader
    )


def test_visibility_early_stopping_tracks_terminal_epoch_and_restores_best(
    tmp_path, monkeypatch
):
    trainer = _trainer_with_controlled_validation(
        tmp_path, monkeypatch, [1.0, 1.0, 1.0, 1.0, 1.0], epochs=5
    )
    history = trainer.fit()

    last = torch.load(tmp_path / "checkpoints/last.pt", map_location="cpu", weights_only=False)
    best = torch.load(tmp_path / "checkpoints/best.pt", map_location="cpu", weights_only=False)
    assert history.stopped_early is True
    assert history.termination_reason == "early_stopping"
    assert len(history.train_loss) == 3
    assert history.best_epoch == 1
    assert last["epoch"] == 3 and last["planned_epochs"] == 5
    assert last["training_complete"] is True
    assert last["termination_reason"] == "early_stopping"
    assert last["early_stopping_state"] == {"wait": 2, "stopped": True}
    assert best["epoch"] == history.best_epoch
    assert best["termination_reason"] == "early_stopping"
    assert all(
        torch.equal(trainer.model.state_dict()[name], best["model_state"][name])
        for name in best["model_state"]
    )


def test_visibility_early_stopping_distinguishes_budget_completion(tmp_path, monkeypatch):
    trainer = _trainer_with_controlled_validation(
        tmp_path, monkeypatch, [1.0, 0.5], epochs=2
    )
    history = trainer.fit()

    last = torch.load(tmp_path / "checkpoints/last.pt", map_location="cpu", weights_only=False)
    assert history.stopped_early is False
    assert history.termination_reason == "budget_completed"
    assert len(history.train_loss) == 2
    assert last["epoch"] == 2 and last["planned_epochs"] == 2
    assert last["termination_reason"] == "budget_completed"

