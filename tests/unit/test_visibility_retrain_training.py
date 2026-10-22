from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
import torch

from car_smart_assist.perception.visibility.scorer import CalibrationStats
from car_smart_assist.perception.visibility.trainer import (
    VisibilityTrainer,
    _model_state_sha256,
    _seed_worker,
)


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
    assert best["termination_reason"] is None  # best恢复点尚未触发后续早停
    assert best["final_summary"]["termination_reason"] == "early_stopping"
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



@pytest.mark.parametrize("bad_lr", [0.0, -1.0, float("nan"), float("inf"), float("-inf")])
def test_resume_rejects_invalid_learning_rate_without_changing_checkpoint(
    tmp_path, monkeypatch, bad_lr
):
    trainer = _trainer_with_controlled_validation(tmp_path, monkeypatch, [0.5], epochs=1)
    trainer.fit()
    checkpoint = tmp_path / "checkpoints/last.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["optimizer_state"]["param_groups"][0]["lr"] = bad_lr
    torch.save(payload, checkpoint)
    before = checkpoint.read_bytes()

    resumed = VisibilityTrainer(
        _config(tmp_path, epochs=2), project_root=tmp_path, resume=checkpoint
    )
    monkeypatch.setattr(resumed, "_build_datasets", trainer._build_datasets)
    with pytest.raises(ValueError, match="学习率已为0或无效"):
        resumed.fit()
    assert checkpoint.read_bytes() == before

class _TrainingInterrupted(RuntimeError):
    pass


def _interrupt_after_saved_last(monkeypatch, epoch):
    import car_smart_assist.perception.visibility.trainer as training

    original = training._atomic_save
    interrupted = False

    def save_then_interrupt(payload, path):
        nonlocal interrupted
        original(payload, path)
        if path.name == "last.pt" and payload["epoch"] == epoch and not interrupted:
            interrupted = True
            raise _TrainingInterrupted(f"simulated interruption after epoch {epoch} last save")

    monkeypatch.setattr(training, "_atomic_save", save_then_interrupt)
    return lambda: interrupted


def _assert_nested_equal(actual, expected):
    if isinstance(actual, torch.Tensor):
        assert isinstance(expected, torch.Tensor)
        assert torch.equal(actual, expected)
    elif isinstance(actual, np.ndarray):
        assert isinstance(expected, np.ndarray)
        assert np.array_equal(actual, expected)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            _assert_nested_equal(actual[key], expected[key])
    elif isinstance(actual, (tuple, list)):
        assert type(actual) is type(expected)
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            _assert_nested_equal(left, right)
    else:
        assert actual == expected


def test_same_budget_interruption_restores_exact_training_trajectory(tmp_path, monkeypatch):
    uninterrupted_dir = tmp_path / "uninterrupted"
    interrupted_dir = tmp_path / "interrupted"
    uninterrupted = _trainer_with_controlled_validation(
        uninterrupted_dir, monkeypatch, [1.0, 0.9, 0.8, 0.7], epochs=4
    )
    uninterrupted.fit()
    expected = torch.load(
        uninterrupted_dir / "checkpoints/last.pt", map_location="cpu", weights_only=False
    )

    first_half = _trainer_with_controlled_validation(
        interrupted_dir, monkeypatch, [1.0, 0.9, 0.8, 0.7], epochs=4
    )
    did_interrupt = _interrupt_after_saved_last(monkeypatch, 2)
    with pytest.raises(_TrainingInterrupted, match="after epoch 2"):
        first_half.fit()
    assert did_interrupt()
    checkpoint = interrupted_dir / "checkpoints/last.pt"
    saved_at_interrupt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert saved_at_interrupt["epoch"] == 2

    resumed = VisibilityTrainer(
        _config(interrupted_dir, epochs=4), project_root=interrupted_dir, resume=checkpoint
    )
    samples = torch.linspace(0, 1, 4 * 3 * 32 * 32).reshape(4, 3, 32, 32)

    class TensorRows:
        def __len__(self):
            return len(samples)

        def __getitem__(self, index):
            return samples[index]

    dataset = TensorRows()
    monkeypatch.setattr(resumed, "_build_datasets", lambda: (dataset, dataset, dataset, []))
    values = iter([0.8, 0.7])
    monkeypatch.setattr(resumed, "_eval_loss", lambda _loader: next(values))
    monkeypatch.setattr(
        resumed,
        "_calibrate",
        lambda _loader: CalibrationStats(mean=0.01, std=0.002, p95=0.02, p99=0.03, n=4),
    )
    resumed.fit()
    actual = torch.load(checkpoint, map_location="cpu", weights_only=False)

    _assert_nested_equal(actual["model_state"], expected["model_state"])
    _assert_nested_equal(actual["optimizer_state"], expected["optimizer_state"])
    _assert_nested_equal(actual["rng_state"], expected["rng_state"])
    for key in ("train_loss", "val_loss", "epoch", "planned_epochs", "best_epoch", "best_val_loss"):
        assert actual[key] == expected[key]
    assert actual["scheduler_state"]["T_max"] == expected["scheduler_state"]["T_max"] == 4
    assert actual["scheduler_state"]["last_epoch"] == expected["scheduler_state"]["last_epoch"] == 4
    assert actual["optimizer_state"]["param_groups"][0]["lr"] == pytest.approx(
        expected["optimizer_state"]["param_groups"][0]["lr"]
    )


def test_early_stopped_resume_extends_from_current_lr_and_optimizer_state(
    tmp_path, monkeypatch
):

    trainer = _trainer_with_controlled_validation(
        tmp_path, monkeypatch, [1.0, 1.0], epochs=4
    )
    trainer.cfg["train"]["early_stopping"]["patience"] = 1
    trainer.fit()
    checkpoint = tmp_path / "checkpoints/last.pt"
    stopped = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert stopped["epoch"] == 2
    assert stopped["training_complete"] is True
    restored_lr = stopped["optimizer_state"]["param_groups"][0]["lr"]
    assert restored_lr > 0

    step_lrs = []
    steps_before = []
    original_step = torch.optim.AdamW.step

    def record_step(optimizer, *args, **kwargs):
        step_lrs.append(tuple(float(group["lr"]) for group in optimizer.param_groups))
        current = [
            float(state["step"].item())
            for state in optimizer.state.values()
            if "step" in state
        ]
        steps_before.append(max(current, default=0.0))
        return original_step(optimizer, *args, **kwargs)

    monkeypatch.setattr(torch.optim.AdamW, "step", record_step)
    resumed = _trainer_with_controlled_validation(
        tmp_path, monkeypatch, [0.9, 0.8, 0.7, 0.6], epochs=6
    )
    resumed.cfg["train"]["early_stopping"]["patience"] = 10
    resumed.resume_path = checkpoint
    history = resumed.fit()

    assert history.termination_reason == "budget_completed"
    assert history.planned_epochs == 6
    assert len(step_lrs) == 8  # 4 resumed epochs, 2 batches per epoch
    epoch_lrs = [step_lrs[index][0] for index in range(0, len(step_lrs), 2)]
    assert epoch_lrs[0] == pytest.approx(restored_lr)
    assert all(left >= right for left, right in zip(epoch_lrs, epoch_lrs[1:]))
    assert steps_before[0] == pytest.approx(4.0)  # AdamW state continued from epoch 2
    final = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert final["epoch"] == 6 and final["planned_epochs"] == 6
    assert final["optimizer_state"]["param_groups"][0]["lr"] == pytest.approx(0.0, abs=1e-12)
    assert final["scheduler_state"]["T_max"] == 4
    assert final["scheduler_state"]["last_epoch"] == 4


def test_interrupted_extra_budget_segment_does_not_add_extra_epochs_again(
    tmp_path, monkeypatch
):
    initial = _trainer_with_controlled_validation(
        tmp_path, monkeypatch, [1.0, 1.0], epochs=4
    )
    initial.cfg["train"]["early_stopping"]["patience"] = 1
    initial.fit()
    checkpoint = tmp_path / "checkpoints/last.pt"
    initial_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert initial_payload["epoch"] == 2
    assert initial_payload["optimizer_state"]["param_groups"][0]["lr"] > 0

    def configured_resume(losses):
        resumed = _trainer_with_controlled_validation(
            tmp_path, monkeypatch, losses, epochs=10
        )
        resumed.cfg["train"]["resume_extra_epochs"] = 4
        resumed.cfg["train"]["early_stopping"]["patience"] = 10
        resumed.resume_path = checkpoint
        return resumed

    interrupted_at_four = configured_resume([0.9, 0.8])
    did_interrupt_four = _interrupt_after_saved_last(monkeypatch, 4)
    with pytest.raises(_TrainingInterrupted, match="after epoch 4"):
        interrupted_at_four.fit()
    assert did_interrupt_four()
    payload_four = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload_four["epoch"] == 4 and payload_four["planned_epochs"] == 6
    assert payload_four["training_complete"] is False

    interrupted_at_five = configured_resume([0.7])
    did_interrupt_five = _interrupt_after_saved_last(monkeypatch, 5)
    with pytest.raises(_TrainingInterrupted, match="after epoch 5"):
        interrupted_at_five.fit()
    assert did_interrupt_five()
    payload_five = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload_five["epoch"] == 5 and payload_five["planned_epochs"] == 6

    final_resume = configured_resume([0.6])
    history = final_resume.fit()
    final = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert history.termination_reason == "budget_completed"
    assert final["epoch"] == 6
    assert final["planned_epochs"] == 6
    assert len(final["train_loss"]) == 6
    assert len(final["val_loss"]) == 6


def test_no_improvement_resume_restores_original_best_state_and_metrics(tmp_path, monkeypatch):
    initial = _trainer_with_controlled_validation(
        tmp_path, monkeypatch, [1.0, 1.0], epochs=4
    )
    initial.cfg["train"]["early_stopping"]["patience"] = 1
    initial.fit()
    checkpoint = tmp_path / "checkpoints/last.pt"
    interrupted = torch.load(checkpoint, map_location="cpu", weights_only=False)
    original_best = torch.load(
        tmp_path / "checkpoints/best.pt", map_location="cpu", weights_only=False
    )
    assert interrupted["epoch"] == 2
    assert interrupted["optimizer_state"]["param_groups"][0]["lr"] > 0
    assert original_best["best_epoch"] == 1
    original_state = {
        name: value.clone() for name, value in original_best["model_state"].items()
    }
    original_identity = original_best["best_model_sha256"]
    original_best_loss = original_best["best_val_loss"]

    resumed = _trainer_with_controlled_validation(
        tmp_path, monkeypatch, [1.1, 1.1, 1.1, 1.1], epochs=6
    )
    resumed.cfg["train"]["early_stopping"]["enabled"] = False
    resumed.resume_path = checkpoint
    history = resumed.fit()

    final_best = torch.load(
        tmp_path / "checkpoints/best.pt", map_location="cpu", weights_only=False
    )
    final_last = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert final_last["epoch"] == 6
    assert len(final_last["val_loss"][-4:]) == 4
    assert all(value > original_best_loss for value in final_last["val_loss"][-4:])
    assert history.best_epoch == final_best["best_epoch"] == final_last["best_epoch"] == 1
    assert final_best["best_val_loss"] == final_last["best_val_loss"] == original_best_loss
    assert final_best["best_model_sha256"] == final_last["best_model_sha256"] == original_identity
    assert _model_state_sha256(final_best["model_state"]) == original_identity
    assert all(
        torch.equal(final_best["model_state"][name], state)
        for name, state in original_state.items()
    )
    assert all(
        torch.equal(resumed.model.state_dict()[name], state)
        for name, state in original_state.items()
    )
    assert any(
        not torch.equal(final_last["model_state"][name], state)
        for name, state in original_state.items()
    )
def test_best_checkpoint_keeps_best_resume_history_and_separate_final_summary(
    tmp_path, monkeypatch
):
    trainer = _trainer_with_controlled_validation(
        tmp_path, monkeypatch, [1.0, 1.0, 1.0, 1.0], epochs=4
    )
    trainer.cfg["train"]["early_stopping"]["patience"] = 1
    history = trainer.fit()
    best = torch.load(tmp_path / "checkpoints/best.pt", map_location="cpu", weights_only=False)
    assert best["epoch"] == history.best_epoch == best["best_epoch"]
    assert len(best["train_loss"]) == len(best["val_loss"]) == best["epoch"]
    assert best["final_summary"]["training_complete"] is True
    assert len(best["final_summary"]["train_loss"]) == len(history.train_loss)
    assert len(best["final_summary"]["val_loss"]) == len(history.val_loss)
    assert best["final_summary"]["termination_reason"] == history.termination_reason
