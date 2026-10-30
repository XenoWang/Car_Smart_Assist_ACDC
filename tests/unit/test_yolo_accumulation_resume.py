"""Actual YOLO optimizer-step timing across accumulated training and resume."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import BatchSampler, SequentialSampler


SELECTION_CONTRACT = {
    "schema_version": 1,
    "postprocess_version": "best_class_threshold_v1",
    "ap_mode": "ap_sweep",
    "sweep_confidence": 0.001,
    "imgsz": 960,
    "nms_iou": 0.5,
    "max_det": 100,
    "max_nms": 30000,
    "max_wh": 7680,
    "agnostic_nms": False,
    "class_selection": "original_best_class_only",
    "threshold_comparison": ">=",
    "task": "detect",
    "fitness": "0.1*mAP50+0.9*mAP50_95",
    "joint_selection": None,
}

TRAINING_SEGMENT = {
    "start_epoch": 0,
    "target_epochs": 2,
    "resume_extra_epochs": None,
    "epochs_override": None,
    "early_stopping_policy": "preserve_history",
}


@pytest.fixture(autouse=True)
def _restore_torch_rng_after_cpu_training():
    state = torch.get_rng_state().clone()
    try:
        yield
    finally:
        torch.set_rng_state(state)


def _vendor_batch():
    return {"img": torch.full((2, 3, 64, 64), 128, dtype=torch.uint8)}


class _TinyLoader:
    def __init__(self, batches, batch_size):
        self.dataset = tuple(range(batches * batch_size))
        self.batch_size = batch_size
        self.drop_last = False
        self.sampler = SequentialSampler(self.dataset)
        self.batch_sampler = BatchSampler(self.sampler, batch_size, drop_last=False)
        self._batches = batches

    def __len__(self):
        return self._batches


def _make_trainer(tmp_path, *, batches=3, batch_size=2, nbs=4,
                  warmup_epochs=0, initial_weight=1.0, start_epoch=0,
                  stop_controller=None, scaler_enabled=False):
    from ultralytics.utils.torch_utils import ModelEMA

    from car_smart_assist.perception.yolo_training import (
        attach_training_state_callbacks,
        detection_trainer_for_resume,
    )

    trainer_type = detection_trainer_for_resume(None, SELECTION_CONTRACT)
    trainer = trainer_type.__new__(trainer_type)
    trainer.model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        trainer.model.weight.fill_(initial_weight)
    trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.1, momentum=0.9)
    trainer.scheduler = torch.optim.lr_scheduler.LambdaLR(trainer.optimizer, lambda _: 1.0)
    trainer.scaler = torch.amp.GradScaler(
        "cpu", enabled=scaler_enabled, init_scale=8.0, growth_interval=2
    )
    trainer.ema = ModelEMA(trainer.model)
    trainer.train_loader = _TinyLoader(batches, batch_size)
    trainer.batch_size = batch_size
    trainer.args = SimpleNamespace(
        nbs=nbs, warmup_epochs=warmup_epochs, multi_scale=False, imgsz=64
    )
    trainer.amp = scaler_enabled
    trainer.accumulate = max(round(nbs / batch_size), 1)
    trainer.epoch = start_epoch
    trainer.epochs = 2
    trainer.start_epoch = start_epoch
    trainer.device = torch.device("cpu")
    trainer.stop = False
    trainer.best_fitness = 0.2
    trainer.fitness = 0.2
    trainer.stopper = SimpleNamespace(
        best_fitness=0.2, best_epoch=0, patience=3, possible_stop=False
    )
    trainer.wdir = tmp_path / "weights"
    trainer.wdir.mkdir(parents=True, exist_ok=True)
    trainer.last = trainer.wdir / "last.pt"
    trainer.best = trainer.wdir / "best.pt"
    trainer.last.write_bytes(b"tiny-linear last checkpoint")
    trainer.best.write_bytes(b"tiny-linear paired best checkpoint")

    callbacks = {}
    model_wrapper = SimpleNamespace(
        add_callback=lambda name, callback: callbacks.update({name: callback})
    )
    attach_training_state_callbacks(
        model_wrapper,
        "tiny-linear-dataset-v1",
        None,
        selection_contract=SELECTION_CONTRACT,
        training_segment=TRAINING_SEGMENT,
        stop_controller=stop_controller,
    )
    return trainer, callbacks


def _record_optimizer_steps(trainer, recorded):
    optimizer_step = trainer.optimizer.step

    def record(*args, **kwargs):
        recorded.append(trainer._project_current_batch)
        return optimizer_step(*args, **kwargs)

    trainer.optimizer.step = record


def _run_epoch(trainer, callbacks, epoch, targets, vendor_last_opt_step):
    """Drive pinned vendor batch ordering around the real project trainer hooks."""
    trainer.epoch = epoch
    callbacks["on_train_epoch_start"](trainer)
    batches_per_epoch = len(trainer.train_loader)
    for local_batch, target in enumerate(targets):
        batch_index = epoch * batches_per_epoch + local_batch
        actual_accumulate = max(round(trainer.args.nbs / trainer.batch_size), 1)
        trainer.accumulate = actual_accumulate
        trainer.preprocess_batch(_vendor_batch())
        weight = trainer.model.weight.squeeze()
        loss = 0.5 * (weight - target).square() / actual_accumulate
        loss.backward()

        # Pinned Ultralytics 8.3.162 caller contract. The project gate itself
        # is exercised only through ProjectDetectionTrainer above.
        if batch_index - vendor_last_opt_step >= trainer.accumulate:
            trainer.optimizer_step()
            vendor_last_opt_step = batch_index
        callbacks["on_train_batch_end"](trainer)
    return vendor_last_opt_step


def _start_vendor_training(trainer, callbacks):
    callbacks["on_train_start"](trainer)
    # BaseTrainer._do_train calls zero_grad after on_train_start and before
    # the first on_train_epoch_start callback.
    trainer.optimizer.zero_grad(set_to_none=True)


def test_continuous_and_stopped_resume_match_hand_calculated_optimizer_trajectory(tmp_path):
    from car_smart_assist.perception.yolo_training import (
        attach_training_state_callbacks,
        load_training_state,
    )

    targets = [0.0, 2.0, 2.0, -1.0, 3.0, 1.0]

    continuous_dir = tmp_path / "continuous"
    continuous_dir.mkdir()
    continuous, continuous_callbacks = _make_trainer(continuous_dir)
    continuous_steps = []
    _record_optimizer_steps(continuous, continuous_steps)
    _start_vendor_training(continuous, continuous_callbacks)
    vendor_last = _run_epoch(
        continuous, continuous_callbacks, 0, targets[:3], -1
    )
    _run_epoch(continuous, continuous_callbacks, 1, targets[3:], vendor_last)

    stopped_dir = tmp_path / "stopped"
    stopped_dir.mkdir()
    from car_smart_assist.utils.stop import StopController

    stop_controller = StopController(default_directory=stopped_dir / "weights")
    stopped, stopped_callbacks = _make_trainer(
        stopped_dir, stop_controller=stop_controller
    )
    stopped_steps = []
    _record_optimizer_steps(stopped, stopped_steps)
    _start_vendor_training(stopped, stopped_callbacks)
    _run_epoch(stopped, stopped_callbacks, 0, targets[:3], -1)
    stop_controller.request()
    stopped_callbacks["on_fit_epoch_end"](stopped)
    assert stopped._project_manual_stop and stopped.stop
    stopped_callbacks["on_model_save"](stopped)

    state_path = stopped.wdir / "last_training_state.pt"
    last = torch.load(state_path, map_location="cpu", weights_only=True)
    # This independent disk load is immutable while CPU Optimizer.load_state_dict
    # may share state tensors with the mapping passed to it (pitfall P36).
    independent_disk_oracle = torch.load(
        state_path, map_location="cpu", weights_only=True
    )
    assert last["accumulation"]["last_processed_batch"] == 2
    assert last["manual_stop"] is True
    assert last["accumulation"]["last_optimizer_attempt"] == 1
    assert torch.equal(
        last["accumulation"]["pending_gradients"]["weight"],
        torch.tensor([[-0.5]]),
    )
    assert stopped_steps == [1]

    resume_state = load_training_state(
        state_path, stopped.last, "tiny-linear-dataset-v1", SELECTION_CONTRACT
    )
    resumed, resumed_callbacks = _make_trainer(
        stopped_dir, initial_weight=99.0, start_epoch=1
    )
    resumed_steps = []
    _record_optimizer_steps(resumed, resumed_steps)
    resumed_wrapper = SimpleNamespace(
        add_callback=lambda name, callback: resumed_callbacks.update({name: callback})
    )
    attach_training_state_callbacks(
        resumed_wrapper,
        "tiny-linear-dataset-v1",
        resume_state,
        selection_contract=SELECTION_CONTRACT,
        training_segment=TRAINING_SEGMENT,
    )
    resumed_callbacks["on_train_start"](resumed)
    resumed.optimizer.zero_grad(set_to_none=True)
    assert resumed.model.weight.grad is None
    _run_epoch(resumed, resumed_callbacks, 1, targets[3:], -1)

    # Independent hand calculation, lr=.1 and momentum=.9:
    # batches 0/1 average to 0 (w=1); pending batch 2 contributes -1 to
    # batch 3's +2, so the next mean is .5 (w=.95); batches 4/5 average -1.05,
    # making the momentum buffer -.6 and final weight 1.01.
    assert continuous_steps == [1, 3, 5]
    assert stopped_steps + resumed_steps == [1, 3, 5]
    torch.testing.assert_close(
        continuous.model.weight, torch.tensor([[1.01]]), rtol=0, atol=1e-6
    )
    torch.testing.assert_close(
        resumed.model.weight, continuous.model.weight, rtol=0, atol=0
    )
    parameter_id = independent_disk_oracle["optimizer"]["param_groups"][0]["params"][0]
    torch.testing.assert_close(
        continuous.optimizer.state[continuous.model.weight]["momentum_buffer"],
        torch.tensor([[-0.6]]), rtol=0, atol=1e-6,
    )
    assert torch.equal(
        independent_disk_oracle["optimizer"]["state"][parameter_id]["momentum_buffer"],
        torch.tensor([[0.0]]),
    )
    torch.testing.assert_close(
        resumed.optimizer.state[resumed.model.weight]["momentum_buffer"],
        torch.tensor([[-0.6]]), rtol=0, atol=1e-6,
    )
    assert continuous.ema.updates == resumed.ema.updates == 3
    torch.testing.assert_close(
        continuous.ema.ema.weight, resumed.ema.ema.weight, rtol=0, atol=0
    )


def test_real_project_trainer_tracks_vendor_warmup_accumulate_changes(tmp_path):
    trainer, callbacks = _make_trainer(
        tmp_path, batches=20, nbs=8, warmup_epochs=1
    )
    actual_steps = []
    _record_optimizer_steps(trainer, actual_steps)
    _start_vendor_training(trainer, callbacks)
    callbacks["on_train_epoch_start"](trainer)

    vendor_last_opt_step = -1
    for batch_index in range(20):
        # Pinned vendor warmup rule supplies A to preprocess_batch. This
        # expected schedule is independent of the project's temporary gate.
        trainer.accumulate = max(
            1,
            int(np.interp(batch_index, [0, 100], [1, 4]).round()),
        )
        actual_accumulate = trainer.accumulate
        trainer.preprocess_batch(_vendor_batch())
        (0.5 * trainer.model.weight.square().sum() / actual_accumulate).backward()
        if batch_index - vendor_last_opt_step >= trainer.accumulate:
            trainer.optimizer_step()
            vendor_last_opt_step = batch_index
        callbacks["on_train_batch_end"](trainer)

    # Vendor warmup gives A=1 through batch 16, then A=2 at batch 17.
    # Actual optimizer attempts therefore occur at 0..16 and 18.
    assert actual_steps == [*range(17), 18]
    assert torch.isfinite(trainer.model.weight).all()
    assert trainer.ema.updates == len(actual_steps)


@pytest.mark.parametrize("gradient_kind", ["none", "zero", "inf", "nan"])
def test_pending_gradient_values_survive_save_restore_and_vendor_zero_grad(
    tmp_path, gradient_kind
):
    from car_smart_assist.perception.yolo_training import (
        attach_training_state_callbacks,
        load_training_state,
    )

    trainer, callbacks = _make_trainer(tmp_path)
    _start_vendor_training(trainer, callbacks)
    callbacks["on_train_epoch_start"](trainer)
    trainer.preprocess_batch(_vendor_batch())
    (0.5 * trainer.model.weight.square().sum() / trainer.accumulate).backward()
    callbacks["on_train_batch_end"](trainer)
    if gradient_kind == "none":
        trainer.model.weight.grad = None
    elif gradient_kind == "zero":
        trainer.model.weight.grad.zero_()
    elif gradient_kind == "inf":
        trainer.model.weight.grad.fill_(float("inf"))
    else:
        trainer.model.weight.grad.fill_(float("nan"))
    callbacks["on_model_save"](trainer)

    state_path = trainer.wdir / "last_training_state.pt"
    state = load_training_state(
        state_path, trainer.last, "tiny-linear-dataset-v1", SELECTION_CONTRACT
    )
    saved_gradient = state["accumulation"]["pending_gradients"]["weight"]
    restored, restored_callbacks = _make_trainer(
        tmp_path, initial_weight=99.0, start_epoch=1
    )
    wrapper = SimpleNamespace(
        add_callback=lambda name, callback: restored_callbacks.update({name: callback})
    )
    attach_training_state_callbacks(
        wrapper, "tiny-linear-dataset-v1", state,
        selection_contract=SELECTION_CONTRACT, training_segment=TRAINING_SEGMENT,
    )
    restored_callbacks["on_train_start"](restored)
    assert restored.model.weight.grad is None
    restored.optimizer.zero_grad(set_to_none=True)
    restored_callbacks["on_train_epoch_start"](restored)
    actual = restored.model.weight.grad

    if gradient_kind == "none":
        assert saved_gradient is None and actual is None
    elif gradient_kind == "zero":
        assert torch.equal(saved_gradient, torch.zeros_like(saved_gradient))
        assert torch.equal(actual, saved_gradient)
    elif gradient_kind == "inf":
        assert torch.isinf(saved_gradient).all() and torch.isinf(actual).all()
    else:
        assert torch.isnan(saved_gradient).all() and torch.isnan(actual).all()


def test_restored_amp_overflow_reaches_real_scaler_attempt_without_weight_update(tmp_path):
    from car_smart_assist.perception.yolo_training import (
        attach_training_state_callbacks,
        load_training_state,
    )

    trainer, callbacks = _make_trainer(tmp_path, scaler_enabled=True)
    _start_vendor_training(trainer, callbacks)
    callbacks["on_train_epoch_start"](trainer)
    actual_accumulate = max(round(trainer.args.nbs / trainer.batch_size), 1)
    trainer.accumulate = actual_accumulate
    trainer.preprocess_batch(_vendor_batch())
    trainer.scaler.scale(
        0.5 * trainer.model.weight.square().sum() / actual_accumulate
    ).backward()
    trainer.model.weight.grad.fill_(float("inf"))
    callbacks["on_train_batch_end"](trainer)
    callbacks["on_model_save"](trainer)

    state = load_training_state(
        trainer.wdir / "last_training_state.pt", trainer.last,
        "tiny-linear-dataset-v1", SELECTION_CONTRACT,
    )
    restored, restored_callbacks = _make_trainer(
        tmp_path, initial_weight=99.0, start_epoch=1, scaler_enabled=True
    )
    wrapper = SimpleNamespace(
        add_callback=lambda name, callback: restored_callbacks.update({name: callback})
    )
    attach_training_state_callbacks(
        wrapper, "tiny-linear-dataset-v1", state,
        selection_contract=SELECTION_CONTRACT, training_segment=TRAINING_SEGMENT,
    )
    restored_callbacks["on_train_start"](restored)
    restored.optimizer.zero_grad(set_to_none=True)
    restored_callbacks["on_train_epoch_start"](restored)
    assert torch.isinf(restored.model.weight.grad).all()
    weight_before = restored.model.weight.detach().clone()
    step_events = []
    _record_optimizer_steps(restored, step_events)

    # Resume is a fresh vendor invocation (local last_opt_step=-1), while the
    # real project trainer retains absolute attempted position 0.
    restored.epoch = 1
    actual_accumulate = max(round(restored.args.nbs / restored.batch_size), 1)
    restored.accumulate = actual_accumulate
    restored.preprocess_batch(_vendor_batch())
    restored.scaler.scale(
        0.5 * restored.model.weight.square().sum() / actual_accumulate
    ).backward()
    batch_index = len(restored.train_loader)
    assert batch_index - (-1) >= restored.accumulate
    restored.optimizer_step()

    assert step_events == []  # GradScaler saw Inf and skipped SGD.step.
    torch.testing.assert_close(restored.model.weight, weight_before, rtol=0, atol=0)
    assert restored.optimizer.state == {}
    assert restored.scaler.get_scale() == 4.0
    assert restored._project_last_optimizer_attempt == batch_index
    assert restored._project_vendor_last_optimizer_attempt == batch_index
    assert restored.ema.updates == 1  # Match vendor's post-attempt EMA ordering.


@pytest.mark.parametrize(
    "mutation,reason",
    [
        ("legacy-format", "format"),
        ("missing-accumulation", "format"),
        ("accumulation-version", "format"),
    ],
)
def test_incomplete_or_legacy_training_state_is_rejected(mutation, reason, tmp_path):
    from car_smart_assist.perception.yolo_training import (
        attach_training_state_callbacks,
    )

    trainer, callbacks = _make_trainer(tmp_path)
    _start_vendor_training(trainer, callbacks)
    callbacks["on_train_epoch_start"](trainer)
    trainer.preprocess_batch(_vendor_batch())
    (0.5 * trainer.model.weight.square().sum() / trainer.accumulate).backward()
    callbacks["on_train_batch_end"](trainer)
    callbacks["on_model_save"](trainer)
    state = torch.load(
        trainer.wdir / "last_training_state.pt", map_location="cpu", weights_only=True
    )
    if mutation == "legacy-format":
        state["format_version"] = 4
    elif mutation == "missing-accumulation":
        state.pop("accumulation")
    else:
        state["accumulation"]["format_version"] = 0
    with pytest.raises(ValueError, match=reason):
        attach_training_state_callbacks(
            SimpleNamespace(add_callback=lambda *_: None),
            "tiny-linear-dataset-v1", state,
            selection_contract=SELECTION_CONTRACT, training_segment=TRAINING_SEGMENT,
        )


@pytest.mark.parametrize("layout_change", ["optimizer", "batch", "parameters"])
def test_resume_rejects_changed_optimizer_identity_or_batch_parameter_layout(
    tmp_path, layout_change
):
    from car_smart_assist.perception.yolo_training import (
        attach_training_state_callbacks,
        load_training_state,
    )

    saved, callbacks = _make_trainer(tmp_path)
    _start_vendor_training(saved, callbacks)
    callbacks["on_train_epoch_start"](saved)
    saved.preprocess_batch(_vendor_batch())
    (0.5 * saved.model.weight.square().sum() / saved.accumulate).backward()
    callbacks["on_train_batch_end"](saved)
    callbacks["on_model_save"](saved)
    state = load_training_state(
        saved.wdir / "last_training_state.pt", saved.last,
        "tiny-linear-dataset-v1", SELECTION_CONTRACT,
    )
    restored, restored_callbacks = _make_trainer(
        tmp_path, initial_weight=99.0, start_epoch=1
    )
    if layout_change == "optimizer":
        restored.optimizer = torch.optim.AdamW(restored.model.parameters(), lr=0.1)
    elif layout_change == "batch":
        restored.batch_size = 4
    else:
        restored.model = torch.nn.Linear(1, 1, bias=True)
        restored.optimizer = torch.optim.SGD(
            restored.model.parameters(), lr=0.1, momentum=0.9
        )
    before = {key: value.detach().clone() for key, value in restored.model.state_dict().items()}
    wrapper = SimpleNamespace(
        add_callback=lambda name, callback: restored_callbacks.update({name: callback})
    )
    attach_training_state_callbacks(
        wrapper, "tiny-linear-dataset-v1", state,
        selection_contract=SELECTION_CONTRACT, training_segment=TRAINING_SEGMENT,
    )
    with pytest.raises(ValueError, match="layout changed"):
        restored_callbacks["on_train_start"](restored)
    assert restored.model.state_dict().keys() == before.keys()
    for name, value in before.items():
        assert torch.equal(restored.model.state_dict()[name], value)
