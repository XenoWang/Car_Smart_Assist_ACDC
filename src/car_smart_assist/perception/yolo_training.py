"""在 Ultralytics 推理检查点之外，另外保留全精度训练状态。"""

from __future__ import annotations

import hashlib
import json
import math
from copy import copy
from pathlib import Path

import numpy as np
import torch

from car_smart_assist.config.detection import yolo_selection_validator_args
from car_smart_assist.utils.stop import StopController


STOPPER_FIELDS = ("best_fitness", "best_epoch", "patience", "possible_stop")
EARLY_STOPPING_POLICY = "preserve_history"
TRAINING_STATE_FORMAT = 5
ACCUMULATION_STATE_FORMAT = 1


def _type_identity(value):
    return f"{type(value).__module__}.{type(value).__qualname__}"


def _accumulation_layout(trainer):
    parameters = list(trainer.model.named_parameters())
    names = {id(parameter): name for name, parameter in parameters}
    try:
        optimizer_names = [
            [names[id(parameter)] for parameter in group["params"]]
            for group in trainer.optimizer.param_groups
        ]
    except KeyError as error:
        raise ValueError("Optimizer contains a parameter outside the training model") from error
    nb = len(trainer.train_loader)
    warmup_epochs = float(trainer.args.warmup_epochs)
    return {
        "batches_per_epoch": nb,
        "dataset_size": len(trainer.train_loader.dataset),
        "batch_size": trainer.batch_size,
        "loader_batch_size": trainer.train_loader.batch_size,
        "drop_last": trainer.train_loader.drop_last,
        "sampler_type": _type_identity(trainer.train_loader.sampler),
        "batch_sampler_type": _type_identity(trainer.train_loader.batch_sampler),
        "nbs": trainer.args.nbs,
        "warmup_epochs": warmup_epochs,
        "warmup_iterations": max(round(warmup_epochs * nb), 100) if warmup_epochs > 0 else -1,
        "amp": trainer.amp,
        "model_type": _type_identity(trainer.model),
        "optimizer_type": _type_identity(trainer.optimizer),
        "scaler_type": _type_identity(trainer.scaler),
        "scaler_enabled": trainer.scaler.is_enabled(),
        "parameter_layout": [
            {
                "name": name, "shape": list(parameter.shape), "dtype": str(parameter.dtype),
                "requires_grad": parameter.requires_grad,
            }
            for name, parameter in parameters
        ],
        "optimizer_parameter_names": optimizer_names,
    }


def _require_accumulation_state(state):
    def incompatible(reason):
        raise ValueError(
            f"Complete gradient-accumulation resume state is missing or incompatible ({reason}); "
            "retain the old weights and use a new run directory (--name) to fine-tune them"
        )

    accumulation = state.get("accumulation")
    if (
        state.get("format_version") != TRAINING_STATE_FORMAT
        or not isinstance(accumulation, dict)
        or accumulation.get("format_version") != ACCUMULATION_STATE_FORMAT
    ):
        incompatible("format")
    for name in ("batches_per_epoch", "dataset_size", "batch_size", "loader_batch_size", "nbs", "accumulate"):
        if type(accumulation.get(name)) is not int or accumulation[name] < 1:
            incompatible(name)
    for name in ("drop_last", "amp", "scaler_enabled"):
        if type(accumulation.get(name)) is not bool:
            incompatible(name)
    for name in ("sampler_type", "batch_sampler_type", "model_type", "optimizer_type", "scaler_type"):
        if not isinstance(accumulation.get(name), str) or not accumulation[name]:
            incompatible(name)
    warmup = accumulation.get("warmup_epochs")
    if type(warmup) not in (int, float) or not math.isfinite(warmup) or warmup < 0:
        incompatible("warmup_epochs")
    nb = accumulation["batches_per_epoch"]
    expected_warmup = max(round(warmup * nb), 100) if warmup > 0 else -1
    if type(accumulation.get("warmup_iterations")) is not int or accumulation["warmup_iterations"] != expected_warmup:
        incompatible("warmup_iterations")
    last_batch = accumulation.get("last_processed_batch")
    last_attempt = accumulation.get("last_optimizer_attempt")
    epoch = state.get("epoch")
    if (
        type(epoch) is not int or epoch < 0
        or type(last_batch) is not int or not epoch * nb <= last_batch < (epoch + 1) * nb
        or type(last_attempt) is not int or not -1 <= last_attempt <= last_batch
    ):
        incompatible("batch/optimizer-attempt counters")
    scale = accumulation.get("scaler_scale")
    if type(scale) not in (int, float) or not math.isfinite(scale) or scale <= 0:
        incompatible("scaler scale")
    scaler = state.get("scaler")
    if not isinstance(scaler, dict) or state.get("scaler_enabled") is not accumulation["scaler_enabled"]:
        incompatible("scaler mode")
    if accumulation["scaler_enabled"]:
        if set(scaler) != {"scale", "growth_factor", "backoff_factor", "growth_interval", "_growth_tracker"}:
            incompatible("scaler state")
        for name in ("scale", "growth_factor", "backoff_factor"):
            if type(scaler[name]) not in (int, float) or not math.isfinite(scaler[name]):
                incompatible(f"scaler {name}")
        if (
            scaler["scale"] != scale or scaler["growth_factor"] <= 1
            or not 0 < scaler["backoff_factor"] < 1
            or type(scaler["growth_interval"]) is not int or scaler["growth_interval"] < 1
            or type(scaler["_growth_tracker"]) is not int
            or not 0 <= scaler["_growth_tracker"] < scaler["growth_interval"]
        ):
            incompatible("scaler counters/scale")
    elif scaler or scale != 1.0:
        incompatible("disabled scaler state")
    layout = accumulation.get("parameter_layout")
    gradients = accumulation.get("pending_gradients")
    model_state = state.get("model")
    if not isinstance(layout, list) or not layout or not isinstance(gradients, dict) or not isinstance(model_state, dict):
        incompatible("parameter layout/gradients")
    names = []
    for entry in layout:
        if not isinstance(entry, dict) or not isinstance(entry.get("name"), str) or not entry["name"]:
            incompatible("parameter name")
        name = entry["name"]
        shape = entry.get("shape")
        if (
            name in names or not isinstance(shape, list)
            or any(type(size) is not int or size < 0 for size in shape)
            or type(entry.get("requires_grad")) is not bool
        ):
            incompatible("parameter layout")
        parameter = model_state.get(name)
        if (
            not isinstance(parameter, torch.Tensor) or list(parameter.shape) != shape
            or str(parameter.dtype) != entry.get("dtype")
        ):
            incompatible(f"model parameter {name}")
        names.append(name)
        gradient = gradients.get(name)
        # Pending AMP gradients remain scaled; Inf/NaN must reach the next real GradScaler attempt.
        if gradient is not None and (
            not isinstance(gradient, torch.Tensor) or gradient.device.type != "cpu"
            or gradient.layout != torch.strided or list(gradient.shape) != shape
            or str(gradient.dtype) != entry["dtype"] or gradient.requires_grad
        ):
            incompatible(f"pending gradient {name}")
    if set(gradients) != set(names):
        incompatible("pending gradient names")
    optimizer_names = accumulation.get("optimizer_parameter_names")
    optimizer = state.get("optimizer")
    groups = optimizer.get("param_groups") if isinstance(optimizer, dict) else None
    if not isinstance(optimizer_names, list) or not isinstance(groups, list) or len(optimizer_names) != len(groups):
        incompatible("optimizer parameter groups")
    seen_names, seen_ids = set(), set()
    for group_names, group in zip(optimizer_names, groups):
        parameter_ids = group.get("params") if isinstance(group, dict) else None
        if (
            not isinstance(group_names, list) or not isinstance(parameter_ids, list)
            or len(group_names) != len(parameter_ids)
        ):
            incompatible("optimizer parameter order")
        for name, parameter_id in zip(group_names, parameter_ids):
            if (
                not isinstance(name, str) or name not in names or name in seen_names
                or type(parameter_id) is not int or parameter_id in seen_ids
            ):
                incompatible("optimizer parameter identity")
            seen_names.add(name)
            seen_ids.add(parameter_id)
    optimizer_state = optimizer.get("state")
    if not isinstance(optimizer_state, dict) or not set(optimizer_state).issubset(seen_ids):
        incompatible("optimizer state parameter identity")
    return accumulation


def _restore_accumulation_gate(trainer):
    trainer.accumulate = trainer._project_actual_accumulate


def _capture_accumulation(trainer):
    _restore_accumulation_gate(trainer)
    return {
        **_accumulation_layout(trainer),
        "format_version": ACCUMULATION_STATE_FORMAT,
        "last_processed_batch": trainer._project_last_processed_batch,
        "last_optimizer_attempt": trainer._project_last_optimizer_attempt,
        "accumulate": trainer._project_actual_accumulate,
        "scaler_scale": trainer.scaler.get_scale(),
        "pending_gradients": {
            name: None if parameter.grad is None else parameter.grad.detach().cpu().clone()
            for name, parameter in trainer.model.named_parameters()
        },
    }


def _cuda_device_ids():
    return [str(torch.cuda.get_device_properties(i).uuid) for i in range(torch.cuda.device_count())]


def require_model_selection_contract(record, contract):
    previous = record.get("model_selection_contract")
    if previous is None or json.dumps(previous, sort_keys=True) != json.dumps(contract, sort_keys=True):
        raise ValueError(
            "Existing run model-selection contract is missing or changed; "
            "use a new run directory (--name) to fine-tune from the old best weights"
        )


def require_best_checkpoint(state, checkpoint: Path) -> str:
    expected = state.get("best_checkpoint_sha256")
    if not expected or not checkpoint.is_file():
        raise ValueError(
            "Paired best checkpoint or its identity is missing; full resume/final output is refused. "
            "Use the original paired run or a new run directory (--name) to fine-tune weights"
        )
    actual = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(
            "Best checkpoint does not match the paired training state; full resume/final output "
            "is refused. Restore the paired best or use a new run directory (--name)"
        )
    return actual


def require_training_history(state):
    stopper = state.get("early_stopper")
    segment = state.get("training_segment")
    if (
        state.get("format_version") != TRAINING_STATE_FORMAT
        or not isinstance(stopper, dict)
        or any(name not in stopper for name in STOPPER_FIELDS)
        or not isinstance(segment, dict)
        or segment.get("early_stopping_policy") != EARLY_STOPPING_POLICY
        or type(segment.get("start_epoch")) is not int
        or type(segment.get("target_epochs")) is not int
        or segment["start_epoch"] < 0
        or segment["target_epochs"] <= segment["start_epoch"]
        or not {"resume_extra_epochs", "epochs_override"}.issubset(segment)
    ):
        raise ValueError(
            "Complete format5 early-stopper/segment/accumulation resume history is missing or incompatible; "
            "retain the old weights and use a new run directory (--name) to fine-tune them"
        )
    _require_accumulation_state(state)


def _plain_numbers(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _plain_numbers(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_plain_numbers(item) for item in value)
    return value


def load_training_state(
    path: Path, checkpoint: Path, dataset_signature: str, selection_contract: dict
) -> dict:
    if not path.is_file():
        raise ValueError(
            "Full-precision resume state is missing; use a new run directory (--name) "
            "to fine-tune trained weights"
        )
    state = torch.load(path, map_location="cpu", weights_only=True)
    require_model_selection_contract(state, selection_contract)
    if (
        state["dataset_signature"] != dataset_signature
        or state["checkpoint_sha256"] != hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    ):
        raise ValueError("Resume state does not match dataset/checkpoint")
    require_best_checkpoint(state, checkpoint.parent / "best.pt")
    require_training_history(state)
    return state


def attach_training_state_callbacks(
    model, dataset_signature: str, resume_state: dict | None, best_fitness: float | None = None,
    *, selection_contract: dict, training_segment: dict, stop_controller: StopController | None = None,
) -> None:
    if resume_state is not None:
        require_model_selection_contract(resume_state, selection_contract)
        require_training_history(resume_state)
    elif best_fitness is not None:
        raise ValueError("Historical best fitness requires paired complete training state")

    def save_state(trainer):
        if getattr(trainer, "_project_best_written", trainer.best_fitness == trainer.fitness):
            # Ultralytics刚把本轮checkpoint同时保存为best，可建立新身份。
            best_sha256 = hashlib.sha256(trainer.best.read_bytes()).hexdigest()
        else:
            best_sha256 = require_best_checkpoint(
                {"best_checkpoint_sha256": trainer._project_best_checkpoint_sha256}, trainer.best
            )
        trainer._project_best_checkpoint_sha256 = best_sha256
        state = {
            "format_version": TRAINING_STATE_FORMAT,
            "dataset_signature": dataset_signature,
            "model_selection_contract": selection_contract,
            "checkpoint_sha256": hashlib.sha256(trainer.last.read_bytes()).hexdigest(),
            "best_checkpoint_sha256": best_sha256,
            "training_segment": dict(training_segment),
            "early_stopper": {
                name: _plain_numbers(getattr(trainer.stopper, name)) for name in STOPPER_FIELDS
            },
            "epoch": trainer.epoch,
            "manual_stop": bool(getattr(trainer, "_project_manual_stop", False)),
            "training_complete": trainer.epoch + 1 >= trainer.epochs or bool(
                trainer.stop and not getattr(trainer, "_project_manual_stop", False)
            ),
            "termination_reason": "manual_stop" if getattr(trainer, "_project_manual_stop", False) else (
                "budget_completed" if trainer.epoch + 1 >= trainer.epochs else (
                    "early_stopping" if trainer.stop else None
                )
            ),
            "best_fitness": trainer.best_fitness,
            "model": trainer.model.state_dict(),
            "optimizer": _plain_numbers(trainer.optimizer.state_dict()),
            "scheduler": _plain_numbers(trainer.scheduler.state_dict()),
            "scaler": trainer.scaler.state_dict(),
            "scaler_enabled": trainer.scaler.is_enabled(),
            "ema": trainer.ema.ema.state_dict(),
            "ema_updates": trainer.ema.updates,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if trainer.device.type == "cuda" else [],
            "cuda_device_ids": _cuda_device_ids() if trainer.device.type == "cuda" else [],
            "accumulation": _capture_accumulation(trainer),
        }
        _require_accumulation_state(state)
        path = trainer.wdir / "last_training_state.pt"
        temporary = path.with_suffix(".tmp")
        torch.save(state, temporary)
        temporary.replace(path)
        trainer._project_saved_epoch = trainer.epoch

    def restore_state(trainer):
        trainer._project_stop_controller = stop_controller
        trainer._project_manual_stop = False
        trainer._project_saved_epoch = -1
        if stop_controller is not None:
            stop_controller.watch_directory(trainer.wdir)
            stop_controller.require_running()
            # Every epoch must retain the historical best, even with save=False.
            trainer.args.save = True
        trainer._project_best_checkpoint_sha256 = None
        trainer._project_last_optimizer_attempt = -1
        # This mirrors vendor's per-invocation last_opt_step, which always starts at -1.
        trainer._project_vendor_last_optimizer_attempt = -1
        trainer._project_last_processed_batch = -1
        trainer._project_current_batch = -1
        trainer._project_batch_index = 0
        trainer._project_actual_accumulate = trainer.accumulate
        trainer._project_pending_gradients = None
        if resume_state is None:
            return
        if resume_state["epoch"] != trainer.start_epoch - 1:
            raise ValueError("Resume epoch does not match full-precision state")
        saved_cuda = resume_state["cuda_rng"]
        if saved_cuda and (
            not torch.cuda.is_available()
            or len(saved_cuda) != torch.cuda.device_count()
            or resume_state.get("cuda_device_ids") != _cuda_device_ids()
        ):
            raise ValueError("CUDA device layout changed; restore the original visible GPUs or use --fresh/new run")
        trainer._project_best_checkpoint_sha256 = require_best_checkpoint(resume_state, trainer.best)
        accumulation = _require_accumulation_state(resume_state)
        for name, value in _accumulation_layout(trainer).items():
            if accumulation.get(name) != value:
                raise ValueError(
                    f"Resume gradient-accumulation layout changed ({name}); restore the original "
                    "loader/batch/AMP/model/optimizer setup or use a new run directory (--name)"
                )
        trainer.model.load_state_dict(resume_state["model"])
        trainer.optimizer.load_state_dict(resume_state["optimizer"])
        trainer.scheduler.load_state_dict(resume_state["scheduler"])
        if trainer.scaler.is_enabled():
            trainer.scaler.load_state_dict(resume_state["scaler"])
        trainer.ema.ema.load_state_dict(resume_state["ema"])
        trainer.ema.updates = resume_state["ema_updates"]
        trainer.best_fitness = resume_state["best_fitness"]
        # 中断恢复和新追加段都保留完整等待历史，不重置patience窗口。
        for name in STOPPER_FIELDS:
            setattr(trainer.stopper, name, resume_state["early_stopper"][name])
        torch.set_rng_state(resume_state["torch_rng"])
        if torch.cuda.is_available() and resume_state["cuda_rng"]:
            torch.cuda.set_rng_state_all(resume_state["cuda_rng"])
        trainer._project_last_optimizer_attempt = accumulation["last_optimizer_attempt"]
        trainer._project_last_processed_batch = accumulation["last_processed_batch"]
        trainer._project_actual_accumulate = accumulation["accumulate"]
        _restore_accumulation_gate(trainer)
        # Vendor zero_grad runs after on_train_start; restore these only at the first epoch start.
        trainer._project_pending_gradients = {
            name: None if gradient is None else gradient.detach().clone()
            for name, gradient in accumulation["pending_gradients"].items()
        }

    def finalize_state(trainer):
        # final_eval已核验旧best，并记录strip_optimizer合法改写后的身份。
        path = trainer.wdir / "last_training_state.pt"
        state = torch.load(path, map_location="cpu", weights_only=True)
        require_training_history(state)
        state["best_checkpoint_sha256"] = require_best_checkpoint(
            {"best_checkpoint_sha256": trainer._project_best_checkpoint_sha256}, trainer.best
        )
        state["checkpoint_sha256"] = hashlib.sha256(trainer.last.read_bytes()).hexdigest()
        temporary = path.with_suffix(".tmp")
        torch.save(state, temporary)
        temporary.replace(path)

    def mark_manual_stop(trainer):
        trainer._project_manual_stop = True
        path = trainer.wdir / "last_training_state.pt"
        if path.is_file():
            state = torch.load(path, map_location="cpu", weights_only=True)
            require_training_history(state)
            state.update(manual_stop=True, termination_reason="manual_stop")
            temporary = path.with_suffix(".tmp")
            torch.save(state, temporary)
            temporary.replace(path)

    def before_epoch(trainer):
        if stop_controller is not None and stop_controller.requested:
            mark_manual_stop(trainer)
            stop_controller.require_running()
        trainer._project_batch_index = 0
        trainer._project_current_batch = -1
        _restore_accumulation_gate(trainer)
        pending = trainer._project_pending_gradients
        if pending is not None:
            for name, parameter in trainer.model.named_parameters():
                gradient = pending[name]
                parameter.grad = None if gradient is None else gradient.to(
                    device=parameter.device, dtype=parameter.dtype
                ).clone()
            trainer._project_pending_gradients = None

    def finish_batch(trainer):
        # Without warmup the next batch reuses self.accumulate, so never leave the temporary gate.
        _restore_accumulation_gate(trainer)

    def before_validation(trainer):
        # Also covers vendor's time-budget break, which can skip on_train_batch_end.
        _restore_accumulation_gate(trainer)
        trainer._project_previous_best_fitness = trainer.best_fitness
        if stop_controller is not None and stop_controller.requested:
            trainer.args.val = True

    def finish_epoch(trainer):
        if stop_controller is not None and stop_controller.requested:
            mark_manual_stop(trainer)
            trainer.stop = True

    model.add_callback("on_train_epoch_start", before_epoch)
    model.add_callback("on_train_batch_end", finish_batch)
    model.add_callback("on_train_epoch_end", before_validation)
    model.add_callback("on_fit_epoch_end", finish_epoch)
    model.add_callback("on_model_save", save_state)
    model.add_callback("on_train_start", restore_state)
    model.add_callback("on_train_end", finalize_state)


def detection_trainer_for_resume(resume_state: dict | None, selection_contract: dict, *, validated_data=None):
    """等项目本地的 Ultralytics 设置初始化完成后再延迟导入。"""
    from ultralytics.data import build_yolo_dataset
    from ultralytics.models.yolo.detect import DetectionTrainer
    from ultralytics.utils.checks import check_imgsz
    from ultralytics.utils.torch_utils import de_parallel

    class ProjectDetectionTrainer(DetectionTrainer):
        def preprocess_batch(self, batch):
            batch = super().preprocess_batch(batch)
            # Warmup has already selected the real A. Adjust only vendor's comparison gate:
            # ni - V >= A + L - V is exactly ni - L >= A, including across invocations.
            self._project_actual_accumulate = self.accumulate
            self._project_current_batch = self.epoch * len(self.train_loader) + self._project_batch_index
            self._project_batch_index += 1
            self._project_last_processed_batch = self._project_current_batch
            self.accumulate = (
                self._project_actual_accumulate + self._project_last_optimizer_attempt
                - self._project_vendor_last_optimizer_attempt
            )
            return batch

        def optimizer_step(self):
            _restore_accumulation_gate(self)
            # The real vendor call preserves scaled-overflow, zero_grad and EMA behavior.
            super().optimizer_step()
            self._project_last_optimizer_attempt = self._project_current_batch
            self._project_vendor_last_optimizer_attempt = self._project_current_batch

        def save_model(self):
            # Keep each checkpoint replacement atomic, without changing vendor state.
            last, best = self.last, self.best
            self.last = last.with_name(last.name + ".tmp")
            self.best = best.with_name(best.name + ".tmp")
            previous = getattr(self, "_project_previous_best_fitness", None)
            improved = not best.exists() or previous is None or self.fitness > previous
            try:
                super().save_model()
                self.last.replace(last)
                self._project_best_written = self.best.exists() and improved
                if self._project_best_written:
                    self.best.replace(best)
            finally:
                temporary_last, temporary_best = self.last, self.best
                self.last, self.best = last, best
                temporary_last.unlink(missing_ok=True)
                temporary_best.unlink(missing_ok=True)

        def validation_args(self):
            args = copy(self.args)
            for name, value in yolo_selection_validator_args(selection_contract).items():
                setattr(args, name, value)
            args.imgsz = check_imgsz(args.imgsz, max_dim=1)
            return args

        def build_dataset(self, img_path, mode="train", batch=None):
            if mode != "val":
                return super().build_dataset(img_path, mode, batch)
            stride = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
            return build_yolo_dataset(
                self.validation_args(), img_path, batch, self.data, mode=mode, rect=True,
                stride=stride,
            )

        def get_validator(self):
            # Joint backend在模块级引用本factory，延迟到此处导入避免循环。
            from car_smart_assist.perception.joint_yolo_backend import DeploymentValidator

            self.loss_names = "box_loss", "cls_loss", "dfl_loss"
            return DeploymentValidator(
                self.test_loader, save_dir=self.save_dir, args=self.validation_args(),
                _callbacks=self.callbacks,
            )

        def final_eval(self):
            state = torch.load(
                self.wdir / "last_training_state.pt", map_location="cpu", weights_only=True
            )
            require_training_history(state)
            if state["checkpoint_sha256"] != hashlib.sha256(self.last.read_bytes()).hexdigest():
                raise ValueError("Final last checkpoint does not match the paired training state")
            require_best_checkpoint(state, self.best)
            controller = getattr(self, "_project_stop_controller", None)
            if controller is not None and controller.requested:
                self._project_manual_stop = True
                state.update(manual_stop=True, termination_reason="manual_stop")
                temporary = (self.wdir / "last_training_state.pt").with_suffix(".tmp")
                torch.save(state, temporary)
                temporary.replace(self.wdir / "last_training_state.pt")
                return
            # 8.3.162会合法重写last/best，先核验再交给原实现收尾。
            super().final_eval()
            self._project_best_checkpoint_sha256 = hashlib.sha256(self.best.read_bytes()).hexdigest()

        def check_resume(self, overrides):
            super().check_resume(overrides)
            if self.resume:
                self.args.epochs = overrides["epochs"]
                if validated_data is not None:
                    # Vendor resume reloads checkpoint args, including the obsolete
                    # physical data path. Use only the data accepted for this run.
                    self.args.data = str(validated_data)

        def resume_training(self, ckpt):
            if self.resume and ckpt is not None and ckpt.get("epoch", -1) < 0:
                if resume_state is None:
                    raise ValueError("Completed checkpoint needs paired FP32 state to resume")
                self.start_epoch = int(resume_state["epoch"]) + 1
                self.best_fitness = float(resume_state["best_fitness"])
                if self.start_epoch > self.epochs - self.args.close_mosaic:
                    self._close_dataloader_mosaic()
            else:
                super().resume_training(ckpt)

    return ProjectDetectionTrainer
