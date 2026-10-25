"""在 Ultralytics 推理检查点之外，另外保留全精度训练状态。"""

from __future__ import annotations

import hashlib
import json
from copy import copy
from pathlib import Path

import numpy as np
import torch

from car_smart_assist.config.detection import yolo_selection_validator_args
from car_smart_assist.utils.stop import StopController


STOPPER_FIELDS = ("best_fitness", "best_epoch", "patience", "possible_stop")
EARLY_STOPPING_POLICY = "preserve_history"


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
        state.get("format_version") != 4
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
            "Complete early-stopper/segment resume history is missing or incompatible; "
            "use a new run directory (--name) to fine-tune trained weights"
        )


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
            "format_version": 4,
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
        }
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
        trainer.model.load_state_dict(resume_state["model"])
        trainer.optimizer.load_state_dict(resume_state["optimizer"])
        trainer.scheduler.load_state_dict(resume_state["scheduler"])
        if trainer.scaler.is_enabled() and resume_state.get("scaler_enabled", bool(resume_state["scaler"])):
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

    def finalize_state(trainer):
        # final_eval已核验旧best，并记录strip_optimizer合法改写后的身份。
        path = trainer.wdir / "last_training_state.pt"
        state = torch.load(path, map_location="cpu", weights_only=True)
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
            state.update(manual_stop=True, termination_reason="manual_stop")
            temporary = path.with_suffix(".tmp")
            torch.save(state, temporary)
            temporary.replace(path)

    def before_epoch(trainer):
        if stop_controller is not None and stop_controller.requested:
            mark_manual_stop(trainer)
            stop_controller.require_running()

    def before_validation(trainer):
        trainer._project_previous_best_fitness = trainer.best_fitness
        if stop_controller is not None and stop_controller.requested:
            trainer.args.val = True

    def finish_epoch(trainer):
        if stop_controller is not None and stop_controller.requested:
            mark_manual_stop(trainer)
            trainer.stop = True

    model.add_callback("on_train_epoch_start", before_epoch)
    model.add_callback("on_train_epoch_end", before_validation)
    model.add_callback("on_fit_epoch_end", finish_epoch)
    model.add_callback("on_model_save", save_state)
    model.add_callback("on_train_start", restore_state)
    model.add_callback("on_train_end", finalize_state)


def detection_trainer_for_resume(resume_state: dict | None, selection_contract: dict):
    """等项目本地的 Ultralytics 设置初始化完成后再延迟导入。"""
    from ultralytics.data import build_yolo_dataset
    from ultralytics.models.yolo.detect import DetectionTrainer
    from ultralytics.utils.checks import check_imgsz
    from ultralytics.utils.torch_utils import de_parallel

    class ProjectDetectionTrainer(DetectionTrainer):
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
