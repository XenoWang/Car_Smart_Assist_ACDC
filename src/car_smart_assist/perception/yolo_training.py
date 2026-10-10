"""在 Ultralytics 推理检查点之外，另外保留全精度训练状态。"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch


def _plain_numbers(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _plain_numbers(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_plain_numbers(item) for item in value)
    return value


def load_training_state(path: Path, checkpoint: Path, dataset_signature: str) -> dict:
    if not path.is_file():
        raise ValueError(
            "Full-precision resume state is missing; start a new run from trained weights"
        )
    state = torch.load(path, map_location="cpu", weights_only=True)
    if (
        state["dataset_signature"] != dataset_signature
        or state["checkpoint_sha256"] != hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    ):
        raise ValueError("Resume state does not match dataset/checkpoint")
    return state


def attach_training_state_callbacks(
    model, dataset_signature: str, resume_state: dict | None, best_fitness: float | None = None
) -> None:
    def save_state(trainer):
        state = {
            "format_version": 2,
            "dataset_signature": dataset_signature,
            "checkpoint_sha256": hashlib.sha256(trainer.last.read_bytes()).hexdigest(),
            "epoch": trainer.epoch,
            "best_fitness": trainer.best_fitness,
            "model": trainer.model.state_dict(),
            "optimizer": _plain_numbers(trainer.optimizer.state_dict()),
            "scheduler": _plain_numbers(trainer.scheduler.state_dict()),
            "scaler": trainer.scaler.state_dict(),
            "ema": trainer.ema.ema.state_dict(),
            "ema_updates": trainer.ema.updates,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        }
        path = trainer.wdir / "last_training_state.pt"
        temporary = path.with_suffix(".tmp")
        torch.save(state, temporary)
        temporary.replace(path)

    def restore_state(trainer):
        if resume_state is None:
            if best_fitness is not None:
                trainer.best_fitness = best_fitness
            return
        if resume_state["epoch"] != trainer.start_epoch - 1:
            raise ValueError("Resume epoch does not match full-precision state")
        trainer.model.load_state_dict(resume_state["model"])
        trainer.optimizer.load_state_dict(resume_state["optimizer"])
        trainer.scheduler.load_state_dict(resume_state["scheduler"])
        trainer.scaler.load_state_dict(resume_state["scaler"])
        trainer.ema.ema.load_state_dict(resume_state["ema"])
        trainer.ema.updates = resume_state["ema_updates"]
        trainer.best_fitness = resume_state.get("best_fitness", trainer.best_fitness)
        torch.set_rng_state(resume_state["torch_rng"])
        if torch.cuda.is_available() and resume_state["cuda_rng"]:
            torch.cuda.set_rng_state_all(resume_state["cuda_rng"])

    def finalize_state(trainer):
        # Ultralytics 在训练完成时会从 last.pt 里删掉 optimizer/epoch。把
        # 独立的 FP32 状态绑到最终文件上，这样跑完的训练也能续训。
        path = trainer.wdir / "last_training_state.pt"
        if not path.is_file():
            return
        state = torch.load(path, map_location="cpu", weights_only=True)
        state["checkpoint_sha256"] = hashlib.sha256(trainer.last.read_bytes()).hexdigest()
        temporary = path.with_suffix(".tmp")
        torch.save(state, temporary)
        temporary.replace(path)

    model.add_callback("on_model_save", save_state)
    model.add_callback("on_train_start", restore_state)
    model.add_callback("on_train_end", finalize_state)


def detection_trainer_for_resume(resume_state: dict | None):
    """等项目本地的 Ultralytics 设置初始化完成后再延迟导入。"""
    from ultralytics.models.yolo.detect import DetectionTrainer

    class ProjectDetectionTrainer(DetectionTrainer):
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
