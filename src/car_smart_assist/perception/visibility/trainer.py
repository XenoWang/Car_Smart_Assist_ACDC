"能见度门控的无监督训练（含检查点保存与续训）。"

from __future__ import annotations

import copy
import hashlib
import json
import logging
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from car_smart_assist.config.visibility import MODEL_DEFAULTS, training_config
from car_smart_assist.perception.visibility.autoencoder import (
    build_autoencoder,
    reconstruction_mean,
)
from car_smart_assist.perception.visibility.dataset import VisibilityImageDataset
from car_smart_assist.perception.visibility.scorer import CalibrationStats

logger = logging.getLogger(__name__)

ResumeSpec = Literal["auto", "none"] | str | Path | None


@dataclass
class TrainHistory:
    """训练过程记录，用于报告与排查。"""

    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    best_epoch: int = -1
    best_val_loss: float = float("inf")
    stopped_early: bool = False
    seconds: float = 0.0
    resumed_from: str | None = None
    start_epoch: int = 0
    planned_epochs: int = 0
    termination_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "final_train_loss": self.train_loss[-1] if self.train_loss else None,
            "best_val_loss": self.best_val_loss,
            "best_epoch": self.best_epoch,
            "stopped_early": self.stopped_early,
            "epochs_run": len(self.train_loss),
            "start_epoch": self.start_epoch,
            "planned_epochs": self.planned_epochs,
            "termination_reason": self.termination_reason,
            "resumed_from": self.resumed_from,
            "seconds": round(self.seconds, 1),
        }


def _model_state_sha256(state: dict[str, torch.Tensor]) -> str:
    """只绑定权重，校准与历史记录的更新不会改变其身份。"""
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        header = json.dumps([name, str(tensor.dtype), list(tensor.shape)],
                            ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _seed_worker(_worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def resolve_device(spec: str = "auto") -> torch.device:
    """auto 优先用当前 PyTorch 可用的 CUDA GPU；显式指定 CUDA 但不可用时直接报错。"""
    spec = str(spec).strip().lower()
    automatic = spec == "auto"
    if automatic:
        if not torch.cuda.is_available():
            logger.warning(
                "CUDA 不可用，自动使用 CPU（PyTorch=%s，CUDA runtime=%s）。"
                "GPU 训练需要当前 .venv 中的 CUDA 版 PyTorch 和可用的 NVIDIA 驱动。",
                torch.__version__, torch.version.cuda,
            )
            return torch.device("cpu")
        spec = "cuda"
    device = torch.device(spec)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("指定了 CUDA，但当前 .venv 的 PyTorch 无可用 CUDA GPU；请检查驱动和 PyTorch，或使用 device=auto/cpu")
        index = device.index if device.index is not None else torch.cuda.current_device()
        count = torch.cuda.device_count()
        if index >= count:
            raise ValueError(f"指定了 cuda:{index}，但当前仅检测到 {count} 个 CUDA GPU")
        logger.info(
            "训练/推理设备: cuda:%d — %s（PyTorch CUDA %s）",
            index, torch.cuda.get_device_name(index), torch.version.cuda,
        )
    else:
        logger.info("训练/推理设备: %s（显式配置）", device)
    return device


def _atomic_save(payload: dict[str, Any], path: Path) -> None:
    """原子写检查点：先写临时文件再 rename。

    如果直接覆盖原文件时进程被中断（Ctrl-C、断电、OOM），会留下半个文件，
    下次续训读它直接崩，原始权重也一起没了，训练几小时的成果就丢了。
    同目录下 rename 在 Windows 与 POSIX 上都是原子的。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def apply_config_overrides(cfg: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """按点号路径把覆盖项写入配置副本。

    例：``{"model.kernel_sizes": [3, 5]}`` 会写进 ``cfg["model"]["kernel_sizes"]``。

    放在这里而不是消融脚本里，是为了让测试能用同一份实现校验配置。
    消融预设写错（比如改了 kernel_sizes 却没改 dilations）若不提前发现，
    要等跑到那一组、白等七分钟才崩。
    """
    out = copy.deepcopy(cfg)
    for dotted, value in overrides.items():
        parts = dotted.split(".")
        node = out
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = value
    return out


def resolve_resume_spec(resume: ResumeSpec, ckpt_dir: Path) -> Path | None:
    """决定从哪个检查点续训。

    Args:
        resume: 'auto' / None  -> 优先 last.pt，其次 best.pt；都不存在则返回 None
                'none' / False -> 强制全新训练，返回 None
                其他            -> 视为显式路径
    """
    if resume is None or (isinstance(resume, str) and resume.lower() == "auto"):
        for name in ("last.pt", "best.pt"):
            p = ckpt_dir / name
            if p.exists():
                return p
        return None
    if resume is False or (isinstance(resume, str) and resume.lower() == "none"):
        return None
    p = Path(resume)
    if not p.exists():
        raise FileNotFoundError(f"指定的检查点不存在: {p}")
    return p


class VisibilityTrainer:
    """自编码器的无监督训练器（支持续训）。

    Args:
        cfg: configs/model/visibility.yaml 的 ``visibility`` 段
        project_root: 项目根目录，配置中的相对路径以此为基准
        resume: 续训来源。'auto'（默认）自动复用已有检查点；
            'none' 强制全新训练；也可传具体路径。
            CLI 对应 --resume / --fresh。
    """

    def __init__(
        self,
        cfg: dict[str, Any],
        project_root: str | Path = ".",
        resume: ResumeSpec = "auto",
    ) -> None:
        if cfg.get("data", {}).get("split_mode", "fixed") == "random":
            raise ValueError("训练／校准不能重建随机分区；请设置 data.split_mode=fixed 以供恢复和评估复用")
        self.cfg = cfg
        self.root = Path(project_root)
        self.device = resolve_device(str(cfg.get("device", "auto")))
        self.model_cfg = dict(cfg.get("model", {}))
        self.scoring_cfg = copy.deepcopy(cfg.get("scoring", {}))
        tcfg = cfg.get("train", {})
        self.seed = int(tcfg.get("seed", 42))
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        self.loader_generator = torch.Generator().manual_seed(self.seed)
        self.ckpt_dir = self.root / tcfg.get(
            "checkpoint_dir", "artifacts/checkpoints/visibility"
        )
        self.resume_path = resolve_resume_spec(resume, self.ckpt_dir)
        self.history = TrainHistory()
        self.calibration: CalibrationStats | None = None
        self.scaler: torch.amp.GradScaler | None = None
        self._pending_scaler_state: dict | None = None
        self._resume_train_cfg: dict[str, Any] | None = None
        self._resume_training_complete = False
        self.split_signature: str | None = None
        self._since_best = 0
        self._best_model_sha256: str | None = None
        # 由 _build_datasets 填入各数据分区的样本数
        self.split_stats: dict[str, int] = {}

        # 续训时模型结构必须与检查点一致，否则加载权重会报形状不匹配。
        # 先把检查点里的 model_cfg 读出来，以它为准构建模型。
        ckpt_model_cfg: dict[str, Any] | None = None
        if self.resume_path is not None:
            head = torch.load(self.resume_path, map_location="cpu", weights_only=False)
            if int(head.get("format_version", 1)) < 2:
                raise ValueError(
                    f"检查点 {self.resume_path} 使用旧的 train/calib/test 划分策略，"
                    "其 best 权重曾由校准集选出，不能用于独立校准流程。"
                    "请使用 --fresh 从头训练；旧 best.pt 仍可用于当前推理。"
                )
            if int(head.get("format_version", 1)) < 3 or not head.get("best_model_sha256"):
                raise ValueError(
                    "旧AE检查点没有可核验的历史best身份，不能完整续训。"
                    "请保留原件；可用 --fresh 新开实验，旧权重仍可推理。"
                    "如需沿旧权重微调，请另行指定微调方案。"
                )
            ckpt_model_cfg = head.get("model_cfg")
            if ckpt_model_cfg and ckpt_model_cfg != self.model_cfg:
                logger.warning(
                    "配置中的模型结构与检查点不一致，**以检查点为准**以保证权重可加载。\n"
                    "  检查点: %s\n  当前配置: %s\n"
                    "若要按新结构训练，请加 --fresh 或换一个 checkpoint_dir",
                    ckpt_model_cfg, self.model_cfg,
                )
            del head

        # 续训时把生效的结构写回 self.model_cfg。
        # 不能只在建模型时用一下检查点的结构就丢掉：_make_payload 写的是
        # self.model_cfg，如果这里仍保留配置里的旧值，下一次保存就会把与权重不匹配的
        # 结构写进检查点，后续再续训就彻底错乱了。
        # 同时 _build_datasets 读的是 self.model_cfg 里的 input_size，
        # 模型按检查点的尺寸训练，数据也必须按同一尺寸准备。
        if ckpt_model_cfg:
            self.model_cfg = dict(ckpt_model_cfg)

        self.model = build_autoencoder(self.model_cfg).to(self.device)

    # --- 数据 ---

    def _build_datasets(
        self,
    ) -> tuple[
        VisibilityImageDataset,
        VisibilityImageDataset,
        VisibilityImageDataset,
        list[Path],
    ]:
        """构建 train / validation / calibration 数据集。

        test 子集不在这里构建，它由评估脚本单独划分使用，
        训练过程不接触它，保证它只用于报告这一语义。
        """
        from car_smart_assist.perception.visibility.dataset import (
            ensure_cache,
            list_ref_images,
            sequence_of,
            split_ref_indices,
        )

        dcfg = self.cfg.get("data", {})
        acdc_root = self.root / dcfg.get("acdc_root", "data/raw/acdc")
        paths = list_ref_images(
            acdc_root,
            cleaning_manifest=dcfg.get("cleaning_manifest"),
            project_root=self.root / dcfg.get("project_root", "."),
        )
        if not paths:
            raise FileNotFoundError(
                f"未在 {acdc_root} 下找到正常天气参考图（*_rgb_ref_anon.png）。"
                "请先确认 ACDC 已解压，见 docs/dataset.md 5.1"
            )

        size = tuple(self.model_cfg.get("input_size", MODEL_DEFAULTS["input_size"]))
        cache_name = dcfg.get("cache_path")
        cache_path = (self.root / cache_name) if cache_name else None

        # 缓存按全量参考图构建一次（ensure_cache 只返回路径），
        # 各子集通过索引共享，在各自 worker 进程里惰性打开 memmap。
        cache_path = ensure_cache(paths, size, cache_path)

        # ACDC 参考图来自视频序列，相邻帧几乎重复。按图随机划分会把一帧放进
        # 训练集、把它的邻居放进校准集，校准集里混进了训练样本的复制品，
        # 测出来的泛化能力虚高。按序列整组划分才能反映真实的跨场景泛化。
        groups = [sequence_of(p) for p in paths] if dcfg.get("split_by_sequence", True) else None
        train_idx, val_idx, calib_idx, test_idx = split_ref_indices(
            len(paths),
            train_ratio=float(dcfg.get("train_ratio", 0.80)),
            val_ratio=float(dcfg.get("val_ratio", 0.05)),
            calib_ratio=float(dcfg.get("calib_ratio", 0.05)),
            test_ratio=float(dcfg.get("test_ratio", 0.10)),
            seed=int(dcfg.get("split_seed", 42)),
            mode=str(dcfg.get("split_mode", "fixed")),
            groups=groups,
        )

        split_payload = {
            "paths": [
                (
                    p.relative_to(acdc_root).as_posix(),
                    p.stat().st_size,
                    p.stat().st_mtime_ns,
                )
                for p in paths
            ],
            "input_size": size,
            "grouping": "recording_family-v1" if groups is not None else "image",
            "split_algorithm": (
                "recording-family-greedy-nonempty-sse-v1"
                if groups is not None else "image-permutation-v1"
            ),
            "groups": groups,
            "train": train_idx.tolist(),
            "val": val_idx.tolist(),
            "calib": calib_idx.tolist(),
            "test": test_idx.tolist(),
        }
        self.split_signature = hashlib.sha256(
            json.dumps(split_payload, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

        # 只有训练子集做增强。验证和校准都用原图，保持评价口径一致。
        tcfg = training_config(self.cfg.get("train", {}))
        train_ds = VisibilityImageDataset(
            paths, size, cache_path=cache_path, indices=train_idx,
            augment=bool(tcfg["augment"]), augmentation_cfg=tcfg.get("augmentation"),
        )
        val_ds = VisibilityImageDataset(
            paths, size, cache_path=cache_path, indices=val_idx, augment=False
        )
        calib_ds = VisibilityImageDataset(
            paths, size, cache_path=cache_path, indices=calib_idx, augment=False
        )
        stats = {
            "train": len(train_idx),
            "val": len(val_idx),
            "calib": len(calib_idx),
            "test": len(test_idx),
        }
        logger.info(
            "参考图 %d 张 -> 训练 %d / 验证 %d / 校准 %d / 测试 %d "
            "（划分模式 %s，缓存 %s）\n"
            "  验证集只用于选择权重，校准集只用于最终校准，测试集只用于最终报告",
            len(paths), stats["train"], stats["val"], stats["calib"], stats["test"],
            dcfg.get("split_mode", "fixed"),
            "已启用" if cache_path is not None else "未启用（实时解码）",
        )
        self.split_stats = stats
        return train_ds, val_ds, calib_ds, paths

    # --- 检查点 ---

    def _make_payload(
        self,
        epoch: int,
        opt: torch.optim.Optimizer | None,
        sched: Any,
        val_loss: float | None,
    ) -> dict[str, Any]:
        return {
            "format_version": 3,
            "best_model_sha256": self._best_model_sha256,
            "model_state": self.model.state_dict(),
            "model_cfg": self.model_cfg,
            "scoring_cfg": self.scoring_cfg,
            "optimizer_state": opt.state_dict() if opt is not None else None,
            "scheduler_state": sched.state_dict() if sched is not None else None,
            "scaler_state": self.scaler.state_dict() if self.scaler is not None else None,
            "epoch": epoch,
            "history": self.history.to_dict(),
            "train_loss": self.history.train_loss,
            "val_loss": self.history.val_loss,
            "best_val_loss": self.history.best_val_loss,
            "current_val_loss": val_loss,
            "best_epoch": self.history.best_epoch,
            "calibration": self.calibration.to_dict() if self.calibration else None,
            "config_snapshot": {
                "data": self.cfg.get("data", {}),
                "model": self.model_cfg,
                "train": training_config(self.cfg.get("train", {})),
            },
            "split_signature": self.split_signature,
            "seed": self.seed,
            "training_complete": self.history.termination_reason is not None,
            "termination_reason": self.history.termination_reason,
            "planned_epochs": self.history.planned_epochs,
            "early_stopping_state": {
                "wait": self._since_best,
                "stopped": self.history.stopped_early,
            },
            "rng_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if self.device.type == "cuda" else [],
                "loader": self.loader_generator.get_state(),
            },
        }

    def _load_verified_best(self, reference: dict[str, Any]) -> dict[str, Any]:
        expected = reference.get("best_model_sha256")
        if (not isinstance(expected, str) or len(expected) != 64
                or any(char not in "0123456789abcdef" for char in expected)):
            raise ValueError("AE历史best身份缺失或无效，不能完整续训；请保留原件并另行指定微调方案或 --fresh")
        path = self.ckpt_dir / "best.pt"
        if not path.is_file():
            raise ValueError(
                f"历史best缺失：{path}。请使用保有匹配best的原训练目录；"
                "不会把last或恢复后的第一轮当成历史best。"
            )
        best = torch.load(path, map_location="cpu", weights_only=False)
        for key in ("split_signature", "seed", "model_cfg", "best_epoch", "best_val_loss"):
            if key not in reference or key not in best or best[key] != reference[key]:
                raise ValueError(f"历史best与恢复记录的{key}不匹配：{path}，拒绝完整续训")
        if (best.get("best_model_sha256") != expected
                or best.get("epoch") != reference["best_epoch"]
                or best.get("current_val_loss") != reference["best_val_loss"]
                or _model_state_sha256(best["model_state"]) != expected):
            raise ValueError(f"历史best权重身份或指标不匹配：{path}，拒绝完整续训")
        return best

    def _restore(self, opt: torch.optim.Optimizer, sched: Any) -> int:
        """从检查点恢复全部状态，返回已完成的轮次。"""
        assert self.resume_path is not None
        # 用模型和优化器的 load_state_dict 迁移，避免完整检查点先占显存。
        ckpt = torch.load(self.resume_path, map_location="cpu", weights_only=False)

        if ckpt.get("split_signature") != self.split_signature:
            raise ValueError(
                "检查点与当前数据/切分不匹配，不能比较历史验证损失或继续早停。"
                "请确认数据和 split 配置未变，或使用 --fresh 重新训练。"
            )

        self._load_verified_best(ckpt)
        self._best_model_sha256 = ckpt["best_model_sha256"]
        self.model.load_state_dict(ckpt["model_state"])
        self._pending_scaler_state = ckpt.get("scaler_state")
        if ckpt.get("optimizer_state") is not None:
            opt.load_state_dict(ckpt["optimizer_state"])
        else:
            logger.warning("检查点中没有优化器状态，动量会从零重建，续训初期损失可能短暂反弹")
        if ckpt.get("scheduler_state") is not None and sched is not None:
            sched.load_state_dict(ckpt["scheduler_state"])

        epoch = int(ckpt.get("epoch", 0))
        train_loss = list(ckpt.get("train_loss", []))
        val_loss = list(ckpt.get("val_loss", []))
        if epoch < 0 or len(train_loss) < epoch or len(val_loss) < epoch:
            raise ValueError("检查点的恢复轮次与损失历史不完整，不能完整续训；请保留原件并用 --fresh 新开实验")
        trimmed_history = len(train_loss) > epoch or len(val_loss) > epoch
        if trimmed_history:
            logger.warning(
                "检查点包含恢复轮次 %d 之后的汇总历史，续训仅保留前 %d 轮；"
                "旧汇总耗时不能归属恢复点，将从0重新累计", epoch, epoch,
            )
        self.history.train_loss = train_loss[:epoch]
        self.history.val_loss = val_loss[:epoch]
        self.history.best_val_loss = float(ckpt.get("best_val_loss", float("inf")))
        self.history.best_epoch = int(ckpt.get("best_epoch", -1))
        self.history.resumed_from = str(self.resume_path)
        self.history.start_epoch = epoch
        self.history.planned_epochs = int(ckpt.get("planned_epochs", 0))
        self._resume_train_cfg = ckpt.get("config_snapshot", {}).get("train")
        self._resume_training_complete = (
            not trimmed_history and bool(ckpt.get("training_complete", False))
        )
        self.history.seconds = (
            0.0 if trimmed_history else float(ckpt.get("history", {}).get("seconds", 0.0))
        )
        self._since_best = int(ckpt.get("early_stopping_state", {}).get("wait", 0))
        rng_state = ckpt.get("rng_state")
        if rng_state is not None:
            if ckpt.get("seed") != self.seed:
                raise ValueError("检查点随机种子与当前配置不一致，请用 --fresh 新开实验")
            random.setstate(rng_state["python"])
            np.random.set_state(rng_state["numpy"])
            torch.set_rng_state(rng_state["torch"])
            if self.device.type == "cuda" and rng_state["cuda"]:
                torch.cuda.set_rng_state_all(rng_state["cuda"])
            self.loader_generator.set_state(rng_state["loader"])
        else:
            logger.warning("旧检查点未保存完整随机/采样状态，无法保证中断恢复的随机轨迹一致")

        cal = ckpt.get("calibration")
        if cal:
            self.calibration = CalibrationStats(**cal)

        logger.info(
            "已载入检查点 %s：完成 %d 轮，历史最优验证损失 %.6f@%d",
            self.resume_path, epoch, self.history.best_val_loss, self.history.best_epoch,
        )
        return epoch

    # --- 训练 ---

    def fit(self) -> TrainHistory:
        """执行无监督训练（必要时先恢复已有检查点）。"""
        tcfg = training_config(self.cfg.get("train", {}))
        dcfg = self.cfg.get("data", {})
        # test 子集不在训练里构建 —— 只有评估脚本会用它
        train_ds, val_ds, calib_ds, _ = self._build_datasets()

        nw = int(dcfg.get("num_workers", 4))
        bs = int(tcfg["batch_size"])
        train_loader = DataLoader(
            train_ds, batch_size=bs, shuffle=True, num_workers=nw,
            pin_memory=self.device.type == "cuda", drop_last=False,
            generator=self.loader_generator, worker_init_fn=_seed_worker,
        )
        val_loader = DataLoader(
            val_ds, batch_size=bs, shuffle=False, num_workers=nw,
            pin_memory=self.device.type == "cuda",
            generator=torch.Generator().manual_seed(self.seed + 1), worker_init_fn=_seed_worker,
        )
        calib_loader = DataLoader(
            calib_ds, batch_size=bs, shuffle=False, num_workers=nw,
            pin_memory=self.device.type == "cuda",
            generator=torch.Generator().manual_seed(self.seed + 2), worker_init_fn=_seed_worker,
        )

        if str(tcfg["optimizer"]).lower() != "adamw":
            raise ValueError("当前能见度训练仅支持 optimizer=adamw")
        opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(tcfg["lr"]),
            weight_decay=float(tcfg["weight_decay"]),
        )

        epochs = int(tcfg["epochs"])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))

        # --- 恢复 ---
        start_epoch = 0
        if self.resume_path is not None:
            start_epoch = self._restore(opt, sched)
            same_budget = self._resume_train_cfg is not None and all(
                self._resume_train_cfg.get(key) == tcfg.get(key)
                for key in ("epochs", "resume_extra_epochs")
            )
            if (
                same_budget
                and not self._resume_training_complete
                and self.history.planned_epochs > start_epoch
            ):
                # 追加段中断后继续原终点，不能每次恢复都再加一份 extra。
                epochs = self.history.planned_epochs
            elif start_epoch >= epochs:
                # 续训时 epochs 表示「再训多少轮」，不是「总共多少轮」，
                # 否则从第 38 轮续训、epochs=40 就只剩 2 轮，等于什么都没做。
                extra = tcfg.get("resume_extra_epochs")
                extra = int(extra) if extra else epochs
                logger.info(
                    "已完成 %d 轮 >= 配置的 %d 轮；把 epochs 解释为「再训 %d 轮」",
                    start_epoch, epochs, extra,
                )
                epochs = start_epoch + extra
            elif tcfg.get("resume_extra_epochs"):
                epochs = start_epoch + int(tcfg["resume_extra_epochs"])

            requested_remaining = epochs - start_epoch
            if requested_remaining > 0:
                restored_lrs = [float(group["lr"]) for group in opt.param_groups]
                if any(not np.isfinite(lr) or lr <= 0 for lr in restored_lrs):
                    raise ValueError(
                        "恢复的学习率已为0或无效，不能继续追加训练。"
                        "请另行指定微调方案；不会自动恢复初始学习率。"
                    )
                if requested_remaining != sched.T_max - sched.last_epoch:
                    # scheduler.step 在每轮保存前执行；last_epoch 是当前段已走步数。
                    for group, lr in zip(opt.param_groups, restored_lrs, strict=True):
                        group["initial_lr"] = lr
                    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                        opt, T_max=requested_remaining, eta_min=0.0
                    )
                    logger.info(
                        "续训预算调整：第 %d -> %d 轮，从恢复学习率 %s 连续余弦衰减至0",
                        start_epoch, epochs, restored_lrs,
                    )
        self.history.planned_epochs = epochs
        self.history.termination_reason = None
        self.history.stopped_early = False

        use_amp = bool(tcfg["amp"]) and self.device.type == "cuda"
        dtype = str(tcfg["dtype"])
        if dtype not in ("bfloat16", "float16", "float32"):
            raise ValueError("train.dtype 可选 bfloat16 / float16 / float32")
        amp_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16
        if dtype == "float32":
            use_amp = False
        if use_amp and dtype == "bfloat16":
            with torch.cuda.device(self.device):
                if not torch.cuda.is_bf16_supported():
                    logger.warning("当前 GPU 不支持 bfloat16，继续使用 GPU float32 训练")
                    use_amp = False
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp and dtype == "float16")
        if scaler.is_enabled() and self._pending_scaler_state:
            scaler.load_state_dict(self._pending_scaler_state)
        self.scaler = scaler
        clip = float(tcfg["grad_clip_norm"])
        # 去噪自编码器：只给**输入**加噪，重建目标还是无噪的原图。
        # 这是重建类任务里最有效的正则化手段之一，而且不像 weight decay 那样
        # 把输出推向「保守的模糊均值」（那正好会抹掉我们要检测的高频信号）。
        denoise_sigma = float(tcfg["denoise_sigma"])
        if denoise_sigma > 0:
            logger.info("去噪自编码器已启用：输入加高斯噪声 sigma=%.3f（目标保持干净）", denoise_sigma)
        es = tcfg["early_stopping"]
        patience = int(es["patience"]) if es["enabled"] else epochs
        min_delta = float(es["min_delta"])

        last_path = self.ckpt_dir / "last.pt"
        best_path = self.ckpt_dir / "best.pt"

        if start_epoch == 0:
            logger.info(
                "开始全新训练：%d 轮，batch=%d，设备=%s，AMP=%s",
                epochs, bs, self.device, amp_dtype if use_amp else "off",
            )
        else:
            logger.info(
                "继续训练：第 %d -> %d 轮，batch=%d，设备=%s，AMP=%s",
                start_epoch + 1, epochs, bs, self.device, amp_dtype if use_amp else "off",
            )

        t0 = time.time()
        elapsed_before = self.history.seconds
        since_best = self._since_best
        for epoch in range(start_epoch + 1, epochs + 1):
            self.model.train()
            run = torch.zeros((), dtype=torch.float64, device=self.device)
            n = 0
            for x in train_loader:
                clean = x.to(self.device, non_blocking=True)
                # 去噪目标：输入被污染，重建目标还是 clean
                noisy = clean
                if denoise_sigma > 0:
                    noisy = (clean + denoise_sigma * torch.randn_like(clean)).clamp_(0.0, 1.0)
                opt.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=self.device.type, dtype=amp_dtype, enabled=use_amp
                ):
                    recon = self.model(noisy)
                    loss = nn.functional.mse_loss(recon.float(), clean.float())
                scaler.scale(loss).backward()
                if clip > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)
                scaler.step(opt)
                scaler.update()
                batch_n = len(clean)
                run.add_(loss.detach().to(torch.float64), alpha=batch_n)
                n += batch_n
            if n == 0:
                raise ValueError("训练数据为空，无法计算训练损失")
            sched.step()
            train_loss = float(run) / n
            self.history.train_loss.append(train_loss)

            val_loss = self._eval_loss(val_loader)
            self.history.val_loss.append(val_loss)
            improved = val_loss < self.history.best_val_loss - min_delta
            if improved:
                self.history.best_val_loss = val_loss
                self.history.best_epoch = epoch
                self._best_model_sha256 = _model_state_sha256(self.model.state_dict())
                since_best = 0
            else:
                since_best += 1
            self._since_best = since_best
            self.history.stopped_early = bool(es["enabled"] and since_best >= patience)
            self.history.termination_reason = (
                "early_stopping" if self.history.stopped_early else (
                    "budget_completed" if epoch == epochs else None
                )
            )
            self.history.seconds = elapsed_before + time.time() - t0

            if self._best_model_sha256 is None:
                raise ValueError("尚无可核验的有限验证集best，拒绝保存伪最优检查点")
            # 先保存best，再保存引用其模型身份的last；不以当前轮补造旧best。
            payload = self._make_payload(epoch, opt, sched, val_loss)
            if improved:
                _atomic_save(payload, best_path)
            _atomic_save(payload, last_path)

            if epoch % 5 == 0 or epoch == start_epoch + 1:
                logger.info(
                    "  epoch %3d/%d  train=%.6f  val=%.6f  best=%.6f@%d%s",
                    epoch, epochs, train_loss, val_loss,
                    self.history.best_val_loss, self.history.best_epoch,
                    "  *" if improved else "",
                )
            if self.history.stopped_early:
                logger.info("  验证损失连续 %d 轮未改善，提前停止于 epoch %d", patience, epoch)
                break

        self.history.seconds = elapsed_before + time.time() - t0

        # 回滚到验证集选出的最优权重，再用独立校准集算零校准统计。
        # 重算而不是沿用旧值：续训后模型变了，旧的重建误差分布不再匹配。
        best_payload = self._load_verified_best({
            "best_model_sha256": self._best_model_sha256,
            "split_signature": self.split_signature, "seed": self.seed,
            "model_cfg": self.model_cfg, "best_epoch": self.history.best_epoch,
            "best_val_loss": self.history.best_val_loss,
        })
        self.model.load_state_dict(best_payload["model_state"])
        logger.info("已回滚到 epoch %d 的最优权重", best_payload.get("epoch", -1))
        self.model.to(self.device).eval()
        self.calibration = self._calibrate(calib_loader)

        best_epoch = int(best_payload["epoch"])
        best_train_loss = list(best_payload.get("train_loss", []))
        best_val_loss = list(best_payload.get("val_loss", []))
        if len(best_train_loss) > best_epoch or len(best_val_loss) > best_epoch:
            # 兼容旧版收尾时写入完整历史的best；这些摘要不能伪装成恢复点状态。
            best_payload["history"] = {
                **best_payload.get("history", {}),
                "epochs_run": best_epoch,
                "final_train_loss": best_train_loss[best_epoch - 1] if best_epoch else None,
                "seconds": 0.0,
                "stopped_early": False,
                "termination_reason": None,
            }
            best_payload["training_complete"] = False
            best_payload["termination_reason"] = None
        best_payload["train_loss"] = best_train_loss[:best_epoch]
        best_payload["val_loss"] = best_val_loss[:best_epoch]
        best_payload["scoring_cfg"] = self.scoring_cfg
        best_payload["calibration"] = self.calibration.to_dict()
        # epoch、优化器、RNG 与顶层历史始终属于 best 恢复点。
        # 全程摘要单独保存，不能让从 best 恢复把后续轮次重复追加到曲线末尾。
        best_payload["final_summary"] = {
            "history": self.history.to_dict(),
            "train_loss": list(self.history.train_loss),
            "val_loss": list(self.history.val_loss),
            "training_complete": True,
            "termination_reason": self.history.termination_reason,
            "planned_epochs": self.history.planned_epochs,
        }
        _atomic_save(best_payload, best_path)
        logger.info("检查点已保存: %s（last: %s）", best_path, last_path)
        return self.history

    @torch.no_grad()
    def _eval_loss(self, loader: DataLoader) -> float:
        self.model.eval()
        run = torch.zeros((), dtype=torch.float64, device=self.device)
        n = 0
        for x in loader:
            x = x.to(self.device, non_blocking=True)
            batch_n = len(x)
            loss = nn.functional.mse_loss(self.model(x).float(), x.float())
            run.add_(loss.to(torch.float64), alpha=batch_n)
            n += batch_n
        if n == 0:
            raise ValueError("验证数据为空，无法计算验证损失")
        return float(run) / n

    @torch.no_grad()
    def _calibrate(self, loader: DataLoader) -> CalibrationStats:
        """在留出的参考图上算重建误差分布，得到 z 分数的零点与尺度。"""
        self.model.eval()
        vals: list[float] = []
        for x in loader:
            means = reconstruction_mean(self.model, x.to(self.device))
            vals.extend(means.cpu().tolist())
        stats = CalibrationStats.from_errors(vals)
        logger.info(
            "零校准（%d 张正常天气图）：均值 %.6f，标准差 %.6f，p95 %.6f，p99 %.6f",
            stats.n, stats.mean, stats.std, stats.p95, stats.p99,
        )
        return stats
