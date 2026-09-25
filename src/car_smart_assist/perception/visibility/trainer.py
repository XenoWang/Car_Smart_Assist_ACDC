"""能见度门控的无监督训练（含检查点保存与续训）。

职责:
    - 在 ACDC 正常天气参考图上训练自编码器（自重建，无需任何标签）
    - 在留出的参考图上做零校准，得到正常天气的重建误差分布
    - **每一轮都保存检查点**，支持从已有权重继续训练（增量/增强训练）

无监督的含义:
    训练目标只有自重建，损失是 ||x - AE(x)||²，输入即目标，**没有任何人工标注**。
    参考图之所以能充当「正常」的定义，是因为它们按数据集构造就是清晰可见的场景 ——
    这是数据集给的属性，不是我们标的。因此整个流程不引入标注成本。

分割策略:
    参考图分为 train / validation / calibration / test。
    validation 只用于早停和选择 best checkpoint；calibration 只用于最终零校准；
    test 只用于最终评估。三个留出集合按序列互斥。

检查点语义（重要）:
    每次训练都在 checkpoint_dir 下维护两个文件：

        last.pt   每轮覆盖，用于崩溃恢复与续训。含模型 / 优化器 / 调度器 /
                  轮次 / 历史 / 零校准统计 / 模型配置
        best.pt   仅在验证损失创新低时覆盖，用于后续推理与评估

    默认行为是 **续训而非重建**（resume='auto'）：
    若 last.pt 已存在，自动载入权重、优化器与调度器状态，从下一轮接着训。
    要强制全新开始，传 resume='none' 或命令行加 --fresh。

    这样设计的理由：能见度门控的验证集是固定的，
    每次重跑都从零开始既浪费算力，也让「加数据后再训一轮」这类迭代无法进行。
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

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

    def to_dict(self) -> dict[str, Any]:
        return {
            "final_train_loss": self.train_loss[-1] if self.train_loss else None,
            "best_val_loss": self.best_val_loss,
            "best_epoch": self.best_epoch,
            "stopped_early": self.stopped_early,
            "epochs_run": len(self.train_loss),
            "start_epoch": self.start_epoch,
            "resumed_from": self.resumed_from,
            "seconds": round(self.seconds, 1),
        }


def resolve_device(spec: str = "auto") -> torch.device:
    """auto 优先使用当前 PyTorch 可用的 CUDA GPU；显式 CUDA 不可用时明确报错。"""
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

    直接覆盖原文件时若进程被中断（Ctrl-C、断电、OOM），会留下半个文件，
    下次续训读它直接崩，且原始权重也一起没了 —— 训练几小时的成果就这么丢掉。
    同目录下 rename 在 Windows 与 POSIX 上都是原子的。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def apply_config_overrides(cfg: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """按点号路径把覆盖项写入配置副本。

    例：``{"model.kernel_sizes": [3, 5]}`` 会写进 ``cfg["model"]["kernel_sizes"]``。

    放在这里而不是消融脚本里，是为了让**测试能用同一份实现**校验配置 ——
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
        self.cfg = cfg
        self.root = Path(project_root)
        self.device = resolve_device(str(cfg.get("device", "auto")))
        self.model_cfg = dict(cfg.get("model", {}))
        self.scoring_cfg = copy.deepcopy(cfg.get("scoring", {}))
        tcfg = cfg.get("train", {})
        self.ckpt_dir = self.root / tcfg.get(
            "checkpoint_dir", "artifacts/checkpoints/visibility"
        )
        self.resume_path = resolve_resume_spec(resume, self.ckpt_dir)
        self.history = TrainHistory()
        self.calibration: CalibrationStats | None = None
        self.scaler: torch.amp.GradScaler | None = None
        self._pending_scaler_state: dict | None = None
        self.split_signature: str | None = None
        # 由 _build_datasets 填入各数据分区的样本数
        self.split_stats: dict[str, int] = {}

        # 续训时模型结构必须与检查点一致，否则权重加载会报形状不匹配。
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
            ckpt_model_cfg = head.get("model_cfg")
            if ckpt_model_cfg and ckpt_model_cfg != self.model_cfg:
                logger.warning(
                    "配置中的模型结构与检查点不一致，**以检查点为准**以保证权重可加载。\n"
                    "  检查点: %s\n  当前配置: %s\n"
                    "若要按新结构训练，请加 --fresh 或换一个 checkpoint_dir",
                    ckpt_model_cfg, self.model_cfg,
                )
            del head

        # 续训时把生效的结构固化到 self.model_cfg。
        # 不能只在建模型时用一下检查点的结构就丢掉 —— _make_payload 写的是
        # self.model_cfg，若仍保留配置里的旧值，**下一次保存会把与权重不匹配的
        # 结构写进检查点**，后续再续训就彻底错乱了。
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

        test 子集**不在这里构建** —— 它由评估脚本单独划分使用，
        训练过程从不接触它，保证「只用于报告」的语义不被破坏。
        """
        from car_smart_assist.perception.visibility.dataset import (
            ensure_cache,
            list_ref_images,
            sequence_of,
            split_ref_indices,
        )

        dcfg = self.cfg.get("data", {})
        acdc_root = self.root / dcfg.get("acdc_root", "data/raw/acdc")
        paths = list_ref_images(acdc_root)
        if not paths:
            raise FileNotFoundError(
                f"未在 {acdc_root} 下找到正常天气参考图（*_rgb_ref_anon.png）。"
                "请先确认 ACDC 已解压，见 docs/dataset.md 5.1"
            )

        size = tuple(self.model_cfg.get("input_size", MODEL_DEFAULTS["input_size"]))
        cache_name = dcfg.get("cache_path")
        cache_path = (self.root / cache_name) if cache_name else None

        # 缓存按**全量**参考图构建一次（ensure_cache 只返回路径），
        # 各子集通过索引共享，在各自 worker 进程里惰性打开 memmap。
        cache_path = ensure_cache(paths, size, cache_path)

        # ACDC 参考图来自视频序列，相邻帧近乎重复。按图随机划分会把一帧放进
        # 训练集、它的邻居放进校准集 —— 校准集里混进了训练样本的复制品，
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
            "train": train_idx.tolist(),
            "val": val_idx.tolist(),
            "calib": calib_idx.tolist(),
            "test": test_idx.tolist(),
        }
        self.split_signature = hashlib.sha256(
            json.dumps(split_payload, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

        # 只有训练子集做增强。验证和校准均使用原图，保持评价口径一致。
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
            "format_version": 2,
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
        }

    def _restore(self, opt: torch.optim.Optimizer, sched: Any) -> int:
        """从检查点恢复全部状态，返回已完成的轮次。"""
        assert self.resume_path is not None
        # 模型和优化器的 load_state_dict 负责迁移，避免完整检查点先占用显存。
        ckpt = torch.load(self.resume_path, map_location="cpu", weights_only=False)

        if ckpt.get("split_signature") != self.split_signature:
            raise ValueError(
                "检查点与当前数据/切分不匹配，不能比较历史验证损失或继续早停。"
                "请确认数据和 split 配置未变，或使用 --fresh 重新训练。"
            )

        self.model.load_state_dict(ckpt["model_state"])
        self._pending_scaler_state = ckpt.get("scaler_state")
        if ckpt.get("optimizer_state") is not None:
            opt.load_state_dict(ckpt["optimizer_state"])
        else:
            logger.warning("检查点中没有优化器状态，动量会从零重建，续训初期损失可能短暂反弹")
        if ckpt.get("scheduler_state") is not None and sched is not None:
            sched.load_state_dict(ckpt["scheduler_state"])

        self.history.train_loss = list(ckpt.get("train_loss", []))
        self.history.val_loss = list(ckpt.get("val_loss", []))
        self.history.best_val_loss = float(ckpt.get("best_val_loss", float("inf")))
        self.history.best_epoch = int(ckpt.get("best_epoch", -1))
        self.history.resumed_from = str(self.resume_path)
        epoch = int(ckpt.get("epoch", 0))
        self.history.start_epoch = epoch

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
        # test 子集不在训练里构建 —— 它只由评估脚本使用
        train_ds, val_ds, calib_ds, _ = self._build_datasets()

        nw = int(dcfg.get("num_workers", 4))
        bs = int(tcfg["batch_size"])
        train_loader = DataLoader(
            train_ds, batch_size=bs, shuffle=True, num_workers=nw,
            pin_memory=self.device.type == "cuda", drop_last=False,
        )
        val_loader = DataLoader(
            val_ds, batch_size=bs, shuffle=False, num_workers=nw,
            pin_memory=self.device.type == "cuda",
        )
        calib_loader = DataLoader(
            calib_ds, batch_size=bs, shuffle=False, num_workers=nw,
            pin_memory=self.device.type == "cuda",
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
            if start_epoch >= epochs:
                # 续训时 epochs 的语义变为「再训多少轮」，而不是「总共多少轮」，
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
        # 去噪自编码器：只给**输入**加噪，重建目标仍是无噪的原图。
        # 这是重建类任务上最有效的正则化手段之一，且不像 weight decay 那样
        # 把输出推向「保守的模糊均值」（那会恰好抹掉我们要检测的高频信号）。
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
        since_best = 0
        for epoch in range(start_epoch + 1, epochs + 1):
            self.model.train()
            run = torch.zeros((), dtype=torch.float64, device=self.device)
            n = 0
            for x in train_loader:
                clean = x.to(self.device, non_blocking=True)
                # 去噪目标：输入被污染，重建目标仍是 clean
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
                since_best = 0
            else:
                since_best += 1

            # 每轮都写：last.pt 用于续训与崩溃恢复，best.pt 仅在创新低时更新。
            # 写 last.pt 时若尚未有 best，先用当前轮顶替，保证 best.pt 始终可用。
            _atomic_save(self._make_payload(epoch, opt, sched, val_loss), last_path)
            if improved or not best_path.exists():
                _atomic_save(self._make_payload(epoch, opt, sched, val_loss), best_path)

            if epoch % 5 == 0 or epoch == start_epoch + 1:
                logger.info(
                    "  epoch %3d/%d  train=%.6f  val=%.6f  best=%.6f@%d%s",
                    epoch, epochs, train_loss, val_loss,
                    self.history.best_val_loss, self.history.best_epoch,
                    "  *" if improved else "",
                )
            if since_best >= patience:
                self.history.stopped_early = True
                logger.info("  验证损失连续 %d 轮未改善，提前停止于 epoch %d", patience, epoch)
                break

        self.history.seconds = time.time() - t0

        # 回滚到验证集选出的最优权重，再用独立校准集计算零校准统计。
        # 重算而不是沿用旧值：续训后模型变了，旧的重建误差分布不再匹配。
        best_payload = torch.load(best_path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(best_payload["model_state"])
        logger.info("已回滚到 epoch %d 的最优权重", best_payload.get("epoch", -1))
        self.model.to(self.device).eval()
        self.calibration = self._calibrate(calib_loader)

        best_payload["scoring_cfg"] = self.scoring_cfg
        best_payload["calibration"] = self.calibration.to_dict()
        best_payload["history"] = self.history.to_dict()
        best_payload["train_loss"] = self.history.train_loss
        best_payload["val_loss"] = self.history.val_loss
        best_payload["best_val_loss"] = self.history.best_val_loss
        best_payload["best_epoch"] = self.history.best_epoch
        best_payload["current_val_loss"] = self.history.best_val_loss
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
        """在留出的参考图上算重建误差分布 —— z 分数的零点与尺度。"""
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
