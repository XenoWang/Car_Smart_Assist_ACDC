"能见度门控的无监督训练（含检查点保存与续训）。"

from __future__ import annotations

import copy
import hashlib
import json
import logging
import math
import os
import random
import shutil
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from filelock import FileLock

from car_smart_assist.config.visibility import (
    MODEL_DEFAULTS,
    training_config,
    validate_input_size,
)
from car_smart_assist.perception.visibility.autoencoder import (
    build_autoencoder,
    reconstruction_mean,
)
from car_smart_assist.perception.visibility.dataset import VisibilityImageDataset
from car_smart_assist.perception.visibility.scorer import (
    CalibrationStats,
    _load_visibility_checkpoint,
)

from car_smart_assist.utils.stop import StopController

logger = logging.getLogger(__name__)

ResumeSpec = Literal["auto", "none"] | str | Path | None


def load_visibility_config(path: str | Path) -> dict[str, Any]:
    """读取 CLI 配置，并在输入边界拒绝无法使用的配置结构。"""
    try:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ValueError(f"无法读取能见度配置 {path}；请检查路径、UTF-8 编码及 YAML 格式：{exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"能见度配置 {path} 必须为 YAML 字典")
    cfg = raw.get("visibility", raw)
    if not isinstance(cfg, dict):
        raise ValueError(f"能见度配置 {path} 的 visibility 必须为字典")
    sections = (
        "data", "model", "train", "report", "scoring", "thresholds",
        "train.early_stopping", "train.augmentation", "scoring.features",
        "scoring.feature_scales", "scoring.aggregation", "scoring.aggregation.weights",
        "scoring.reconstruction_error",
    )
    for section in sections:
        node = cfg
        for key in section.split("."):
            node = node.get(key, {})
        if not isinstance(node, dict):
            raise ValueError(f"能见度配置 {path} 的 {section} 必须为字典")
    for section, key in (
        ("data", "acdc_root"), ("data", "project_root"), ("data", "cleaning_manifest"),
        ("data", "cache_path"), ("train", "checkpoint_dir"), ("report", "output_dir"),
    ):
        values = cfg.get(section, {})
        value = values.get(key)
        optional = key in ("cleaning_manifest", "cache_path")
        if key in values and not (optional and value is None) and not isinstance(value, (str, os.PathLike)):
            raise ValueError(f"能见度配置 {path} 的 {section}.{key} 必须为路径字符串")
    return cfg


def validate_visibility_manifest(cleaning_manifest: str | Path | None, project_root: str | Path) -> None:
    """先验证清洗清单结构，避免共享读取器将坏输入误报为程序异常。"""
    if cleaning_manifest is None:
        return
    path = Path(project_root) / cleaning_manifest
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取清洗清单 {path}；请检查路径或重新运行 scripts/clean_data.py：{exc}") from exc
    if (not isinstance(payload, dict) or not isinstance(payload.get("invalid"), list)
            or any(not isinstance(entry, str) for entry in payload["invalid"])):
        raise ValueError(f"清洗清单 {path} 的 invalid 必须为字符串列表；请重新运行 scripts/clean_data.py")


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
    training_complete: bool = False
    manual_stop: bool = False

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
            "training_complete": self.training_complete,
            "manual_stop": self.manual_stop,
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
    try:
        device = torch.device(spec)
    except RuntimeError as exc:
        raise ValueError(f"无效能见度设备 {spec!r}；请使用 auto、cpu 或 cuda[:编号]") from exc
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


_TRANSACTION_DIRECTORY = ".visibility_checkpoint_transaction"
_CHECKPOINT_NAMES = ("best.pt", "last.pt")
_TRANSACTION_FILES = {
    "owner.json", "journal.json", "journal.json.pending",
    *_CHECKPOINT_NAMES,
    *(name + ".tmp" for name in _CHECKPOINT_NAMES),
    *(name + ".before" for name in _CHECKPOINT_NAMES),
}


def _checkpoint_file_sha256(path: Path) -> str | None:
    if path.is_symlink() or path.resolve() != path:
        raise ValueError(f"拒绝链接检查点或事务文件，保留原件：{path}")
    if not path.exists():
        return None
    if not path.is_file() or path.stat().st_nlink != 1:
        raise ValueError(f"检查点或事务文件必须是非链接普通文件：{path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_checkpoint_bytes(path: Path, payload: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    if path.read_bytes() != payload:
        raise OSError(f"事务写入回读不一致，保留证据：{path}")


def _visibility_transaction_state(directory: Path):
    directory = Path(os.path.abspath(directory))
    if directory.is_symlink() or directory.resolve() != directory:
        raise ValueError(f"检查点目录不能包含链接：{directory}")
    transaction = directory / _TRANSACTION_DIRECTORY
    if transaction.is_symlink() or transaction.resolve() != transaction:
        raise ValueError(f"事务目录不能包含链接，保留证据：{transaction}")
    if not transaction.exists():
        return directory, transaction, {}, None
    if not transaction.is_dir():
        raise ValueError(f"事务路径不是目录，保留证据：{transaction}")
    files = {path.name: path for path in transaction.iterdir()}
    for name, path in files.items():
        if name not in _TRANSACTION_FILES:
            raise ValueError(f"事务目录含未知文件，保留证据：{path}")
        _checkpoint_file_sha256(path)
    owner = None
    if "owner.json" in files:
        try:
            owner = json.loads(files["owner.json"].read_text(encoding="utf-8"))
            if (
                set(owner) != {"format_version", "transaction_id", "checkpoint_directory"}
                or type(owner["format_version"]) is not int or owner["format_version"] != 1
                or owner["checkpoint_directory"] != str(directory)
                or not isinstance(owner["transaction_id"], str)
                or len(owner["transaction_id"]) != 32
                or any(char not in "0123456789abcdef" for char in owner["transaction_id"])
            ):
                raise ValueError("ownership 标记无效")
        except (ValueError, TypeError, KeyError) as exc:
            raise ValueError(f"事务目录归属无法确认，保留原件：{transaction}") from exc
    return directory, transaction, files, owner


def recover_visibility_checkpoint_transaction(directory: Path) -> bool:
    """持有 run.lock 时核验并前滚未来保存的事务，不猜测历史混合 pair。"""
    directory, transaction, files, owner = _visibility_transaction_state(directory)
    if not transaction.exists():
        return False
    if not files:
        transaction.rmdir()
        return False
    if "journal.json" not in files:
        if owner is None:
            raise ValueError(f"无 journal 且无法确认事务归属，保留原件：{transaction}")
        archive_visibility_checkpoints(directory, transaction_only=True)
        return False
    journal_bytes = files["journal.json"].read_bytes()
    try:
        journal = json.loads(journal_bytes)
        if (
            set(journal) != {"format_version", "transaction_id", "checkpoint_directory",
                             "before_sha256", "after_sha256", "updates"}
            or type(journal["format_version"]) is not int or journal["format_version"] != 1
            or journal["checkpoint_directory"] != str(directory)
            or not isinstance(journal["transaction_id"], str)
            or len(journal["transaction_id"]) != 32
            or any(char not in "0123456789abcdef" for char in journal["transaction_id"])
            or (owner is not None and owner["transaction_id"] != journal["transaction_id"])
        ):
            raise ValueError("事务身份无效")
        before, after, updates = journal["before_sha256"], journal["after_sha256"], journal["updates"]
        if (
            not isinstance(before, dict) or set(before) != set(_CHECKPOINT_NAMES)
            or not isinstance(after, dict) or set(after) != set(_CHECKPOINT_NAMES)
            or not isinstance(updates, dict) or not updates or not set(updates) <= set(_CHECKPOINT_NAMES)
        ):
            raise ValueError("事务目标无效")
        for value in (*before.values(), *after.values()):
            if value is not None and (
                not isinstance(value, str) or len(value) != 64
                or any(char not in "0123456789abcdef" for char in value)
            ):
                raise ValueError("事务摘要无效")
        expected_files = {"owner.json", "journal.json"}
        for name in _CHECKPOINT_NAMES:
            if name not in updates:
                if before[name] != after[name]:
                    raise ValueError("未更新目标的摘要改变")
                continue
            record = updates[name]
            if (
                not isinstance(record, dict)
                or set(record) != {"candidate", "candidate_sha256", "backup", "backup_sha256"}
                or record["candidate"] != name or record["candidate_sha256"] != after[name]
                or after[name] is None or record["backup_sha256"] != before[name]
                or record["backup"] != (name + ".before" if before[name] is not None else None)
            ):
                raise ValueError("事务候选或备份记录无效")
            expected_files.add(name)
            if record["backup"] is not None:
                expected_files.add(name + ".before")
        if not files.keys() <= expected_files:
            raise ValueError("事务含不属于该 journal 的文件")
    except (ValueError, TypeError, KeyError) as exc:
        raise ValueError(f"保存事务 journal 无效，保留原件：{transaction}") from exc

    current = {name: _checkpoint_file_sha256(directory / name) for name in _CHECKPOINT_NAMES}
    if any(current[name] not in (before[name], after[name]) for name in _CHECKPOINT_NAMES):
        raise ValueError(f"正式检查点丢失或摘要不属于事务，拒绝恢复：{transaction}")
    all_after = current == after
    if owner is None and not all_after:
        raise ValueError(f"未完成事务缺少 ownership 标记，保留原件：{transaction}")
    # 先核验整对和全部待发布候选，不能先修 best 再发现 last 无法发布。
    for name, record in updates.items():
        candidate = _checkpoint_file_sha256(transaction / name)
        if candidate is not None and candidate != after[name]:
            raise ValueError(f"事务候选摘要不匹配：{transaction / name}")
        if current[name] != after[name] and candidate is None:
            raise ValueError(f"事务待发布候选缺失：{transaction / name}")
        if record["backup"] is not None:
            backup = _checkpoint_file_sha256(transaction / (name + ".before"))
            if backup is not None and backup != before[name]:
                raise ValueError(f"事务旧文件备份摘要不匹配：{transaction / (name + '.before')}")
            if not all_after and backup is None:
                raise ValueError(f"未完成事务缺少旧文件备份：{transaction / (name + '.before')}")
    for name in _CHECKPOINT_NAMES:
        if name not in updates:
            continue
        observed = _checkpoint_file_sha256(directory / name)
        if observed not in (before[name], after[name]):
            raise ValueError(f"发布前正式检查点改变，保留事务：{directory / name}")
        if observed != after[name]:
            candidate = transaction / name
            if _checkpoint_file_sha256(candidate) != after[name]:
                raise ValueError(f"发布前候选改变，保留事务：{candidate}")
            os.replace(candidate, directory / name)
            if _checkpoint_file_sha256(directory / name) != after[name]:
                raise OSError(f"发布检查点回读不一致，保留事务：{directory / name}")
    if {name: _checkpoint_file_sha256(directory / name) for name in _CHECKPOINT_NAMES} != after:
        raise ValueError(f"事务发布后整对摘要不匹配，保留证据：{transaction}")
    _, _, remaining, _ = _visibility_transaction_state(directory)
    if not remaining.keys() <= expected_files or remaining["journal.json"].read_bytes() != journal_bytes:
        raise ValueError(f"清理前事务内容改变，保留证据：{transaction}")
    for name, path in remaining.items():
        if name not in {"owner.json", "journal.json"}:
            path.unlink()
    (transaction / "owner.json").unlink(missing_ok=True)
    (transaction / "journal.json").unlink()
    transaction.rmdir()
    return True


def _publish_checkpoint_transaction(directory: Path, updates: dict[str, dict[str, Any]]) -> None:
    """备份与候选全部持久化后登记事务，再按 best→last 发布。调用者持 run.lock。"""
    if not updates or not set(updates) <= set(_CHECKPOINT_NAMES):
        raise ValueError("AE 保存事务只允许更新 best.pt/last.pt")
    recover_visibility_checkpoint_transaction(directory)
    directory, transaction, _, _ = _visibility_transaction_state(directory)
    directory.mkdir(parents=True, exist_ok=True)
    before = {name: _checkpoint_file_sha256(directory / name) for name in _CHECKPOINT_NAMES}
    transaction.mkdir()
    owner = {"format_version": 1, "transaction_id": uuid.uuid4().hex,
             "checkpoint_directory": str(directory)}
    _write_checkpoint_bytes(transaction / "owner.json", json.dumps(owner, indent=2).encode("utf-8"))
    for name in _CHECKPOINT_NAMES:
        if name in updates and before[name] is not None:
            original = (directory / name).read_bytes()
            if hashlib.sha256(original).hexdigest() != before[name]:
                raise ValueError(f"备份前正式检查点改变：{directory / name}")
            _write_checkpoint_bytes(transaction / (name + ".before"), original)
    after = dict(before)
    records = {}
    for name in _CHECKPOINT_NAMES:
        if name not in updates:
            continue
        candidate = transaction / name
        _atomic_save(updates[name], candidate)
        original = candidate.read_bytes()
        with candidate.open("r+b") as stream:
            stream.flush()
            os.fsync(stream.fileno())
        if candidate.read_bytes() != original:
            raise OSError(f"候选检查点回读不一致：{candidate}")
        after[name] = hashlib.sha256(original).hexdigest()
        records[name] = {"candidate": name, "candidate_sha256": after[name],
                         "backup": name + ".before" if before[name] is not None else None,
                         "backup_sha256": before[name]}
    if {name: _checkpoint_file_sha256(directory / name) for name in _CHECKPOINT_NAMES} != before:
        raise ValueError(f"准备事务期间正式检查点改变，保留证据：{transaction}")
    journal = {**owner, "before_sha256": before, "after_sha256": after, "updates": records}
    journal_bytes = json.dumps(journal, indent=2).encode("utf-8")
    pending = transaction / "journal.json.pending"
    _write_checkpoint_bytes(pending, journal_bytes)
    os.replace(pending, transaction / "journal.json")
    if (transaction / "journal.json").read_bytes() != journal_bytes:
        raise OSError(f"事务 journal 发布回读不一致：{transaction}")
    recover_visibility_checkpoint_transaction(directory)


def archive_visibility_checkpoints(
    directory: Path, report_path: Path | None = None, *, transaction_only: bool = False
) -> None:
    """Preserve explicit-fresh artifacts before model initialization."""
    directory, transaction, transaction_files, owner = _visibility_transaction_state(directory)
    if transaction.exists() and not transaction_files:
        transaction.rmdir()
    if transaction.exists() and owner is None:
        raise ValueError(f"无法确认残事务归属，拒绝归档或删除：{transaction}")
    paths = [] if transaction_only else [directory / "last.pt", directory / "best.pt"]
    if report_path is not None and not transaction_only:
        report_path = Path(os.path.abspath(report_path))
        paths.append(report_path)
    paths.extend(transaction_files.values())
    for path in paths:
        _checkpoint_file_sha256(path)
    existing = {path: path.read_bytes() for path in paths if path.is_file()}
    if not existing:
        return
    history = directory / "history"
    if history.is_symlink() or history.resolve() != history:
        raise ValueError(f"归档目录不能包含链接：{history}")
    archive = history / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    archive.mkdir(parents=True)
    for path, original in existing.items():
        target = archive / (_TRANSACTION_DIRECTORY if path.parent == transaction else "") / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        with target.open("r+b") as stream:
            stream.flush()
            os.fsync(stream.fileno())
        if target.read_bytes() != original or path.read_bytes() != original:
            raise OSError(f"检查点归档回读不一致：{path}")
    if any(path.read_bytes() != original for path, original in existing.items()):
        raise OSError("归档期间旧训练产物改变，保留原件并停止")
    # 归档全部回读一致后才删原件；事务标记仍留到最后。
    ordered = sorted(existing, key=lambda path: (
        path.parent == transaction,
        2 if path.name == "journal.json" else 1 if path.name == "owner.json" else 0,
        str(path),
    ))
    for path in ordered:
        path.unlink()
    if transaction.exists():
        transaction.rmdir()


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


def resolve_resume_spec(
    resume: ResumeSpec, ckpt_dir: Path, project_root: str | Path = "."
) -> Path | None:
    """决定从哪个检查点续训。

    Args:
        resume: 'auto' / None  -> 优先 last.pt，其次 best.pt；都不存在则返回 None
                'none' / False -> 强制全新训练，返回 None
                其他            -> 视为显式路径，相对路径按 project_root 解析
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
    if not p.is_absolute():
        p = Path(project_root) / p
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
        self.root = Path(project_root).resolve()
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
        automatic = resume is None or (isinstance(resume, str) and resume.lower() == "auto")
        fresh = resume is False or (isinstance(resume, str) and resume.lower() == "none")
        explicit = None if automatic or fresh else Path(resume)
        if explicit is not None and not explicit.is_absolute():
            explicit = self.root / explicit
        self._resume_spec = explicit if explicit is not None else resume
        local = automatic or (explicit is not None and Path(os.path.abspath(explicit)) in {
            Path(os.path.abspath(self.ckpt_dir / name)) for name in _CHECKPOINT_NAMES
        })
        self._deferred_resume = not fresh and local and (
            self.ckpt_dir / _TRANSACTION_DIRECTORY / "journal.json"
        ).exists()
        self.resume_path = None if self._deferred_resume else resolve_resume_spec(
            self._resume_spec, self.ckpt_dir, self.root
        )
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
        self._initialize_model()

    def _initialize_model(self) -> None:
        # 续训时模型结构必须与检查点一致，否则加载权重会报形状不匹配。
        # 先把检查点里的 model_cfg 读出来，以它为准构建模型。
        ckpt_model_cfg: dict[str, Any] | None = None
        if self.resume_path is not None:
            head = _load_visibility_checkpoint(self.resume_path)
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

        # 未完成事务的恢复尚未决定有效结构，等待 fit 持锁恢复后再校验。
        if not self._deferred_resume:
            validate_input_size(
                self.cfg.get("data", {}),
                tuple(self.model_cfg.get("input_size", MODEL_DEFAULTS["input_size"])),
            )
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
            reference_split_signature,
        )

        dcfg = self.cfg.get("data", {})
        acdc_root = self.root / dcfg.get("acdc_root", "data/raw/acdc")
        validate_visibility_manifest(
            dcfg.get("cleaning_manifest"), self.root / dcfg.get("project_root", ".")
        )
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

        self.split_signature = reference_split_signature(
            paths, acdc_root, size, groups, train_idx, val_idx, calib_idx, test_idx
        )

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
            "training_complete": self.history.training_complete,
            "manual_stop": self.history.manual_stop,
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
                "cuda_device_ids": [
                    str(torch.cuda.get_device_properties(i).uuid)
                    for i in range(torch.cuda.device_count())
                ] if self.device.type == "cuda" else [],
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
        best = _load_visibility_checkpoint(path)
        for key in ("split_signature", "seed", "model_cfg", "best_epoch", "best_val_loss"):
            if key not in reference or key not in best or best[key] != reference[key]:
                raise ValueError(f"历史best与恢复记录的{key}不匹配：{path}，拒绝完整续训")
        if (best.get("best_model_sha256") != expected
                or best.get("epoch") != reference["best_epoch"]
                or best.get("current_val_loss") != reference["best_val_loss"]
                or _model_state_sha256(best["model_state"]) != expected):
            raise ValueError(f"历史best权重身份或指标不匹配：{path}，拒绝完整续训")
        return best

    def _restore(self, opt: torch.optim.Optimizer, sched: Any, *, require_full_state=False) -> int:
        """从检查点恢复全部状态，返回已完成的轮次。"""
        assert self.resume_path is not None
        # 用模型和优化器的 load_state_dict 迁移，避免完整检查点先占显存。
        ckpt = _load_visibility_checkpoint(self.resume_path)
        if require_full_state and any(
            ckpt.get(key) is None
            for key in ("optimizer_state", "scheduler_state", "scaler_state", "rng_state")
        ):
            raise ValueError("--continue 需要包含优化器、调度器、AMP 和 RNG 的完整 last 状态")

        if ckpt.get("split_signature") != self.split_signature:
            raise ValueError(
                "检查点与当前数据/切分不匹配，不能比较历史验证损失或继续早停。"
                "请确认数据和 split 配置未变，或使用 --fresh 重新训练。"
            )

        rng_state = ckpt.get("rng_state")
        if self.device.type == "cuda" and rng_state is not None and rng_state["cuda"]:
            saved_devices = rng_state.get("cuda_device_ids")
            if not saved_devices:
                raise ValueError(
                    "旧AE检查点缺少CUDA设备身份，无法核验RNG布局，不能完整CUDA续训；"
                    "请保留原件并用 --fresh 新开实验，旧权重仍可推理。"
                )
            current_devices = [
                str(torch.cuda.get_device_properties(i).uuid)
                for i in range(torch.cuda.device_count())
            ]
            if len(rng_state["cuda"]) != len(current_devices) or saved_devices != current_devices:
                raise ValueError(
                    "CUDA设备数量、顺序或身份已变化，不能完整恢复AE随机状态；"
                    "请恢复原可见GPU布局，或使用 --fresh 新开实验。"
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
        self._resume_termination_reason = ckpt.get("termination_reason")
        self._resume_training_complete = (
            not trimmed_history and bool(ckpt.get("training_complete", False))
        )
        self.history.seconds = (
            0.0 if trimmed_history else float(ckpt.get("history", {}).get("seconds", 0.0))
        )
        self._since_best = int(ckpt.get("early_stopping_state", {}).get("wait", 0))
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

    def fit(self, *, stop_controller: StopController | None = None, continue_only: bool = False) -> TrainHistory:
        controller = stop_controller or StopController(default_directory=self.ckpt_dir)
        controller.watch_directory(self.ckpt_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        with FileLock(str((self.ckpt_dir / "run.lock").resolve()), timeout=0, is_singleton=True), controller:
            controller.require_running()
            if self._resume_spec is False or (
                isinstance(self._resume_spec, str) and self._resume_spec.lower() == "none"
            ):
                archive_visibility_checkpoints(
                    self.ckpt_dir,
                    self.root / self.cfg.get("report", {}).get("output_dir", "artifacts/reports/visibility") / "train_history.json",
                )
            else:
                recovered = recover_visibility_checkpoint_transaction(self.ckpt_dir)
                previous = self.resume_path
                self.resume_path = resolve_resume_spec(self._resume_spec, self.ckpt_dir, self.root)
                reinitialize = self._deferred_resume or recovered or self.resume_path != previous
                self._deferred_resume = False
                if reinitialize:
                    self._initialize_model()
            if continue_only and (
                self.resume_path is None
                or self.resume_path.resolve() != (self.ckpt_dir / "last.pt").resolve()
            ):
                raise ValueError("--continue 必须恢复本输出目录的配对 last.pt")
            return self._fit(controller, continue_only=continue_only)

    def _fit(self, stop_controller: StopController, *, continue_only=False) -> TrainHistory:
        """执行无监督训练（必要时先恢复已有检查点）。"""
        tcfg = training_config(self.cfg.get("train", {}))
        dcfg = self.cfg.get("data", {})
        stop_controller.require_running()
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
        completed_stage = False
        if self.resume_path is not None:
            start_epoch = self._restore(opt, sched, require_full_state=continue_only)
            if continue_only:
                epochs = self.history.planned_epochs
                if epochs <= 0 or epochs < start_epoch:
                    raise ValueError("--continue 的 last 缺少有效原累计训练目标")
                if self._resume_training_complete or start_epoch == epochs:
                    self.history.training_complete = True
                    self.history.termination_reason = (
                        "budget_completed" if start_epoch == epochs else "early_stopping"
                    )
                    self.history.stopped_early = self.history.termination_reason == "early_stopping"
                    logger.info("原训练阶段已完成（%d/%d），--continue 不追加轮数", start_epoch, epochs)
                    completed_stage = True
                if not completed_stage and any(not np.isfinite(group["lr"]) or group["lr"] <= 0 for group in opt.param_groups):
                    raise ValueError("恢复的学习率已为0或无效，不能继续原训练阶段")
            else:
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
        if not completed_stage:
            self.history.termination_reason = None
            self.history.training_complete = False
            self.history.stopped_early = False
        self.history.manual_stop = False

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
        remaining = () if completed_stage else range(start_epoch + 1, epochs + 1)
        for epoch in remaining:
            if stop_controller.requested:
                break
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
                if not torch.isfinite(loss):
                    raise ValueError("训练损失非有限，保留上一轮检查点并停止")
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), clip if clip > 0 else float("inf"),
                    error_if_nonfinite=not scaler.is_enabled(),
                )
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
            if not math.isfinite(train_loss) or not math.isfinite(val_loss):
                raise ValueError("训练或验证损失非有限，拒绝发布检查点")
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
            self.history.training_complete = self.history.stopped_early or epoch == epochs
            self.history.seconds = elapsed_before + time.time() - t0

            if self._best_model_sha256 is None:
                raise ValueError("尚无可核验的有限验证集best，拒绝保存伪最优检查点")
            # 先保存best，再保存引用其模型身份的last；不以当前轮补造旧best。
            payload = self._make_payload(epoch, opt, sched, val_loss)
            updates = {"best.pt": payload} if improved else {}
            updates["last.pt"] = payload
            _publish_checkpoint_transaction(self.ckpt_dir, updates)

            if epoch % 5 == 0 or epoch == start_epoch + 1:
                logger.info(
                    "  epoch %3d/%d  train=%.6f  val=%.6f  best=%.6f@%d%s",
                    epoch, epochs, train_loss, val_loss,
                    self.history.best_val_loss, self.history.best_epoch,
                    "  *" if improved else "",
                )
            if stop_controller.requested:
                break
            if self.history.stopped_early:
                logger.info("  验证损失连续 %d 轮未改善，提前停止于 epoch %d", patience, epoch)
                break

        self.history.seconds = elapsed_before + time.time() - t0

        def pause_if_requested():
            if not stop_controller.requested:
                return False
            if len(self.history.train_loss) == start_epoch:
                stop_controller.require_running()
            self.history.manual_stop = True
            self.history.termination_reason = "manual_stop"
            if not last_path.is_file():
                stop_controller.require_running()
            last = _load_visibility_checkpoint(last_path)
            last.update(
                history=self.history.to_dict(), manual_stop=True,
                termination_reason="manual_stop", training_complete=self.history.training_complete,
            )
            _publish_checkpoint_transaction(self.ckpt_dir, {"last.pt": last})
            return True

        if pause_if_requested():
            return self.history

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
        if pause_if_requested():
            return self.history

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
        updates = {"best.pt": best_payload}
        if completed_stage:
            last = _load_visibility_checkpoint(last_path)
            last.update(
                history=self.history.to_dict(), training_complete=True,
                manual_stop=False, termination_reason=self.history.termination_reason,
            )
            updates["last.pt"] = last
        _publish_checkpoint_transaction(self.ckpt_dir, updates)
        pause_if_requested()
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
            if not torch.isfinite(loss):
                raise ValueError("验证损失非有限，保留上一轮检查点并停止")
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
