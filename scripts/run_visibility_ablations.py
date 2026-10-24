"""能见度门控的消融实验运行脚本。"""

from __future__ import annotations

import argparse
from filelock import FileLock
import hashlib
import json
import logging
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from car_smart_assist.utils.stop import STOP_EXIT_CODE, StopController  # noqa: E402

from car_smart_assist.config.visibility import MODEL_DEFAULTS  # noqa: E402
from car_smart_assist.perception.visibility import (  # noqa: E402
    VisibilityGate,
    VisibilityLevel,
    VisibilityScorer,
    degrade,
    resolve_device,
)
from car_smart_assist.perception.visibility.dataset import (  # noqa: E402
    list_ref_images,
    read_rgb_image,
    sequence_of,
    split_ref_indices,
)
from car_smart_assist.perception.visibility.trainer import (  # noqa: E402
    VisibilityTrainer,
    apply_config_overrides,
    archive_visibility_checkpoints,
)

logger = logging.getLogger(__name__)

# 评估规模；数据配置相同的预设共享参考图与划分。
N_PER_SPLIT = 60          # train / val / calib / test 各抽多少张
N_SYNTH = 60              # 合成退化实验的基准图数量
SEVERITIES = (0.5, 0.75, 1.0)
KINDS = ("fog", "darkness", "occlusion", "blur")
MAX_SEV = SEVERITIES[-1]


def load_refs(paths, size) -> list[np.ndarray]:
    return [read_rgb_image(path, size) for path in paths]


def _signature(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_identity(path: Path) -> dict[str, str]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": path.resolve().as_posix(), "sha256": digest.hexdigest()}


def _prepare_inputs(cfg: dict[str, Any], root: Path) -> dict[str, Any]:
    dcfg = cfg.get("data", {})
    if dcfg.get("split_mode", "fixed") == "random":
        raise ValueError("消融不能独立重建训练和评估随机分区；请设置 data.split_mode=fixed")
    acdc_root = root / dcfg.get("acdc_root", "data/raw/acdc")
    cleaning_root = root / dcfg.get("project_root", ".")
    manifest = dcfg.get("cleaning_manifest")
    paths = list_ref_images(
        acdc_root, cleaning_manifest=manifest, project_root=cleaning_root
    )
    if not paths:
        raise ValueError("未找到正常天气参考图")
    groups = [sequence_of(p) for p in paths] if dcfg.get("split_by_sequence", True) else None
    splits = split_ref_indices(
        len(paths),
        train_ratio=float(dcfg.get("train_ratio", 0.80)),
        val_ratio=float(dcfg.get("val_ratio", 0.05)),
        calib_ratio=float(dcfg.get("calib_ratio", 0.05)),
        test_ratio=float(dcfg.get("test_ratio", 0.10)),
        seed=int(dcfg.get("split_seed", 42)),
        mode=str(dcfg.get("split_mode", "fixed")),
        groups=groups,
    )
    data_identity = {
        "acdc_root": acdc_root.resolve().as_posix(),
        "reference_count": len(paths),
        "ordered_reference_sha256": _signature([_file_identity(p) for p in paths]),
        "cleaning_manifest": _file_identity(cleaning_root / manifest) if manifest else None,
    }
    size = tuple(cfg.get("model", {}).get("input_size", MODEL_DEFAULTS["input_size"]))
    protocol = {
        "version": "visibility-ablation-val-v2",
        "implementation_sha256": _signature([
            _file_identity(root / name) for name in (
                "scripts/run_visibility_ablations.py",
                "src/car_smart_assist/config/visibility.py",
                "src/car_smart_assist/perception/visibility/scorer.py",
                "src/car_smart_assist/perception/visibility/gate.py",
                "src/car_smart_assist/perception/visibility/dataset.py",
                "src/car_smart_assist/perception/visibility/degradation.py",
                "src/car_smart_assist/perception/visibility/autoencoder.py",
                "src/car_smart_assist/perception/visibility/trainer.py",
            )
        ]),
        "input_size": list(size),
        "decode": "RGB-uint8-Pillow-BILINEAR",
        "grouping": "recording_family-v1" if groups is not None else "image",
        "groups_sha256": _signature(groups),
        "split_indices": {
            name: indices.tolist()
            for name, indices in zip(("train", "val", "calib", "test"), splits, strict=True)
        },
        "sample_per_split": N_PER_SPLIT,
        "sample_seed": 0,
        "sample_order": ["train", "val", "calib"],
        "synthetic_base": "sampled-validation",
        "synthetic_seed": "enumerated-base-index",
        "synthetic_kinds": list(KINDS),
        "severities": list(SEVERITIES),
        "metric": "BLIND-rate-at-maximum-severity",
    }
    return {"paths": paths, "size": size, "splits": splits,
            "data_identity": data_identity, "protocol": protocol}



# 评估（对比用的精简指标）



def evaluate_run(
    scorer: VisibilityScorer,
    gate: VisibilityGate,
    refs: list[np.ndarray],
    tr_idx: np.ndarray,
    val_idx: np.ndarray,
    ca_idx: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """只用训练集和验证集比较，测试集留给最终评估。"""

    def pick(idx):
        return rng.choice(idx, size=min(N_PER_SPLIT, len(idx)), replace=False)

    tr_i, val_i, ca_i = pick(tr_idx), pick(val_idx), pick(ca_idx)
    tr = scorer.score_arrays([refs[i] for i in tr_i])
    val = scorer.score_arrays([refs[i] for i in val_i])
    ca = scorer.score_arrays([refs[i] for i in ca_i])

    tr_m = float(np.mean([s.recon_mean for s in tr]))
    val_m = float(np.mean([s.recon_mean for s in val]))
    ca_m = float(np.mean([s.recon_mean for s in ca]))
    val_verdicts = gate.judge_many(val)
    val_fpr = sum(1 for v in val_verdicts if v.level is VisibilityLevel.BLIND) / max(len(val), 1)

    # 合成退化：召回率 + 信息量
    base = [refs[i] for i in val_i]
    synth: dict[str, Any] = {}
    for kind in KINDS:
        curve = []
        for sev in SEVERITIES:
            imgs = [degrade(b, kind, sev, seed=i)[0] for i, b in enumerate(base)]
            scores = scorer.score_arrays(imgs)
            verdicts = gate.judge_many(scores)
            blind = sum(1 for v in verdicts if v.level is VisibilityLevel.BLIND) / max(len(verdicts), 1)
            info = float(np.mean([s.information for s in scores]))
            curve.append({"severity": sev, "blind_rate": blind, "info": info})
        synth[kind] = {
            "curve": curve,
            "recall": curve[-1]["blind_rate"],
            "info_at_max": curve[-1]["info"],
        }

    return {
        "params_m": sum(p.numel() for p in scorer.model.parameters()) / 1e6,
        "recon_train": tr_m,
        "recon_val": val_m,
        "recon_calib": ca_m,
        "overfit_ratio": val_m / tr_m if tr_m > 0 else float("nan"),
        "val_fpr": val_fpr,
        "synth": synth,
        "mean_recall": float(np.mean([synth[k]["recall"] for k in KINDS])),
    }



# 主流程



def main() -> int:
    ap = argparse.ArgumentParser(description="能见度门控消融实验")
    ap.add_argument("--config", default="configs/model/visibility.yaml")
    ap.add_argument("--only", nargs="*", default=None, help="只跑指定 tag")
    ap.add_argument("--list", action="store_true", help="只列出预设")
    ap.add_argument("--epochs", type=int, default=None, help="覆盖训练轮数（调试用）")
    ap.add_argument("--force", action="store_true", help="忽略已有结果，全部重跑")
    ap.add_argument(
        "--force-meaningless", action="store_true",
        help="明知门控判定与模型无关（use_recon_z=false）仍要跑，"
             "例如只想比较重建质量。",
    )
    ap.add_argument("--log-level", default="INFO")
    modes = ap.add_mutually_exclusive_group()
    modes.add_argument("--stop", action="store_true", help="只写停止请求")
    modes.add_argument("--continue", dest="continue_training", action="store_true", help="继续原组进度与预算")
    modes.add_argument("--fresh", action="store_true", help="归档后从头重训所选组")
    ap.add_argument("--stop-file", type=Path, default=None)
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    root = Path(__file__).resolve().parents[1]
    raw = yaml.safe_load((root / args.config).read_text(encoding="utf-8"))
    base = raw["visibility"]

    presets = [{"tag": "baseline", "set": {}}] + list(base.get("ablations", []))
    if args.list:
        for p in presets:
            print(f"  {p['tag']:20s} {p.get('set', {})}")
        return 0
    if args.only:
        presets = [p for p in presets if p["tag"] in set(args.only)]
        if not presets:
            logger.error("--only 指定的 tag 都不存在")
            return 2

    if any(apply_config_overrides(base, p.get("set", {})).get("data", {}).get("split_mode", "fixed") == "random"
           for p in presets):
        logger.error("消融不能独立重建训练和评估随机分区；请设置 data.split_mode=fixed")
        return 2

    with StopController(
        args.stop_file, default_directory=root / "artifacts/checkpoints/visibility_ablations"
    ) as controller:
        for preset in presets:
            controller.watch_directory(root / "artifacts/checkpoints/visibility_ablations" / preset["tag"])
        if args.stop:
            controller.request_stop_file()
            print(f"已请求轮末停止：{controller.path}")
            return 0
        if args.fresh or args.continue_training:
            controller.resume()
        controller.require_running()
        if args.continue_training and args.force:
            logger.error("--continue 不能与重新执行全部组的 --force 一起使用")
            return 2
        out_dir = root / "artifacts/reports/visibility/ablations"
        out_dir.mkdir(parents=True, exist_ok=True)
        results_path = out_dir / "results.json"
        if args.fresh or args.force:
            old = [p for p in (results_path, out_dir / "training_progress.json") if p.is_file()]
            if old:
                archive = out_dir / "history" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
                archive.mkdir(parents=True)
                for path in old:
                    original = path.read_bytes()
                    shutil.copy2(path, archive / path.name)
                    if (archive / path.name).read_bytes() != original:
                        raise OSError(f"消融历史归档回读不一致：{path}")
        results: dict[str, Any] = {}
        if results_path.exists():
            results = json.loads(results_path.read_text(encoding="utf-8"))
            if not isinstance(results, dict):
                logger.error("已有消融结果身份不明；请保留原结果，使用 --force 明确重跑")
                return 2
            logger.info("已载入 %d 组历史结果，复用前逐组核对实验身份", len(results))

        runs = []
        for preset in presets:
            cfg = apply_config_overrides(base, preset.get("set", {}))
            cfg.setdefault("train", {})["checkpoint_dir"] = (
                f"artifacts/checkpoints/visibility_ablations/{preset['tag']}"
            )
            if args.epochs is not None:
                cfg["train"]["epochs"] = args.epochs
            runs.append((preset, cfg))

        # 关闭重建判据时，同一输入与评分配置下的判档不经过 AE。
        # 参数量、重建质量和训练耗时仍可用于模型比较；不自动开启重建判据。
        disabled = [p["tag"] for p, cfg in runs
                    if not bool(cfg.get("thresholds", {}).get("use_recon_z", False))]
        if disabled and not args.force_meaningless:
            logger.error(
                "预设 %s 关闭重建判据；同一输入和信息量配置下的判档与AE结构无关。"
                "如需比较参数量、重建质量或耗时，请显式加 --force-meaningless。", disabled,
            )
            return 3

        # 在启动任何训练前核验所有所选缓存，避免训练一半才发现另一行身份冲突。
        inputs_by_config: dict[str, dict[str, Any]] = {}
        prepared = []
        for preset, cfg in runs:
            data_key = _signature({"data": cfg.get("data", {}),
                                   "input_size": cfg.get("model", {}).get(
                                       "input_size", MODEL_DEFAULTS["input_size"])})
            if data_key not in inputs_by_config:
                inputs_by_config[data_key] = _prepare_inputs(cfg, root)
            inputs = inputs_by_config[data_key]
            identity = {"version": 1, "config": cfg, "data": inputs["data_identity"],
                        "evaluation_protocol": inputs["protocol"]}
            signature = _signature(identity)
            tag = preset["tag"]
            previous = results.get(tag)
            if tag in results and not (args.force or args.fresh) and (
                not isinstance(previous, dict)
                or previous.get("experiment_signature") != signature
                or previous.get("experiment_identity") != identity
                or "val_fpr" not in previous or "synth" not in previous
            ):
                logger.error(
                    "%s 的历史结果缺少可核验身份，或数据/配置/评估协议已变；"
                    "拒绝复用和混表。请保留原结果，使用 --force 明确重跑。", tag,
                )
                return 2
            prepared.append((preset, cfg, inputs, identity, signature))

        state_path = out_dir / "training_progress.json"
        suite_identity = {p["tag"]: sig for p, _, _, _, sig in prepared}
        if args.continue_training:
            if not state_path.is_file():
                logger.error("--continue 缺少原组进度，不能推测缺失 last 是未开始")
                return 2
            suite = json.loads(state_path.read_text(encoding="utf-8"))
            if suite.get("identity") != suite_identity:
                logger.error("原消融配置或数据身份已改变")
                return 2
        else:
            suite = {"identity": suite_identity,
                     "groups": {p["tag"]: {"status": "pending"} for p in presets}}

        def record_group(tag, status, directory):
            row = {"status": status}
            if status == "completed":
                row["last_sha256"] = _file_identity(directory / "last.pt")["sha256"]
                row["best_sha256"] = _file_identity(directory / "best.pt")["sha256"]
            suite["groups"][tag] = row
            temporary = state_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(suite, ensure_ascii=False, indent=2), encoding="utf-8")
            temporary.replace(state_path)

        for i, (preset, cfg, inputs, identity, signature) in enumerate(prepared, 1):
            controller.require_running()
            tag = preset["tag"]
            directory = root / cfg["train"]["checkpoint_dir"]
            row = suite["groups"].get(tag, {})
            group_continue = args.continue_training and row.get("status") != "pending"
            if args.continue_training:
                if row.get("status") not in {"pending", "running", "paused", "completed"}:
                    logger.error("%s 的原组进度缺失或无效", tag)
                    return 2
                if row.get("status") == "pending":
                    if any((directory / item).exists() for item in ("last.pt", "best.pt")):
                        logger.error("%s 标记未开始但已有检查点", tag)
                        return 2
                elif not all((directory / item).is_file() for item in ("last.pt", "best.pt")):
                    logger.error("%s 缺少已开始组的配对 last/best", tag)
                    return 2
                if row.get("status") == "completed":
                    if tag not in results or any(
                        _file_identity(directory / item)["sha256"] != row.get(key)
                        for item, key in (("last.pt", "last_sha256"), ("best.pt", "best_sha256"))
                    ):
                        logger.error("%s 已完成结果或配对检查点已改变", tag)
                        return 2
            if tag in results and not (args.force or args.fresh):
                if not args.continue_training and all(
                    (directory / item).is_file() for item in ("last.pt", "best.pt")
                ):
                    record_group(tag, "completed", directory)
                logger.info("[%d/%d] 跳过 %s（实验身份一致）", i, len(presets), tag)
                continue
            logger.info("[%d/%d] %s  覆盖: %s", i, len(presets), tag, preset.get("set", {}))
            if "refs" not in inputs:
                logger.info("解码参考图 %d 张，尺寸 %s", len(inputs["paths"]), inputs["size"])
                inputs["refs"] = load_refs(inputs["paths"], inputs["size"])
            refs = inputs["refs"]
            tr_idx, val_idx, ca_idx, _ = inputs["splits"]

            t0 = time.time()
            record_group(tag, "running", directory)
            directory.mkdir(parents=True, exist_ok=True)
            with FileLock(str((directory / "run.lock").resolve()), timeout=0, is_singleton=True):
                if args.fresh or args.force:
                    archive_visibility_checkpoints(directory)
                trainer = VisibilityTrainer(
                    cfg, project_root=root, resume=directory / "last.pt" if group_continue else "none"
                )
                history = trainer.fit(stop_controller=controller, continue_only=group_continue)
            if history.manual_stop or controller.requested:
                record_group(tag, "paused", directory)
                return STOP_EXIT_CODE
            train_s = time.time() - t0

            ckpt = root / cfg["train"]["checkpoint_dir"] / "best.pt"
            scorer = VisibilityScorer.from_checkpoint(
                ckpt,
                device=resolve_device(str(cfg.get("device", "auto"))),
                cfg=cfg.get("model", {}),
                scoring_cfg=cfg.get("scoring", {}),
            )
            if scorer.input_size != inputs["size"]:
                raise ValueError("有效检查点尺寸与已记录的消融评估尺寸不一致，拒绝发布结果")
            gate = VisibilityGate.from_config(cfg)

            rng = np.random.default_rng(0)
            metrics = evaluate_run(
                scorer, gate, refs, tr_idx, val_idx, ca_idx, rng
            )
            metrics["train_seconds"] = round(train_s, 1)
            metrics["epochs_run"] = len(history.train_loss)
            metrics["best_epoch"] = history.best_epoch
            metrics["overrides"] = preset.get("set", {})
            metrics["experiment_identity"] = identity
            metrics["experiment_signature"] = signature
            metrics["comparison_signature"] = _signature({
                "data": identity["data"], "evaluation_protocol": identity["evaluation_protocol"],
                "thresholds": vars(gate.thresholds), "scoring": scorer.scoring_cfg,
            })
            metrics["thresholds"] = vars(gate.thresholds)
            metrics["scoring"] = scorer.scoring_cfg
            controller.require_running()
            results[tag] = metrics

            results_path.write_text(
                json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            record_group(tag, "completed", directory)
            logger.info(
                "  %s 完成: 参数 %.2fM 过拟合比 %.3f 验证误报 %.1f%% 平均召回 %.1f%% (%.0f 秒)",
                tag, metrics["params_m"], metrics["overfit_ratio"],
                metrics["val_fpr"] * 100, metrics["mean_recall"] * 100, train_s,
            )

        controller.require_running()
        # 未选择或未核验的历史行不进入本轮对比报告。
        return write_report({p["tag"]: results[p["tag"]] for p in presets}, out_dir)


def write_report(results: dict[str, Any], out_dir: Path) -> int:
    if not results:
        logger.error("没有任何结果")
        return 2

    order = ["baseline"] + [k for k in results if k != "baseline"]
    order = [k for k in order if k in results]

    L: list[str] = [
        "# 能见度门控消融实验",
        "",
        "每组记录实际配置、过滤后参考图内容签名及分区/抽样协议；只有相同对比签名的组可直接比较。",
        "",
        "## 总表",
        "",
        "| 预设 | 参数(M) | 过拟合比 | 验证误报 | 平均召回 | " + " | ".join(KINDS) + " | 训练(s) |",
        "|------|--------|---------|---------|---------|" + "---|" * len(KINDS) + "--------|",
    ]
    for tag in order:
        m = results[tag]
        rec = " | ".join(f"{m['synth'][k]['recall']:.1%}" for k in KINDS)
        L.append(
            f"| {tag} | {m['params_m']:.2f} | {m['overfit_ratio']:.3f} | "
            f"{m['val_fpr']:.1%} | {m['mean_recall']:.1%} | {rec} | {m['train_seconds']:.0f} |"
        )

    L += ["", "## 各组配置", "", "| 预设 | 覆盖项 | use_recon_z | 实验签名 | 对比签名 |",
          "|------|--------|-------------|----------|----------|"]
    for tag in order:
        row = results[tag]
        recon_enabled = bool(row.get("thresholds", {}).get("use_recon_z", False))
        L.append(f"| {tag} | `{row.get('overrides') or '（基线）'}` | {recon_enabled} | "
                 f"`{row.get('experiment_signature', '未记录')}` | "
                 f"`{row.get('comparison_signature', '未记录')}` |")
    for tag in order:
        row = results[tag]
        identity = row.get("experiment_identity", {})
        L += ["", f"### {tag} 实际配置与协议", "",
              f"- 配置：`{identity.get('config', {})}`",
              f"- 数据身份：`{identity.get('data', {})}`",
              f"- 评估协议：`{identity.get('evaluation_protocol', {})}`",
              f"- 有效门控：`{row.get('thresholds', {})}`",
              f"- 有效评分配置：`{row.get('scoring', {})}`", ""]

    base = results.get("baseline")
    if base:
        L += ["", "## 相对基线的增益", "",
              "| 预设 | 召回变化 | 参数量变化 | 过拟合比变化 |", "|------|---------|-----------|-------------|"]
        for tag in order:
            if tag == "baseline":
                continue
            m = results[tag]
            if (not m.get("comparison_signature")
                    or m["comparison_signature"] != base.get("comparison_signature")
                    or m.get("thresholds") != base.get("thresholds")
                    or m.get("scoring") != base.get("scoring")):
                L.append(f"| {tag} | 数据或协议不同，不作直接比较 | — | — |")
                continue
            L.append(
                f"| {tag} | {(m['mean_recall'] - base['mean_recall']) * 100:+.1f}pp | "
                f"{m['params_m'] - base['params_m']:+.2f}M | "
                f"{m['overfit_ratio'] - base['overfit_ratio']:+.3f} |"
            )

    L += ["", "## 逐退化召回曲线", ""]
    for tag in order:
        L += [f"### {tag}", "", "| 退化 | " + " | ".join(f"s={s}" for s in SEVERITIES) + " |",
              "|------|" + "---|" * len(SEVERITIES)]
        for k in KINDS:
            row = " | ".join(f"{c['blind_rate']:.0%}" for c in results[tag]["synth"][k]["curve"])
            L.append(f"| {k} | {row} |")
        L.append("")

    recalls = {round(results[t]["mean_recall"], 6) for t in order}
    fps = {round(results[t]["val_fpr"], 6) for t in order}
    degenerate = len(recalls) == 1 and len(fps) == 1 and len(order) > 1
    if degenerate:
        L += [
            "## 判档指标相同", "",
            f"全部 {len(order)} 组的平均召回与验证误报在保留6位小数后相同。",
            "该现象本身不能证明指标与模型无关；需核对各组实际 use_recon_z、输入和评分配置。", "",
        ]
    L += [
        "## 读表须知", "",
        "- 过拟合比 = 验证重建误差 / 训练重建误差，只是本次抽样下的重建差距诊断，",
        "  不能单凭接近1或整组划分断言真实场景泛化或不存在泄漏。",
        "- 参数量、重建误差与训练耗时都与模型有关。use_recon_z=false 时，",
        "  相同输入与信息量评分配置下的门控判档不依赖AE权重；启用时还读取重建z分数。",
        "- 验证判BLIND比例和合成退化召回用于消融比较；本脚本不评估test子集。",
        "- 平均召回是所记录合成退化在最大强度下判为BLIND的比例均值，",
        "  没有真实逐帧可见性真值，也不构成真实召回的上界。",
        "- 抽样规模、分区、评分与阈值均影响指标；请结合各组配置和实验签名读表。", "",
    ]

    (out_dir / "ablations.md").write_text("\n".join(L), encoding="utf-8")
    logger.info("对比报告已写出: %s", out_dir / "ablations.md")

    print("\n" + "=" * 78)
    print("消融实验完成")
    print("=" * 78)
    print(f"  {'预设':<20}{'参数M':>8}{'过拟合比':>10}{'验证误报':>8}{'平均召回':>10}")
    print("  " + "-" * 74)
    for tag in order:
        m = results[tag]
        print(
            f"  {tag:<20}{m['params_m']:>8.2f}{m['overfit_ratio']:>10.3f}"
            f"{m['val_fpr'] * 100:>7.1f}%{m['mean_recall'] * 100:>9.1f}%"
        )
    print(f"\n  报告: {out_dir / 'ablations.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
