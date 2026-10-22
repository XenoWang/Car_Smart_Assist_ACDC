"""检查能见度门控：召回率、误报率、单调性。"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from car_smart_assist.perception.visibility import (  # noqa: E402
    VisibilityGate,
    VisibilityScorer,
    degrade,
    resolve_device,
)
from car_smart_assist.perception.visibility.dataset import (  # noqa: E402
    list_adverse_images,
    list_ref_images,
    read_rgb_image,
)

logger = logging.getLogger(__name__)



# 数据读取



def load_images(paths: list[Path], size: tuple[int, int]) -> list[np.ndarray]:
    """解码失败直接报错，避免图片与 ref_paths 的索引错位。"""
    return [read_rgb_image(path, size) for path in paths]


def summarize(verdicts) -> dict[str, Any]:
    """把一组判定汇总成各档占比。"""
    n = max(len(verdicts), 1)
    counts = defaultdict(int)
    for v in verdicts:
        counts[v.level.value] += 1
    return {
        "n": len(verdicts),
        "visible": counts["visible"] / n,
        "degraded": counts["degraded"] / n,
        "blind": counts["blind"] / n,
        "blocked": counts["blind"] / n,  # BLIND 表示阻断感知
    }



# 各项实验



def eval_clean_control(
    scorer: VisibilityScorer, gate: VisibilityGate, refs: list[np.ndarray], n: int, seed: int
) -> dict[str, Any]:
    """误报率：清晰的正常天气图有多少被拦下，期望接近 0。"""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(refs), size=min(n, len(refs)), replace=False)
    scores = scorer.score_arrays([refs[i] for i in idx])
    verdicts = gate.judge_many(scores)
    s = summarize(verdicts)
    logger.info(
        "清晰图（对照）: visible=%.1f%% degraded=%.1f%% blind=%.1f%%",
        s["visible"] * 100, s["degraded"] * 100, s["blind"] * 100,
    )
    return {"false_positive_rate": s["blind"], "degraded_rate": s["degraded"], **s}


def eval_fit_quality(
    scorer: VisibilityScorer,
    refs: list[np.ndarray],
    ref_paths: list[Path],
    dcfg: dict[str, Any],
    gate: VisibilityGate,
    n: int,
) -> dict[str, Any]:
    """检查过拟合 / 欠拟合，并在**测试集**上给出最终误报率。

    四个划分的角色要分清：
        train  模型在这些数据上拟合过 —— 指标最好，但不代表泛化
        val    参与早停与权重选择
        calib  只用来算零校准统计
        test   本次仅用于报告；是否用于过往决策需核对实验记录

    ⚠️ 测重建误差必须直接用**原始图像**，不能拿训练日志里的 train_loss 对比。
    训练损失算在开了增强的数据上，验证/测试损失没有增强；
    所以这里分别在原始图像上抽样，统一重算重建误差。

    采用与 trainer 相同的划分入口；历史数据身份及实际分区仍需核对。
    """
    from car_smart_assist.perception.visibility.dataset import (
        sequence_of,
        split_ref_indices,
    )

    if dcfg.get("split_mode", "fixed") == "random":
        raise ValueError("评估不能重建训练时的随机分区；请使用 data.split_mode=fixed 的匹配检查点")

    # 划分参数必须和 trainer 完全一致，否则各集合会错位
    groups = (
        [sequence_of(p) for p in ref_paths]
        if dcfg.get("split_by_sequence", True)
        else None
    )
    tr_idx, val_idx, ca_idx, te_idx = split_ref_indices(
        len(ref_paths),
        train_ratio=float(dcfg.get("train_ratio", 0.80)),
        val_ratio=float(dcfg.get("val_ratio", 0.05)),
        calib_ratio=float(dcfg.get("calib_ratio", 0.05)),
        test_ratio=float(dcfg.get("test_ratio", 0.10)),
        seed=int(dcfg.get("split_seed", 42)),
        mode=str(dcfg.get("split_mode", "fixed")),
        groups=groups,
    )

    rng = np.random.default_rng(int(dcfg.get("split_seed", 42)))

    def probe(idx: np.ndarray):
        take = rng.choice(idx, size=min(n, len(idx)), replace=False)
        s = scorer.score_arrays([refs[i] for i in take], [str(ref_paths[i]) for i in take])
        return (
            float(np.mean([x.recon_mean for x in s])),
            float(np.median([x.information for x in s])),
            s,
        )

    tr_mean, tr_info, _ = probe(tr_idx)
    val_mean, val_info, _ = probe(val_idx)
    ca_mean, ca_info, _ = probe(ca_idx)
    te_mean, te_info, te_scores = probe(te_idx)

    val_ratio = val_mean / tr_mean if tr_mean > 0 else float("nan")
    test_ratio = te_mean / tr_mean if tr_mean > 0 else float("nan")
    if test_ratio < 1.05:
        verdict = "测试集与训练集差距较小"
    elif test_ratio < 1.30:
        verdict = "测试集存在轻度泛化差距"
    else:
        verdict = "⚠️ 测试集泛化差距明显，需检查模型容量与数据分布"

    # 测试集误报率 —— 只有这个数字没被任何决策影响过
    te_verdicts = gate.judge_many(te_scores)
    te_fpr = summarize(te_verdicts)["blind"]

    logger.info(
        "拟合质量: train=%.6f val=%.6f calib=%.6f test=%.6f "
        "比值(test/train)=%.3f -> %s",
        tr_mean, val_mean, ca_mean, te_mean, test_ratio, verdict,
    )
    logger.info(
        "  信息量中位数 train=%.3f val=%.3f calib=%.3f test=%.3f",
        tr_info, val_info, ca_info, te_info,
    )
    logger.info(
        "  **测试集误报率 = %.2f%%**（本次抽样 %d 张，未参与任何决策）",
        te_fpr * 100, min(n, len(te_idx)),
    )

    return {
        "train_recon_mean": tr_mean,
        "val_recon_mean": val_mean,
        "calib_recon_mean": ca_mean,
        "test_recon_mean": te_mean,
        "val_train_ratio": val_ratio,
        "test_train_ratio": test_ratio,
        "verdict": verdict,
        "train_information_median": tr_info,
        "val_information_median": val_info,
        "calib_information_median": ca_info,
        "test_information_median": te_info,
        "split_sizes": {
            "train": len(tr_idx), "val": len(val_idx),
            "calib": len(ca_idx), "test": len(te_idx),
        },
        "test_false_positive_rate": te_fpr,
        "n_each": min(n, len(tr_idx), len(val_idx), len(ca_idx), len(te_idx)),
    }


def eval_real_adverse(
    scorer: VisibilityScorer,
    gate: VisibilityGate,
    acdc_root: Path,
    size: tuple[int, int],
    n: int,
    seed: int,
    *,
    cleaning_manifest: str | Path | None = None,
    project_root: str | Path = ".",
) -> dict[str, Any]:
    """汇总真实恶劣天气图的判档分布，不将目录标签当成可用性真值。"""
    rng = np.random.default_rng(seed)
    result: dict[str, Any] = {}
    for cond in ("fog", "night", "rain", "snow"):
        paths = list_adverse_images(
            acdc_root, [cond], cleaning_manifest=cleaning_manifest, project_root=project_root
        )
        if not paths:
            continue
        idx = rng.choice(len(paths), size=min(n, len(paths)), replace=False)
        imgs = load_images([paths[i] for i in idx], size)
        scores = scorer.score_arrays(imgs)
        verdicts = gate.judge_many(scores)
        s = summarize(verdicts)
        info = np.array([sc.information for sc in scores])
        z = np.array([sc.recon_z for sc in scores])
        result[cond] = {
            **s,
            "information_min": float(info.min()),
            "information_p05": float(np.percentile(info, 5)),
            "information_median": float(np.median(info)),
            "recon_z_median": float(np.nanmedian(z)) if np.isfinite(z).any() else None,
        }
        logger.info(
            "真实 %-5s (%d 张): visible=%.1f%% degraded=%.1f%% blind=%.1f%%  "
            "info[最小 %.3f / 中位 %.3f]",
            cond, len(scores), s["visible"] * 100, s["degraded"] * 100, s["blind"] * 100,
            info.min(), np.median(info),
        )
    return result


def eval_synthetic(
    scorer: VisibilityScorer,
    gate: VisibilityGate,
    refs: list[np.ndarray],
    severities: list[float],
    n: int,
    seed: int,
) -> dict[str, Any]:
    """合成退化：度量召回率和单调性。"""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(refs), size=min(n, len(refs)), replace=False)
    base = [refs[i] for i in idx]

    out: dict[str, Any] = {}
    for kind in ("fog", "darkness", "occlusion", "blur"):
        curve: list[dict[str, Any]] = []
        for sev in severities:
            imgs = [degrade(b, kind, sev, seed=seed + i)[0] for i, b in enumerate(base)]
            scores = scorer.score_arrays(imgs)
            verdicts = gate.judge_many(scores)
            s = summarize(verdicts)
            info = np.array([sc.information for sc in scores])
            zs = np.array([sc.recon_z for sc in scores], dtype=np.float64)
            curve.append(
                {
                    "severity": sev,
                    "blind_rate": s["blind"],
                    "blocked_rate": s["blocked"],
                    "information_mean": float(info.mean()),
                    # 同时记录 recon_z：退化越重它反而越低时，
                    # 说明重建误差与退化反相关（见 gate.py 里 use_recon_z 的说明）
                    "recon_z_median": float(np.nanmedian(zs)) if np.isfinite(zs).any() else None,
                }
            )
            logger.info(
                "  合成 %-9s severity=%.2f -> blind=%.1f%%  info均值=%.3f  recon_z中位=%.2f",
                kind, sev, s["blind"] * 100, info.mean(),
                np.nanmedian(zs) if np.isfinite(zs).any() else float("nan"),
            )

        # 单调性：检查information是否随合成severity单调不增；不等同于真实能见度真值
        infos = [c["information_mean"] for c in curve]
        monotonic = all(
            infos[i] >= infos[i + 1] - 1e-6 for i in range(len(infos) - 1)
        )
        high = curve[-1]
        out[kind] = {
            "curve": curve,
            "recall_at_max_severity": high["blind_rate"],
            "information_monotonic_decreasing": monotonic,
        }
    return out



# 报告



def write_report(report: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "evaluation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    thresholds = report.get("thresholds", {})
    recon_enabled = bool(thresholds.get("use_recon_z", False))
    L: list[str] = [
        "# 能见度门控评估报告", "",
        f"- 配置来源：`{report.get('config_path', '未记录')}`",
        f"- 检查点：`{report.get('checkpoint', '未记录')}`",
        f"- 有效输入尺寸 (H, W)：`{report.get('input_size', '未记录')}`",
        f"- 实际门控参数：`{thresholds}`",
        f"- 重建判据 use_recon_z：`{recon_enabled}`",
        f"- 数据配置：`{report.get('config', {}).get('data', {})}`",
        f"- 信息量评分配置：`{report.get('scoring', {})}`",
        f"- 评估协议：`{report.get('evaluation_protocol', {})}`", "",
    ]

    f = report["fit_quality"]
    sz = f["split_sizes"]
    L += [
        "## 1. 拟合质量（过拟合 / 欠拟合检查）",
        "",
        "| 集合 | 张数 | 重建误差均值 | 角色 |",
        "|------|------|-------------|------|",
        f"| 训练集 | {sz['train']} | {f['train_recon_mean']:.6f} | 模型拟合过，指标最乐观 |",
        f"| 验证集 | {sz['val']} | {f['val_recon_mean']:.6f} | 用于选择权重与早停 |",
        f"| 校准集 | {sz['calib']} | {f['calib_recon_mean']:.6f} | 只用于最终零校准 |",
        f"| **测试集** | {sz['test']} | **{f['test_recon_mean']:.6f}** | 本次仅报告，历史使用需核对 |",
        "",
        f"- 验证/训练误差比：**{f['val_train_ratio']:.3f}**（验证集用于选择权重）",
        f"- 测试/训练误差比：**{f['test_train_ratio']:.3f}**（本次仅作重建诊断，不用于选权重）",
        f"- 结论：**{f['verdict']}**",
        f"- 信息量分数中位数：训练 {f['train_information_median']:.3f} / "
        f"验证 {f['val_information_median']:.3f} / 校准 {f['calib_information_median']:.3f} / "
        f"测试 {f['test_information_median']:.3f}",
        f"- 每组抽样：{f['n_each']}",
        "",
        f"### 测试集误报率：**{f['test_false_positive_rate']:.2%}**",
        "",
        "> 本脚本按记录的数据配置重建四分划分；测试子集在本次评估中仅用于报告。",
        "> 是否曾用于其他人工决策，以及检查点训练时是否采用相同数据，需结合实验记录核对。",
        "",
        "> ⚠️ 这里直接测**原始图像**，不能拿训练日志里的 train_loss 对比：",
        "> 训练损失在开了增强的数据上算，验证损失没有增强，两边口径不一致会**缩小**差距。",
        "> train / val / test 使用同一原始图像重建口径；训练日志损失含增强与去噪，不与其直接比较。",
        "",
    ]

    c = report["clean_control"]
    L += [
        "## 2. 误报率（清晰图，对照组）",
        "",
        f"- 样本数：{c['n']}",
        f"- **被判 BLIND 的比例：{c['false_positive_rate']:.2%}**（越低越好，理想为 0）",
        f"- 被判 DEGRADED 的比例：{c['degraded_rate']:.2%}",
        "",
        "> 这里将参考图作为正常对照统计；没有逐帧可用性真值，不能据此断定每次阻断都错误。",
        "",
    ]

    L += [
        "## 3. 真实恶劣天气（ACDC）",
        "",
        "| 条件 | 张数 | visible | degraded | BLIND | info 最小 | info 中位 | recon_z 中位 |",
        "|------|------|---------|----------|-------|-----------|-----------|--------------|",
    ]
    for cond, s in report["real_adverse"].items():
        rz = s.get("recon_z_median")
        rz_s = f"{rz:.2f}" if rz is not None else "n/a"
        L.append(
            f"| {cond} | {s['n']} | {s['visible']:.1%} | {s['degraded']:.1%} | "
            f"{s['blind']:.1%} | {s['information_min']:.3f} | {s['information_median']:.3f} | {rz_s} |"
        )
    L += [
        "",
        "> ACDC 条件标签不提供逐帧 BLIND 真值；此处只报告判档分布，",
        "> 不能单凭阻断比例判断误报或可用性，需结合逐帧复核。",
        "",
        "> recon_z 的方向与判别力需要独立验证；",
        f"> 本次配置的重建判据为 use_recon_z={recon_enabled}。",
        "",
    ]

    L += ["## 4. 合成退化（召回率与单调性）", ""]
    for kind, d in report["synthetic"].items():
        mono = "✅ 单调递减" if d["information_monotonic_decreasing"] else "❌ **非单调，需排查**"
        L += [
            f"### {kind}",
            "",
            f"- 最大强度下的召回率（判为 BLIND）：**{d['recall_at_max_severity']:.1%}**",
            f"- 信息量随强度变化：{mono}",
            "",
            "| severity | BLIND 比例 | 信息量均值 | recon_z 中位 |",
            "|----------|-----------|-----------|--------------|",
        ]
        for row in d["curve"]:
            rz = row.get("recon_z_median")
            rz_s = f"{rz:.2f}" if rz is not None else "n/a"
            L.append(
                f"| {row['severity']:.2f} | {row['blind_rate']:.1%} | "
                f"{row['information_mean']:.3f} | {rz_s} |"
            )
        L.append("")

    L += [
        "## 5. 局限（必读）",
        "",
        "- 合成退化与真实恶劣天气并不等价：真实的雾浓度非均匀、",
        "  真实夜间的传感器噪声模型也不同于 gamma 拉伸。",
        "  第 4 节只统计本次合成协议下的 BLIND 比例，不能视为真实召回的上界或替代真实验证。",
        "- 本报告只能回答「在人为构造的极端退化下门控是否报警」，",
        "  无法回答「在真实部署的雨雾夜里是否恰好报警」。",
        "  后者需要真实带标注数据，超出本项目的数据条件。",
        ("- 本次启用重建判据，门控同时读取信息量与重建 z 分数；其真实判别力仍需独立验证。"
         if recon_enabled else
         "- 本次关闭重建判据，门控判档只读取确定性信息量；AE 仍用于本报告的重建诊断。"),
        "",
    ]

    (out_dir / "evaluation.md").write_text("\n".join(L), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="能见度门控评估")
    ap.add_argument("--config", default="configs/model/visibility.yaml")
    ap.add_argument("--checkpoint", default=None, help="默认取配置中的 checkpoint_dir/best.pt")
    ap.add_argument("--samples", type=int, default=None, help="每个条件下抽多少张")
    ap.add_argument("--severities", type=float, nargs="*", default=[0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--device", default=None)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    root = Path(__file__).resolve().parents[1]
    cfg_path = root / args.config
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg = raw.get("visibility", raw)
    dcfg, rcfg = cfg.get("data", {}), cfg.get("report", {})
    if dcfg.get("split_mode", "fixed") == "random":
        logger.error("评估不能重建训练时的随机分区；请使用 data.split_mode=fixed 的匹配检查点")
        return 2

    acdc_root = root / dcfg.get("acdc_root", "data/raw/acdc")

    ckpt = Path(args.checkpoint) if args.checkpoint else (
        root / cfg.get("train", {}).get("checkpoint_dir", "artifacts/checkpoints/visibility") / "best.pt"
    )
    if not ckpt.exists():
        logger.error("checkpoint 不存在: %s\n请先运行 scripts/train_visibility.py", ckpt)
        return 2

    device = args.device or str(cfg.get("device", "auto"))
    scorer = VisibilityScorer.from_checkpoint(
        ckpt,
        device=resolve_device(device),
        cfg=cfg.get("model", {}),
        scoring_cfg=cfg.get("scoring", {}),
    )
    # 检查点决定模型有效尺寸，参考图与真实恶劣图必须沿用同一解码口径。
    size = scorer.input_size
    gate = VisibilityGate.from_config(cfg)
    if scorer.calibration is None:
        logger.warning(
            "缺少AE重建校准；信息量仍独立判档，极低时直接BLIND。"
            "只有显式启用重建判据且require_calibration=true时，缺校准才会报评分故障。"
        )
    else:
        logger.info("零校准: mean=%.6f std=%.6f (n=%d)",
                    scorer.calibration.mean, scorer.calibration.std, scorer.calibration.n)

    n = int(args.samples or rcfg.get("sample_per_condition", 200))
    cleaning_root = root / dcfg.get("project_root", ".")
    ref_paths = list_ref_images(
        acdc_root, cleaning_manifest=dcfg.get("cleaning_manifest"), project_root=cleaning_root
    )
    refs = load_images(ref_paths, size)
    if not refs:
        logger.error("未找到正常天气参考图，请确认 ACDC 已解压")
        return 2
    logger.info("加载正常天气参考图 %d 张", len(refs))

    report: dict[str, Any] = {
        "checkpoint": str(ckpt),
        "config_path": str(cfg_path),
        "config": cfg,
        "input_size": list(size),
        "scoring": scorer.scoring_cfg,
        "evaluation_protocol": {
            "sample_per_condition": n,
            "sample_seed": 0,
            "severities": list(args.severities),
            "synthetic_kinds": ["fog", "darkness", "occlusion", "blur"],
        },
        "calibration": scorer.calibration.to_dict() if scorer.calibration else None,
        "thresholds": vars(gate.thresholds),
    }

    # 重建质量与判档指标分别报告；实际使用的判据由 thresholds 记录。
    report["fit_quality"] = eval_fit_quality(scorer, refs, ref_paths, dcfg, gate, n)
    report["clean_control"] = eval_clean_control(scorer, gate, refs, n, 0)
    report["real_adverse"] = eval_real_adverse(
        scorer, gate, acdc_root, size, n, 0,
        cleaning_manifest=dcfg.get("cleaning_manifest"), project_root=cleaning_root,
    )
    logger.info("合成退化实验（%d 张基准图 × %d 种退化 × %d 档强度）...",
                min(n, len(refs)), 4, len(args.severities))
    report["synthetic"] = eval_synthetic(scorer, gate, refs, list(args.severities), n, 0)

    out_dir = root / rcfg.get("output_dir", "artifacts/reports/visibility")
    write_report(report, out_dir)

    print("\n" + "=" * 62)
    print("评估完成")
    print("=" * 62)
    fq = report["fit_quality"]
    print(
        f"  拟合诊断  测试/训练误差比 : {fq['test_train_ratio']:.3f}  -> {fq['verdict']}"
    )
    print(f"  误报率（清晰图被判 BLIND） : {report['clean_control']['false_positive_rate']:.2%}")
    for cond, s in report["real_adverse"].items():
        print(f"  真实 {cond:6s} BLIND 比例      : {s['blind']:.2%}")
    print("  合成退化召回率（最大强度）:")
    for kind, d in report["synthetic"].items():
        flag = "" if d["information_monotonic_decreasing"] else "  ⚠️ 非单调"
        print(f"    {kind:10s} {d['recall_at_max_severity']:6.1%}{flag}")
    print(f"\n  报告: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
