"""能见度门控的验证：召回率、误报率、单调性。

职责:
    - 用合成退化度量门控对「确定看不见」的召回率（没有真值标签时唯一能定量的手段）
    - 用清晰图度量误报率（应当接近 0）
    - 用 ACDC 真实恶劣天气图看分数分布（**不应**大量落入 BLIND —— 那些是「难但可用」）
    - 验证分数随退化程度单调上升
    - 产出报告到 artifacts/reports/visibility/

为什么必须做这一步:
    一个不知道召回率的安全门控等于没有 —— 无法回答「真的看不见时它会不会报警」。
    合成退化的强度是我们自己设的，所以「这张是看不见的」是我们确知的真值，
    于是召回率可以被定量测量。

    单调性同样重要：如果分数在某个退化区间不升反降，
    说明模型在该区间行为反常，这比单纯的低召回更值得查。

设计:
    **退化施加在正常天气参考图上**，而不是恶劣天气图上。
    这样「唯一的变量就是退化本身」—— 排除了场景差异的干扰。
    若拿雨夜图去加雾，分数升高时无法区分是雾还是原本的雨造成的。

局限（报告里会写明）:
    合成退化与真实恶劣天气不完全一致 —— 真实的雾浓度非均匀、
    真实夜间噪声模型不同于 gamma 拉伸。因此这里的召回率是上界意义上的估计，
    不能替代真实场景验证。
"""

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
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from car_smart_assist.config.visibility import MODEL_DEFAULTS  # noqa: E402
from car_smart_assist.perception.visibility import (  # noqa: E402
    VisibilityGate,
    VisibilityScorer,
    degrade,
    resolve_device,
)
from car_smart_assist.perception.visibility.dataset import (  # noqa: E402
    list_adverse_images,
    list_ref_images,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 数据读取
# ---------------------------------------------------------------------------


def load_images(paths: list[Path], size: tuple[int, int]) -> list[np.ndarray]:
    """解码并缩放到指定尺寸，返回 uint8 数组列表。"""
    h, w = size
    out: list[np.ndarray] = []
    for p in paths:
        try:
            with Image.open(p) as im:
                out.append(np.asarray(im.convert("RGB").resize((w, h), Image.BILINEAR)))
        except Exception as exc:  # noqa: BLE001
            logger.warning("跳过无法读取的图 %s: %s", p, exc)
    return out


def summarize(verdicts) -> dict[str, Any]:
    """把一组判定压成各档占比。"""
    n = max(len(verdicts), 1)
    counts = defaultdict(int)
    for v in verdicts:
        counts[v.level.value] += 1
    return {
        "n": len(verdicts),
        "visible": counts["visible"] / n,
        "degraded": counts["degraded"] / n,
        "blind": counts["blind"] / n,
        "blocked": counts["blind"] / n,  # BLIND 即阻断感知
    }


# ---------------------------------------------------------------------------
# 各项实验
# ---------------------------------------------------------------------------


def eval_clean_control(
    scorer: VisibilityScorer, gate: VisibilityGate, refs: list[np.ndarray], n: int, seed: int
) -> dict[str, Any]:
    """误报率：清晰的正常天气图有多少被拦下。期望接近 0。"""
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
    """过拟合 / 欠拟合检查，并在**测试集**上给出最终误报率。

    四划分的角色必须分清：
        train  模型拟合过      —— 指标最好，但不代表泛化
        val    参与早停与权重选择
        calib  只用于计算零校准统计
        test   **不参与任何决定** —— 这里报出的数字才是可引用的

    ⚠️ 必须直接测**原始图像**的重建误差，不能拿训练日志里的 train_loss 对比。
    训练损失是在开了增强的数据上算的，而验证/测试损失没有增强；
    所以这里分别在原始图像上抽样，统一重算重建误差。

    使用与 trainer 完全相同的划分函数，保证切分一致。
    """
    from car_smart_assist.perception.visibility.dataset import (
        sequence_of,
        split_ref_indices,
    )

    # 必须与 trainer 用完全相同的划分参数，否则各集合会错位
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

    # 测试集上的误报率 —— 这是唯一没被任何决策污染过的数字
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
) -> dict[str, Any]:
    """真实恶劣天气图：这些是「难但可用」，不应大量落入 BLIND。"""
    rng = np.random.default_rng(seed)
    result: dict[str, Any] = {}
    for cond in ("fog", "night", "rain", "snow"):
        paths = list_adverse_images(acdc_root, [cond])
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
    """合成退化：度量召回率与单调性。"""
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
                    # 一并记录 recon_z：退化越重它若反而越低，
                    # 说明重建误差与退化反相关（见 gate.py 的 use_recon_z 说明）
                    "recon_z_median": float(np.nanmedian(zs)) if np.isfinite(zs).any() else None,
                }
            )
            logger.info(
                "  合成 %-9s severity=%.2f -> blind=%.1f%%  info均值=%.3f  recon_z中位=%.2f",
                kind, sev, s["blind"] * 100, info.mean(),
                np.nanmedian(zs) if np.isfinite(zs).any() else float("nan"),
            )

        # 单调性：information 应随 severity 单调下降（等价于能见度单调变差）
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


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------


def write_report(report: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "evaluation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    L: list[str] = ["# 能见度门控评估报告", ""]

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
        f"| **测试集** | {sz['test']} | **{f['test_recon_mean']:.6f}** | **不参与任何决定，可引用** |",
        "",
        f"- 验证/训练误差比：**{f['val_train_ratio']:.3f}**（验证集用于选择权重）",
        f"- 测试/训练误差比：**{f['test_train_ratio']:.3f}**（只作最终泛化诊断，不用于选权重）",
        f"- 结论：**{f['verdict']}**",
        f"- 信息量分数中位数：训练 {f['train_information_median']:.3f} / "
        f"验证 {f['val_information_median']:.3f} / 校准 {f['calib_information_median']:.3f} / "
        f"测试 {f['test_information_median']:.3f}",
        f"- 每组抽样：{f['n_each']}",
        "",
        f"### 测试集误报率：**{f['test_false_positive_rate']:.2%}**",
        "",
        "> 这是整个报告里**唯一没有被任何决策污染过**的数字 ——",
        "> 测试集从不参与训练、早停、阈值标定，所以它是可引用的泛化估计。",
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
        "> 这些是清晰的正常天气图，门控不该拦下任何一张。",
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
        "> ⚠️ ACDC 里**没有**「看不清」的帧 —— 否则它不成其为数据集。",
        "> 它的最暗帧是「很暗但可用」。因此这里的 BLIND 比例应当很低；",
        "> 如果偏高，说明阈值定得太激进，会把整个夜路子集废掉。",
        "",
        "> **recon_z 一列请留意方向**：若退化越重它反而越低（甚至为负），",
        "> 说明重建误差与退化反相关，该信号不具备判别力，已在 gate 中停用。",
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
        "  因此第 4 节的召回率是**上界意义上的估计**，不能替代真实场景验证。",
        "- 本报告只能回答「在人为构造的极端退化下门控是否报警」，",
        "  无法回答「在真实部署的雨雾夜里是否恰好报警」。",
        "  后者需要真实带标注数据，超出本项目的数据条件。",
        "- 判定目前**只依赖确定性信息量特征**，重建误差未参与决策（原因见第 3 节）。",
        "  这意味着自编码器在当前任务上尚未贡献判别力，其价值有待重新评估。",
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

    size = tuple(cfg.get("model", {}).get("input_size", MODEL_DEFAULTS["input_size"]))
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
    gate = VisibilityGate.from_config(cfg)
    if scorer.calibration is None:
        logger.warning("无零校准统计，BLIND 判定将被降级 —— 建议重新训练")
    else:
        logger.info("零校准: mean=%.6f std=%.6f (n=%d)",
                    scorer.calibration.mean, scorer.calibration.std, scorer.calibration.n)

    n = int(args.samples or rcfg.get("sample_per_condition", 200))
    refs = load_images(list_ref_images(acdc_root), size)
    if not refs:
        logger.error("未找到正常天气参考图，请确认 ACDC 已解压")
        return 2
    logger.info("加载正常天气参考图 %d 张", len(refs))

    report: dict[str, Any] = {
        "checkpoint": str(ckpt),
        "calibration": scorer.calibration.to_dict() if scorer.calibration else None,
        "thresholds": vars(gate.thresholds),
    }

    # 先做过拟合检查 —— 若模型本身没训好，后面的召回率数字都没有意义
    ref_paths = list_ref_images(acdc_root)
    report["fit_quality"] = eval_fit_quality(scorer, refs, ref_paths, dcfg, gate, n)
    report["clean_control"] = eval_clean_control(scorer, gate, refs, n, 0)
    report["real_adverse"] = eval_real_adverse(scorer, gate, acdc_root, size, n, 0)
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
