"""能见度门控的消融实验驱动器。

职责:
    - 读 configs/model/visibility.yaml 的 ablations 段，逐个预设训练 + 评估
    - 保证所有预设用**完全相同的划分**与**相同的评估口径**，否则结果不可比
    - 产出对比表（json + markdown）到 artifacts/reports/visibility/

为什么要写成驱动器而不是手动跑七次:
    1. **划分必须一致**。手改配置容易连划分种子一起动，那结果就没法比了。
    2. **checkpoint 目录必须隔离**。默认目录共用一个，第二次训练会捡起第一次的
       权重续训 —— 那测的是「继续训练」而不是「换个结构重训」，结论完全错。
    3. **参考图只解码一次**。4006 张 1080p 解码要一分多钟，七组就是七分钟白费。

设计:
    - 每组独立 checkpoint 目录，强制 resume='none'
    - 结果落盘后**跳过已完成**的组，长任务可以中断重入
    - 评估只算对比需要的关键指标，不复用 evaluate_visibility.py 的全量流程
      （那个流程会跑完整的 4×4 合成退化扫描，七组太慢）

用法:
    python scripts/run_visibility_ablations.py                # 全部
    python scripts/run_visibility_ablations.py --only baseline plain_encoder
    python scripts/run_visibility_ablations.py --list         # 只列出预设
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

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
)

logger = logging.getLogger(__name__)

# 评估用规模。七组共用同一批图，解码一次即可。
N_PER_SPLIT = 60          # train / val / calib / test 各抽多少张
N_SYNTH = 60              # 合成退化实验的基准图数量
SEVERITIES = (0.5, 0.75, 1.0)
KINDS = ("fog", "darkness", "occlusion", "blur")
MAX_SEV = SEVERITIES[-1]


def load_refs(paths, size) -> list[np.ndarray]:
    return [read_rgb_image(path, size) for path in paths]


# ---------------------------------------------------------------------------
# 评估（对比用的精简指标）
# ---------------------------------------------------------------------------


def evaluate_run(
    scorer: VisibilityScorer,
    gate: VisibilityGate,
    refs: list[np.ndarray],
    tr_idx: np.ndarray,
    val_idx: np.ndarray,
    ca_idx: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """只在训练集与验证集上比较，测试集保留给最终评估。"""

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


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


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

    out_dir = root / "artifacts/reports/visibility/ablations"
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.json"
    results: dict[str, Any] = {}
    if results_path.exists() and not args.force:
        results = json.loads(results_path.read_text(encoding="utf-8"))
        if any("val_fpr" not in row or "synth" not in row for row in results.values()):
            logger.warning("已有消融结果使用旧评估协议，丢弃并按验证集协议重跑")
            results = {}
        logger.info("已载入 %d 组历史结果（用 --force 可全部重跑）", len(results))

    # --- 有效性闸门（踩过一次坑，代价是 56 分钟 GPU 时间）---
    #
    # 门控判定在 use_recon_z=False 时**只依赖确定性信息量特征**，
    # 而信息量特征由 compute_information_features(图像) 直接算出，不经过模型。
    # 于是「换编码器结构」对召回率、误报率的影响**在数学上恒等于零** ——
    # 无论怎么消融，所有组的结果都会逐位相同。
    #
    # 这不是「架构不重要」，而是「门控根本没用到架构」。
    # 真要通过消融比较架构，必须让被比较的指标依赖模型：
    #   · 重新启用 use_recon_z（并先证明 recon_z 有判别力），或
    #   · 比较重建质量本身（过拟合比、逐类重建误差）
    # 在此之前跑消融只会产生七行一样的表格。
    #
    # 闸门放在解码参考图**之前**：那是两分钟的开销，而且失败得越早越好。
    gate_cfg = base.get("thresholds", {})
    if not bool(gate_cfg.get("use_recon_z", False)) and not args.force_meaningless:
        logger.error(
            "消融实验当前**无效**：thresholds.use_recon_z=false，"
            "门控判定只依赖确定性信息量特征，与模型结构无关，"
            "所有预设的结果必然逐位相同。\n"
            "  真正依赖模型的指标只有重建误差（overfit_ratio 等）。\n"
            "  若仍要跑（例如只想比较重建质量），加 --force-meaningless。"
        )
        return 3

    # --- 参考图与划分只准备一次，七组共用 ---
    dcfg = base.get("data", {})
    size = tuple(base.get("model", {}).get("input_size", MODEL_DEFAULTS["input_size"]))
    ref_paths = list_ref_images(root / dcfg.get("acdc_root", "data/raw/acdc"))
    if not ref_paths:
        logger.error("未找到正常天气参考图")
        return 2
    logger.info("解码参考图 %d 张（七组共用，只做一次）...", len(ref_paths))
    refs = load_refs(ref_paths, size)

    groups = [sequence_of(p) for p in ref_paths] if dcfg.get("split_by_sequence", True) else None
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
    logger.info(
        "统一划分（所有预设共用）：训练 %d / 验证 %d / 校准 %d / 测试 %d",
        len(tr_idx), len(val_idx), len(ca_idx), len(te_idx),
    )

    device = resolve_device(str(base.get("device", "auto")))

    for i, preset in enumerate(presets, 1):
        tag = preset["tag"]
        if tag in results and not args.force:
            logger.info("[%d/%d] 跳过 %s（已有结果）", i, len(presets), tag)
            continue

        logger.info("=" * 66)
        logger.info("[%d/%d] %s  覆盖: %s", i, len(presets), tag, preset.get("set", {}))
        logger.info("=" * 66)

        cfg = apply_config_overrides(base, preset.get("set", {}))
        # 每组独立目录：共用的话第二次会捡起第一次的权重续训，
        # 那测的是「继续训练」而不是「换结构重训」，结论完全错
        cfg.setdefault("train", {})["checkpoint_dir"] = (
            f"artifacts/checkpoints/visibility_ablations/{tag}"
        )
        if args.epochs is not None:
            cfg["train"]["epochs"] = args.epochs

        t0 = time.time()
        trainer = VisibilityTrainer(cfg, project_root=root, resume="none")
        history = trainer.fit()
        train_s = time.time() - t0

        ckpt = root / cfg["train"]["checkpoint_dir"] / "best.pt"
        scorer = VisibilityScorer.from_checkpoint(
            ckpt,
            device=device,
            cfg=cfg.get("model", {}),
            scoring_cfg=cfg.get("scoring", {}),
        )
        gate = VisibilityGate.from_config(cfg)

        rng = np.random.default_rng(0)
        metrics = evaluate_run(
            scorer, gate, refs, tr_idx, val_idx, ca_idx, rng
        )
        metrics["train_seconds"] = round(train_s, 1)
        metrics["epochs_run"] = len(history.train_loss)
        metrics["best_epoch"] = history.best_epoch
        metrics["overrides"] = preset.get("set", {})
        results[tag] = metrics

        results_path.write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info(
            "  %s 完成: 参数 %.2fM 过拟合比 %.3f 验证误报 %.1f%% 平均召回 %.1f%% (%.0f 秒)",
            tag, metrics["params_m"], metrics["overfit_ratio"],
            metrics["val_fpr"] * 100, metrics["mean_recall"] * 100, train_s,
        )

    return write_report(results, out_dir)


def write_report(results: dict[str, Any], out_dir: Path) -> int:
    if not results:
        logger.error("没有任何结果")
        return 2

    order = ["baseline"] + [k for k in results if k != "baseline"]
    order = [k for k in order if k in results]

    L: list[str] = [
        "# 能见度门控消融实验",
        "",
        "所有预设使用**完全相同的划分**（按序列整组、固定种子）与**相同的评估口径**。",
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

    L += ["", "## 各组配置", "", "| 预设 | 覆盖项 |", "|------|--------|"]
    for tag in order:
        L.append(f"| {tag} | `{results[tag].get('overrides') or '（基线）'}` |")

    base = results.get("baseline")
    if base:
        L += ["", "## 相对基线的增益", "",
              "| 预设 | 召回变化 | 参数量变化 | 过拟合比变化 |", "|------|---------|-----------|-------------|"]
        for tag in order:
            if tag == "baseline":
                continue
            m = results[tag]
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

    # 检测「所有组指标完全相同」这种退化情形 —— 它几乎总意味着
    # 被比较的指标与模型无关，而不是「架构真的不影响」。
    recalls = {round(results[t]["mean_recall"], 6) for t in order}
    fps = {round(results[t]["val_fpr"], 6) for t in order}
    degenerate = len(recalls) == 1 and len(fps) == 1 and len(order) > 1

    if degenerate:
        L += [
            "## ⚠️ 本次消融无效（结果逐位相同）",
            "",
            f"全部 {len(order)} 组的平均召回与验证误报**完全相同**"
            f"（召回 {list(recalls)[0]:.1%}，误报 {list(fps)[0]:.1%}）。",
            "",
            "这不是「架构不影响性能」，而是**门控判定与模型结构无关**：",
            "`thresholds.use_recon_z=false` 时，判定只依赖",
            "`compute_information_features(图像)` 算出的确定性特征，",
            "该函数不经过模型 —— 所以换任何编码器，判定结果在数学上必然一致。",
            "",
            "真要通过消融比较架构，必须先让被比较的指标依赖模型：",
            "重新启用 `use_recon_z`（并先证明它对当前模型有判别力），",
            "或改为比较重建质量本身（本表里的 `overfit_ratio` 是唯一有模型依赖性的列）。",
            "",
            "---",
            "",
        ]

    L += [
        "## 读表须知",
        "",
        "- **过拟合比** = validation 重建误差 / train 重建误差。接近 1.0 表示泛化良好；",
        "  显著大于 1 说明模型记住了训练序列。因为已按序列整组划分，",
        "  这个数字现在反映的是真实的跨场景泛化，不再受相邻帧泄漏污染。",
        "  **这是全表唯一依赖模型的指标**，架构差异只应在这里体现。",
        "- 验证误报和合成退化召回用于消融比较；test 集不在此脚本中读取，留给最终评估。",
        "- **平均召回** 是四种退化在最大强度下判为 BLIND 的比例均值。",
        "  注意它衡量的是「确定看不见时会不会报警」，不是日常可用性。",
        "- ⚠️ 召回与误报**不依赖模型**（见上）。只有过拟合比那一列能区分架构，",
        "  且它受抽样规模影响，N 较小时差异可能落在噪声内，不要过度解读。",
        "",
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
