"""完成后的曲线和过拟合检查；只读权重，不再训练或调参数。"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from car_smart_assist.perception.weather_attention_ablation import (  # noqa: E402
    assert_protected_files,
    load_ablation_config,
)
from car_smart_assist.perception.weather_attention_training import write_json  # noqa: E402


def replace_summary(text, section):
    begin, end = (
        "<!-- WEATHER_ABLATION_DIAGNOSTICS_BEGIN -->",
        "<!-- WEATHER_ABLATION_DIAGNOSTICS_END -->",
    )
    block = begin + "\n" + section + end + "\n"
    if begin in text and end in text:
        before, rest = text.split(begin, 1)
        _, after = rest.split(end, 1)
        return before + block + after.lstrip("\n")
    if "<!-- WEATHER_ABLATION_END -->" in text:
        return text.replace(
            "<!-- WEATHER_ABLATION_END -->", block + "<!-- WEATHER_ABLATION_END -->", 1
        )
    return text + "\n" + block


def main():
    root = Path(__file__).resolve().parents[1]
    options = load_ablation_config(root / "configs/model/weather_attention_ablation.yaml")
    folder = root / options["report_dir"]
    result = json.loads((folder / "comparison.json").read_text(encoding="utf-8"))
    if result["status"] != "完成":
        raise ValueError("两组尚未完成，不能汇总最终结果")
    manifest = json.loads((folder / "baseline_manifest.json").read_text(encoding="utf-8"))
    assert_protected_files(root, manifest)
    histories = {
        "baseline_147": json.loads(
            (root / manifest["snapshot_root"] / "history.json").read_text(encoding="utf-8")
        )[: options["training_epochs"]]
    }
    for name in options["experiments"]:
        histories[name] = json.loads((folder / name / "history.json").read_text(encoding="utf-8"))
    diagnostics = {}
    for name, history in histories.items():
        best_epoch = result["models"][name]["best_epoch"]
        best, last = history[best_epoch - 1], history[-1]
        diagnostics[name] = {
            "best_epoch": best_epoch,
            "best_validation_score": best["selection"]["score"],
            "last_validation_score": last["selection"]["score"],
            "validation_decline": best["selection"]["score"] - last["selection"]["score"],
            "best_asl_data_loss": best.get("data_loss", best["loss"]),
            "last_asl_data_loss": last.get("data_loss", last["loss"]),
            "last_l2_sp_penalty": last.get("l2_sp_penalty", 0.0),
        }
    write_json(folder / "training_diagnostics.json", diagnostics)
    os.environ["MPLCONFIGDIR"] = str(folder / ".matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    for name, history in histories.items():
        epochs = [row["epoch"] for row in history]
        axes[0].plot(
            epochs, [row["selection"]["score"] for row in history], label=name, linewidth=1
        )
        axes[1].semilogy(
            epochs,
            [max(row.get("data_loss", row["loss"]), 1e-12) for row in history],
            label=name,
            linewidth=1,
        )
    history = histories["l2_sp"]
    axes[2].plot(
        [row["epoch"] for row in history],
        [row["l2_sp_penalty"] for row in history],
        color="tab:green",
    )
    axes[0].set_title("Validation selection score")
    axes[1].set_title("ASL data loss (not total objective)")
    axes[2].set_title("Weighted L2-SP penalty")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    fig.savefig(folder / "learning_curves.png", dpi=160)
    plt.close(fig)
    lines = [
        "## 验证曲线与过拟合迹象",
        "",
        "| 模型 | 验证 best | 末轮验证 | 回落（百分点） | 末轮 ASL | 末轮 L2-SP |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, row in diagnostics.items():
        lines.append(
            f"| {name} | {100 * row['best_validation_score']:.2f}% | {100 * row['last_validation_score']:.2f}% | {100 * row['validation_decline']:.2f} | {row['last_asl_data_loss']:.3e} | {row['last_l2_sp_penalty']:.3e} |"
        )
    lines += [
        "",
        "这里的验证综合评分不是准确率；回落较小不自动代表更好，整体欠拟合也可能表现为小回落。ASL 与正则项尺度不同，不比较谁的总 loss 更小。",
        "",
        f"[查看三组曲线]({(folder / 'learning_curves.png').as_posix()})",
        "",
        "本轮固定初始化、数据和采样种子，且核对了实际初始和采样哈希。CUDA 算子没有强制完全确定性，单次固定种子的微小差异不能直接当成稳定收益。",
        "",
    ]
    lines += [
        "## 最佳权重在不同来源上的表现",
        "",
        "| 模型 | 来源 | 图片数 | 雨 F1 | 雾 F1 | 雨漏判／雨图 | 雨误报／无雨图 | 雾漏判／雾图 | 雾误报／无雾图 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, model in result["models"].items():
        for source, metrics in model["test"]["by_source"].items():
            rain, fog = [metrics["per_attribute"][attribute] for attribute in ("rain", "fog")]
            scores = [
                "无正负对照" if row["f1"] is None else f"{100 * row['f1']:.2f}%"
                for row in (rain, fog)
            ]
            lines.append(
                f"| {name} | {source} | {metrics['samples']} | {' | '.join(scores)} | {rain['fn']}/{rain['known_positive']} | {rain['fp']}/{rain['known_negative']} | {fog['fn']}/{fog['known_positive']} | {fog['fp']}/{fog['known_negative']} |"
            )
    lines += [
        "",
        "| 模型 | Pixel Accurate 雨条件 | 检出／雨图 |",
        "|---|---|---:|",
    ]
    for name, model in result["models"].items():
        for context, rain in model["test"]["rain_contexts"].items():
            lines.append(f"| {name} | {context} | {rain['tp']}/{rain['known_positive']} |")
    lines += [
        "",
        "ACDC 是道路图片，Pixel Accurate 是受控场景；四个受控雨条件分别报告，避免总分掩盖某一种天气的漏判。受控夜间雨的表现不能直接代表真实道路夜间雨的能力。",
        "",
    ]
    section = "\n".join(lines)
    report_path = folder / "REPORT.md"
    report_path.write_text(
        replace_summary(report_path.read_text(encoding="utf-8"), section), encoding="utf-8"
    )
    report = root / options["report_file"]
    text = report.read_text(encoding="utf-8")
    text = replace_summary(text, section)
    report.write_text(text, encoding="utf-8")
    print(json.dumps(diagnostics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
