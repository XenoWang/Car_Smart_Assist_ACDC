"""训练完成后核对保存结果、固定种子和平均区间，再汇总分数。"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from car_smart_assist.config.weather_attention import load_weather_attention_config  # noqa: E402
from car_smart_assist.perception.weather_attention_ablation import (
    assert_protected_files,  # noqa: E402
)
from car_smart_assist.perception.weather_attention_training import write_json  # noqa: E402


def main():
    root = Path(__file__).resolve().parents[1]
    cfg = load_weather_attention_config(root / "configs/model/weather_attention_swad.yaml")
    folder, checkpoints = root / cfg.report_dir, root / cfg.checkpoint_root
    result = json.loads((folder / "comparison.json").read_text(encoding="utf-8"))
    if result["status"] != "完成":
        raise ValueError("正式训练尚未完成，不能汇总最终成绩")
    manifest = json.loads((folder / "run_manifest.json").read_text(encoding="utf-8"))
    assert_protected_files(root, manifest)
    for path, expected in manifest["source_hashes"].items():
        if hashlib.sha256((root / path).read_bytes()).hexdigest() != expected:
            raise ValueError(f"训练期间代码改变：{path}")
    last = torch.load(checkpoints / "last.pt", map_location="cpu", weights_only=True)
    control = torch.load(
        root / "artifacts/checkpoints/weather_attention_ablation/reduced_tuning/last.pt",
        map_location="cpu",
        weights_only=True,
    )
    history, previous = last["history"], control["history"]
    audit = {
        "completed_epochs": last["epoch"],
        "seed": last["recipe"]["seed"],
        "initial_matches_control": last["initial_model_sha256"] == control["initial_model_sha256"],
        "sampling_matches_control": [r["sampling_sha256"] for r in history]
        == [r["sampling_sha256"] for r in previous],
        "split_matches_control": last["recipe"]["split_signature"]
        == control["recipe"]["split_signature"],
        "datasets_match_control": last["recipe"]["dataset_signatures"]
        == control["recipe"]["dataset_signatures"],
        "consecutive_epochs": [r["epoch"] for r in history]
        == list(range(1, cfg.train["epochs"] + 1)),
        "all_updates_recorded": all(r["skipped_updates"] == 0 for r in history),
        "all_losses_finite": all(
            np.isfinite(r[k]) for r in history for k in ("loss", "data_loss", "swad_validation_bce")
        ),
        "native_checkpoint_matches_last": hashlib.sha256(
            (checkpoints / "single_best.pt").read_bytes()
        ).hexdigest()
        == last["best_sha256"],
        "export_checkpoint_matches_last": hashlib.sha256(
            (checkpoints / "best.pt").read_bytes()
        ).hexdigest()
        == last["export_best_sha256"],
        "averaging_interval_consistent": not result["swad"]["converged"]
        or (
            result["swad"]["averaged_updates"] > 0
            and result["swad"]["averaged_updates"]
            == result["swad"]["end_step"] - result["swad"]["start_step"] + 1
            and result["swad"]["end_step"] <= last["swad_state"]["step"]
        ),
        "swad": result["swad"],
    }
    if (
        not last["training_complete"]
        or last["epoch"] != cfg.train["epochs"]
        or any(value is False for value in audit.values())
    ):
        raise ValueError("训练、固定种子或保存检查未通过")
    best = torch.load(checkpoints / "best.pt", map_location="cpu", weights_only=True)
    assert best["selected_kind"] == result["selected_kind"]
    candidate = torch.load(
        checkpoints / f"{result['selected_kind']}.pt", map_location="cpu", weights_only=True
    )
    assert all(
        torch.equal(value, candidate["model_state"][name])
        for name, value in best["model_state"].items()
    )
    write_json(folder / "reproducibility.json", audit)
    diagnostics = {
        "native_best_epoch": last["best_epoch"],
        "native_best_validation_score": last["best_score"],
        "native_last_validation_score": history[-1]["selection"]["score"],
        "native_validation_decline": last["best_score"] - history[-1]["selection"]["score"],
        "last_asl_data_loss": history[-1]["data_loss"],
        "last_validation_bce": history[-1]["swad_validation_bce"],
        "training_seconds": last["training_seconds"],
        "selected_kind": result["selected_kind"],
        "swad": result["swad"],
    }
    write_json(folder / "training_diagnostics.json", diagnostics)
    os.environ["MPLCONFIGDIR"] = str(folder / ".matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    epochs = [r["epoch"] for r in history]
    axes[0].plot(
        epochs,
        [r["selection"]["score"] for r in history],
        label="new single trajectory",
        linewidth=1,
    )
    axes[0].plot(
        epochs, [r["selection"]["score"] for r in previous], label="previous A control", linewidth=1
    )
    axes[0].set_title("Validation selection score")
    axes[0].legend(fontsize=8)
    axes[1].plot(epochs, [r["swad_validation_bce"] for r in history], linewidth=1)
    meta = result["swad"]
    if meta["converged"]:
        axes[1].axhline(
            meta["loss_threshold"], color="tab:red", linestyle="--", label="valley threshold"
        )
        axes[1].axvspan(
            meta["start_epoch"],
            meta["end_epoch"],
            alpha=0.15,
            color="tab:green",
            label="averaged interval",
        )
        axes[1].legend(fontsize=8)
    axes[1].set_title("Source-balanced validation BCE")
    axes[2].semilogy(epochs, [max(r["data_loss"], 1e-12) for r in history], linewidth=1)
    axes[2].set_title("ASL training data loss")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.2)
    fig.savefig(folder / "learning_curves.png", dpi=160)
    plt.close(fig)
    names = [
        name
        for name in ("reduced_tuning", "baseline_147", "single_best", "swad")
        if name in result["models"]
    ]
    lines = [
        "## SWAD 最终复核",
        "",
        f"实际完成 {last['epoch']} 轮，GPU 训练约 {last['training_seconds'] / 60:.1f} 分钟。最终按验证评分导出 `{result['selected_kind']}`。",
        "",
        "| 模型 | 验证综合评分 | 平均 AP | 平均 F1 | 雨 F1 | 雾 F1 | 雪 F1 | 夜 F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in names:
        row = result["models"][name]
        test = row["test"]
        values = [row["selection"]["score"], test["macro_ap"], test["macro_f1"]] + [
            test["per_attribute"][a]["f1"] for a in ("rain", "fog", "snow", "night")
        ]
        lines.append(f"| {name} | " + " | ".join(f"{100 * x:.2f}%" for x in values) + " |")
    lines += [
        "",
        "### 雨雾漏判与误报分开看",
        "",
        "| 模型 | 来源 | 雨漏判／雨图 | 雨误报／无雨图 | 雾漏判／雾图 | 雾误报／无雾图 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for name in names:
        for source, metrics in result["models"][name]["test"]["by_source"].items():
            rain, fog = [metrics["per_attribute"][a] for a in ("rain", "fog")]
            lines.append(
                f"| {name} | {source} | {rain['fn']}/{rain['known_positive']} | {rain['fp']}/{rain['known_negative']} | {fog['fn']}/{fog['known_positive']} | {fog['fp']}/{fog['known_negative']} |"
            )
    lines += [
        "",
        f"[训练和谷底曲线]({(folder / 'learning_curves.png').as_posix()})",
        "",
        "初始化、300 轮采样顺序、划分和数据签名与 A 组一致，原权重和配置哈希通过。CUDA 没有强制逐位确定性；这仍是单次固定种子的项目开发回归，不是新的官方未见测试。",
        "",
        "采用 [SWAD 原论文](https://proceedings.neurips.cc/paper/2021/file/bcb41ccdc4363c6848a1d760f26c28a0-Paper.pdf) 的密集平均与损失谷底思路，保留本项目的 ASL、余弦计划和冻结 BN，并额外按验证评分比较导出版本。",
        "",
    ]
    section = "\n".join(lines)
    begin, end = "<!-- WEATHER_SWAD_ANALYSIS_BEGIN -->", "<!-- WEATHER_SWAD_ANALYSIS_END -->"
    block = begin + "\n" + section + end + "\n"
    for path in (root / cfg.report_file, root / cfg.evaluation["summary_file"]):
        text = path.read_text(encoding="utf-8")
        if begin in text and end in text:
            before, rest = text.split(begin, 1)
            _, after = rest.split(end, 1)
            text = before + block + after.lstrip("\n")
        elif "<!-- WEATHER_SWAD_END -->" in text:
            text = text.replace("<!-- WEATHER_SWAD_END -->", block + "<!-- WEATHER_SWAD_END -->", 1)
        else:
            text += "\n" + block
        path.write_text(text, encoding="utf-8")
    print(json.dumps(diagnostics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
