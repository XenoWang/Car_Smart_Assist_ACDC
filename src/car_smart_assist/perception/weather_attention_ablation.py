"""固定初始化与采样的两项单独对照，保留第 147 轮基线。"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import shutil
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

import torch
import yaml
from filelock import FileLock

from car_smart_assist.config.weather_attention import (
    attention_model_config,
    load_weather_attention_config,
    validate_weather_attention_config,
)
from car_smart_assist.perception.weather import select_device
from car_smart_assist.perception.weather_attention import (
    WeatherAttention,
    WeatherAttentionPredictor,
    ensure_pretrained_weights,
)
from car_smart_assist.perception.weather_attention_training import (
    attention_probabilities,
    attention_sampling_weights,
    compare_attention,
    detailed_metrics,
    preflight_attention,
    prepare_attention_datasets,
    train_attention,
    write_json,
)
from car_smart_assist.perception.weather_pyramid_training import state_hash

logger = logging.getLogger(__name__)


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_ablation_config(path):
    values = yaml.safe_load(Path(path).read_text(encoding="utf-8"))["weather_attention_ablation"]
    if set(values["experiments"]) != {"reduced_tuning", "l2_sp"}:
        raise ValueError("本轮仅比较减少微调范围及 L2-SP 两项")
    if type(values["expected_seed"]) is not int:
        raise ValueError("固定种子须为整数")
    for key in ("training_epochs", "schedule_epochs", "expected_baseline_epoch"):
        if type(values[key]) is not int or values[key] <= 0:
            raise ValueError("预算、学习率计划及基线轮次须为正整数")
    if values["schedule_epochs"] < values["training_epochs"]:
        raise ValueError("学习率计划必须覆盖训练预算")
    experiments = values["experiments"]
    if set(experiments["reduced_tuning"]) != {"backbone"} or set(
        experiments["reduced_tuning"]["backbone"]
    ) != {"fine_tune_from"}:
        raise ValueError("A 组只能改变骨干微调起始块")
    if set(experiments["l2_sp"]) != {"train"} or set(experiments["l2_sp"]["train"]) != {"l2_sp"}:
        raise ValueError("B 组只能增加 L2-SP 系数")
    return values


def experiment_configs(base, options, root):
    if base.data["seed"] != options["expected_seed"]:
        raise ValueError("当前数据种子与固定对照种子不符")
    if base.train["epochs"] != options["schedule_epochs"]:
        raise ValueError("本轮保持基线原来的学习率计划")
    configs = {}
    protected = (root / options["baseline_checkpoint"]).resolve()
    for name, changes in options["experiments"].items():
        fields = copy.deepcopy(asdict(base))
        fields["train"].update(
            epochs=options["training_epochs"], schedule_epochs=options["schedule_epochs"]
        )
        for section, values in changes.items():
            fields[section].update(values)
        fields.update(
            checkpoint_root=f"{options['checkpoint_root']}/{name}",
            report_dir=f"{options['report_dir']}/{name}",
            report_file=f"{options['report_dir']}/{name}/REPORT.md",
        )
        fields["attributes"], fields["image_size"] = (
            tuple(fields["attributes"]),
            tuple(fields["image_size"]),
        )
        cfg = replace(base, **fields)
        validate_weather_attention_config(cfg)
        output = (root / cfg.checkpoint_root).resolve()
        if not output.is_relative_to(root.resolve()) or output == protected.parent:
            raise ValueError("对照输出必须留在项目独立目录，不能覆盖基线")
        if (root / cfg.report_file).resolve() == (
            root / options["baseline_model_report"]
        ).resolve():
            raise ValueError("单组报告不能覆盖基线报告")
        configs[name] = cfg
    if configs["reduced_tuning"].backbone["fine_tune_from"] <= base.backbone["fine_tune_from"]:
        raise ValueError("A 组须减少可训练骨干范围")
    if configs["l2_sp"].train["l2_sp"] <= 0:
        raise ValueError("B 组需要正的 L2-SP 系数")
    return configs


def assert_protected_files(root, manifest):
    for path, expected in manifest["protected_hashes"].items():
        if file_hash(root / path) != expected:
            raise ValueError(f"基线或原配置已被外部改动：{path}")
    for name, expected in manifest["snapshot_hashes"].items():
        if file_hash(root / manifest["snapshot_root"] / name) != expected:
            raise ValueError(f"基线快照被改动：{name}")


def preserve_baseline(base, options, root, split_info, configs):
    manifest_path = root / options["report_dir"] / "baseline_manifest.json"
    plan = {
        "seed": base.data["seed"],
        "epochs": options["training_epochs"],
        "schedule_epochs": options["schedule_epochs"],
        "experiments": options["experiments"],
        "split_signature": split_info["signature"],
        "base_model": attention_model_config(base),
    }
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["plan"] != plan:
            raise ValueError("对照计划已改变，请指定新的实验输出目录")
        assert_protected_files(root, manifest)
        return manifest
    best = torch.load(root / options["baseline_checkpoint"], map_location="cpu", weights_only=True)
    last = torch.load(root / options["baseline_last"], map_location="cpu", weights_only=True)
    if best["epoch"] != options["expected_baseline_epoch"] or best[
        "model_config"
    ] != attention_model_config(base):
        raise ValueError("基线 best 轮次或结构不符")
    if best["split_signature"] != split_info["signature"] or not best.get("calibration_signature"):
        raise ValueError("基线数据划分改变或尚未校准")
    if last["recipe"]["train"] != base.train or last["recipe"]["seed"] != base.data["seed"]:
        raise ValueError("基线保存的训练配方与当前配置不符")
    if last["best_sha256"] != file_hash(root / options["baseline_checkpoint"]):
        raise ValueError("基线 best 与 last 哈希不符")
    if last["recipe"]["pretrained_sha256"] != file_hash(root / base.backbone["weights_file"]):
        raise ValueError("当前 ImageNet 权重与基线初始化不同")
    history = last["history"]
    eligible = [row for row in history if row["epoch"] <= options["training_epochs"]]
    budget_best = max(eligible, key=lambda row: row["selection"]["score"])
    if budget_best["epoch"] != best["epoch"]:
        raise ValueError("历史基线的最优点不在相同预算内，需另做匹配预算的基线")
    snapshot_root = f"{options['checkpoint_root']}/baseline_147"
    folder = root / snapshot_root
    folder.mkdir(parents=True, exist_ok=True)
    copies = {
        "best.pt": options["baseline_checkpoint"],
        "config.yaml": options["base_config"],
        "comparison.json": options["baseline_report"],
        "history.json": options["baseline_history"],
        "model_report.md": options["baseline_model_report"],
    }
    protected = {
        value: file_hash(root / value) for key, value in copies.items() if key != "model_report.md"
    }
    protected[options["baseline_last"]] = file_hash(root / options["baseline_last"])
    protected[base.backbone["weights_file"]] = file_hash(root / base.backbone["weights_file"])
    for name, source in copies.items():
        shutil.copy2(root / source, folder / name)
    device = select_device(base.device)
    devices = (
        [device.index if device.index is not None else torch.cuda.current_device()]
        if device.type == "cuda"
        else []
    )
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(base.data["seed"])
        initial = WeatherAttention(base, ensure_pretrained_weights(base, root))
        initial_hash = state_hash(initial)
    manifest = {
        "plan": plan,
        "protected_hashes": protected,
        "snapshot_root": snapshot_root,
        "snapshot_hashes": {name: file_hash(folder / name) for name in copies},
        "baseline_epoch": best["epoch"],
        "baseline_trained_epochs": last["epoch"],
        "baseline_validation_selection": best["selection"],
        "baseline_first_budget_best": budget_best["epoch"],
        "reconstructed_initial_sha256": initial_hash,
        "initial_hash_note": "历史运行未记录初始哈希；按保存配方与种子重建，新的两组实际记录哈希。",
        "pretrained_sha256": last["recipe"]["pretrained_sha256"],
        "groups": list(configs),
        "created": datetime.now().isoformat(timespec="seconds"),
    }
    assert_protected_files(root, manifest)
    write_json(manifest_path, manifest)
    for name, cfg in configs.items():
        destination = root / options["report_dir"] / name / "resolved_config.yaml"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            yaml.safe_dump({"weather_attention": asdict(cfg)}, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
    return manifest


def write_ablation_report(options, root, manifest, models, status):
    lines = [
        "# 天气模型单项对照",
        "",
        f"更新：{datetime.now().isoformat(timespec='seconds')}",
        "",
        f"状态：{status}。每组实际预算 {options['training_epochs']} 轮，学习率沿用 {options['schedule_epochs']} 轮计划。",
        "",
        "基线为原第 147 轮 best，只读保留。两组都从相同 ImageNet 权重重新开始，数据四分、初始化与采样种子均固定为 42。",
        "",
        "- A：只将骨干微调起始块从 7 改成 13，前 5 轮冻结安排不变。",
        "- B：保持原微调范围，只加 L2-SP；系数固定 1e-4，公式为系数乘可训练骨干到 ImageNet 起点距离平方和的一半。",
        "- ASL、采样、增广、Dropout、AdamW、模型选择和校准方法保持不变。未叠加早停或 SWAD。",
        "- 原运行完成 500 轮，但其前 300 轮最优也为 147；本轮保持前 300 轮学习率一致。",
        "- 每组验证选 best、校准选阈值；测试为已查看过的项目开发回归，不用于再调方案或系数。",
        "",
    ]
    if models:
        lines += [
            "## 成绩与基线的变化",
            "",
            "| 模型 | best 轮次 | 平均 AP | 平均 F1 | 雨 F1 | 雾 F1 | 雪 F1 | 夜 F1 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for name, row in models.items():
            scores = row["test"]
            f1 = [scores["per_attribute"][n]["f1"] for n in ("rain", "fog", "snow", "night")]
            lines.append(
                f"| {name} | {row['best_epoch']} | {100 * scores['macro_ap']:.2f}% | {100 * scores['macro_f1']:.2f}% | "
                + " | ".join(f"{100 * x:.2f}%" for x in f1)
                + " |"
            )
        if "baseline_147" in models:
            reference = models["baseline_147"]["test"]
            lines += [
                "",
                "| 对照 | 平均 F1 变化（百分点） | 雨 F1 变化 | 雾 F1 变化 | 雪 F1 变化 | 夜 F1 变化 |",
                "|---|---:|---:|---:|---:|---:|",
            ]
            for name, row in models.items():
                if name == "baseline_147":
                    continue
                scores = row["test"]
                delta = [scores["macro_f1"] - reference["macro_f1"]] + [
                    scores["per_attribute"][n]["f1"] - reference["per_attribute"][n]["f1"]
                    for n in ("rain", "fog", "snow", "night")
                ]
                lines.append(f"| {name} | " + " | ".join(f"{100 * x:+.2f}" for x in delta) + " |")
            lines += [
                "",
                "## 误报与漏报",
                "",
                "| 模型 | 雨漏判／雨图 | 雨误报／无雨图 | 雾漏判／雾图 | 雾误报／无雾图 |",
                "|---|---:|---:|---:|---:|",
            ]
            for name, row in models.items():
                rain, fog = [row["test"]["per_attribute"][n] for n in ("rain", "fog")]
                lines.append(
                    f"| {name} | {rain['fn']}/{rain['known_positive']} | {rain['fp']}/{rain['known_negative']} | {fog['fn']}/{fog['known_positive']} | {fog['fp']}/{fog['known_negative']} |"
                )
    else:
        lines += ["成绩尚未产生，不预填提高或下降。", ""]
    lines += [
        "",
        "## 过拟合检查",
        "",
        "ASL 数据损失与 L2-SP 约束单独记录，不能按总 loss 的大小横比两种目标。完整历史记录验证峰值与末轮差；差小也可能意味着欠拟合，要结合分项成绩。",
        "",
        "## 证据与文件",
        "",
        f"- 基线快照：`{manifest['snapshot_root']}`。",
        f"- 基线哈希、配方与划分：`{options['report_dir']}/baseline_manifest.json`。",
        f"- 实际进度：`{options['report_dir']}/progress.json` 及各组 progress.json。",
        f"- 初始参数与采样一致性：`{options['report_dir']}/reproducibility.json`（两组训练后生成）。",
        f"- 完整对照：`{options['report_dir']}/comparison.json`。",
        "",
        "默认 pipeline 不自动切换。保留原始数据、原配置和基线权重。",
    ]
    text = "\n".join(lines) + "\n"
    report = root / options["report_dir"] / "REPORT.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(text, encoding="utf-8")
    destination = root / options["report_file"]
    existing = destination.read_text(encoding="utf-8")
    begin, end = "<!-- WEATHER_ABLATION_BEGIN -->", "<!-- WEATHER_ABLATION_END -->"
    if begin in existing and end in existing:
        before, block = existing.split(begin, 1)
        _, after = block.split(end, 1)
        combined = before + begin + "\n" + text + end + after
    else:
        combined = existing + "\n\n---\n\n" + begin + "\n" + text + end + "\n"
    temporary = destination.with_suffix(".tmp.md")
    temporary.write_text(combined, encoding="utf-8")
    temporary.replace(destination)


def baseline_metrics(base, options, root, datasets, manifest):
    predictor = WeatherAttentionPredictor.from_checkpoint(
        root / manifest["snapshot_root"] / "best.pt", base
    )
    if predictor.cfg.image_size != datasets["test"].images.shape[1:3]:
        raise ValueError("基线与对照的图像缓存尺寸不符")
    values = attention_probabilities(predictor.model, datasets["test"], base, predictor.device)
    return {
        "best_epoch": manifest["baseline_epoch"],
        "thresholds": predictor.thresholds,
        "test": detailed_metrics(datasets["test"], values, base, predictor.thresholds),
        "validation": json.loads((root / options["baseline_report"]).read_text(encoding="utf-8"))[
            "models"
        ]["attention_asl"]["validation"],
        "selection": manifest["baseline_validation_selection"],
        "checkpoint": str(root / manifest["snapshot_root"] / "best.pt"),
    }


def verify_reproducibility(base, options, root, datasets, manifest):
    generator = torch.Generator().manual_seed(base.data["seed"])
    sampling = attention_sampling_weights(datasets["train"], base)
    expected_orders = [
        hashlib.sha256(
            torch.multinomial(
                sampling, len(datasets["train"]), replacement=True, generator=generator
            )
            .numpy()
            .tobytes()
        ).hexdigest()
        for _ in range(options["training_epochs"])
    ]
    records = {}
    for name in options["experiments"]:
        path = root / options["checkpoint_root"] / name / "last.pt"
        payload = torch.load(path, map_location="cpu", weights_only=True)
        records[name] = {
            "completed_epochs": payload["epoch"],
            "initial_sha256": payload["initial_model_sha256"],
            "initial_matches_reconstructed_baseline": payload["initial_model_sha256"]
            == manifest["reconstructed_initial_sha256"],
            "sampling_matches_fixed_seed": [r["sampling_sha256"] for r in payload["history"]]
            == expected_orders,
            "consecutive_epochs": [r["epoch"] for r in payload["history"]]
            == list(range(1, options["training_epochs"] + 1)),
            "no_skipped_updates": all(r["skipped_updates"] == 0 for r in payload["history"]),
        }
        if payload["epoch"] != options["training_epochs"] or not all(
            records[name][key]
            for key in (
                "initial_matches_reconstructed_baseline",
                "sampling_matches_fixed_seed",
                "consecutive_epochs",
            )
        ):
            raise ValueError(f"{name} 初始化、采样或训练轮次不一致")
    result = {
        "seed": base.data["seed"],
        "split_signature": manifest["plan"]["split_signature"],
        "groups": records,
    }
    write_json(root / options["report_dir"] / "reproducibility.json", result)
    return result


def run_ablation(options, root, *, preflight_only=False):
    base = load_weather_attention_config(root / options["base_config"])
    configs = experiment_configs(base, options, root)
    datasets, split_info = prepare_attention_datasets(base, root)
    manifest = preserve_baseline(base, options, root, split_info, configs)
    models = {}
    progress_path = root / options["report_dir"] / "progress.json"
    try:
        if preflight_only:
            for name, cfg in configs.items():
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(cfg.data["seed"])
                    initial = WeatherAttention(cfg, ensure_pretrained_weights(cfg, root))
                    actual_hash = state_hash(initial)
                if actual_hash != manifest["reconstructed_initial_sha256"]:
                    raise ValueError(f"{name} 初始权重与基线配方不同")
                profile = preflight_attention(cfg, root, datasets, split_info)
                profile["fixed_seed_initial_sha256"] = actual_hash
                write_json(root / cfg.report_dir / "preflight.json", profile)
                assert_protected_files(root, manifest)
            write_json(
                progress_path,
                {
                    "status": "训练前检查完成",
                    "groups": list(configs),
                    "epochs_per_group": options["training_epochs"],
                },
            )
            write_ablation_report(options, root, manifest, {}, "训练前检查完成；正式训练尚未开始")
            return
        for name, cfg in configs.items():
            output = root / cfg.report_dir
            profile_path = output / "preflight.json"
            if not profile_path.exists():
                raise ValueError("请先运行 --preflight 完成两组真实数据检查")
            profile = json.loads(profile_path.read_text(encoding="utf-8"))
            if (
                profile["model_config"] != attention_model_config(cfg)
                or profile["split_signature"] != split_info["signature"]
                or profile.get("l2_sp_coefficient", 0) != cfg.train.get("l2_sp", 0)
            ):
                raise ValueError("对照结构、数据或正则已变化，请重新检查")
            assert_protected_files(root, manifest)
            write_json(
                progress_path,
                {
                    "status": "训练中",
                    "active_group": name,
                    "epochs_per_group": options["training_epochs"],
                },
            )
            write_ablation_report(options, root, manifest, models, f"{name} 正在训练")
            logger.info(
                "对照 %s 开始，固定种子 %d，实际 %d 轮，学习率计划 %d 轮",
                name,
                base.data["seed"],
                options["training_epochs"],
                options["schedule_epochs"],
            )
            with FileLock(root / cfg.checkpoint_root / "run.lock", timeout=0):
                model, trained = train_attention(cfg, root, datasets, split_info)
                if trained["initial_model_sha256"] != manifest["reconstructed_initial_sha256"]:
                    raise ValueError("新组初始模型与固定种子基线不同")
                result = compare_attention(cfg, root, datasets, split_info, model, trained)
            models[name] = result["models"]["attention_asl"]
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            assert_protected_files(root, manifest)
        models = {
            "baseline_147": baseline_metrics(base, options, root, datasets, manifest),
            **models,
        }
        reproducibility = verify_reproducibility(base, options, root, datasets, manifest)
        assert_protected_files(root, manifest)
        result = {
            "status": "完成",
            "protocol": "project_fourfold_development_regression",
            "models": models,
            "reproducibility": reproducibility,
            "baseline_unchanged": True,
            "default_pipeline_promoted": False,
            "differences": options["experiments"],
            "epochs_per_group": options["training_epochs"],
            "schedule_epochs": options["schedule_epochs"],
        }
        write_json(root / options["report_dir"] / "comparison.json", result)
        write_json(
            progress_path,
            {
                "status": "完成",
                "groups": list(configs),
                "epochs_per_group": options["training_epochs"],
                "updated": datetime.now().isoformat(timespec="seconds"),
            },
        )
        write_ablation_report(options, root, manifest, models, "完成")
        return result
    except Exception as exc:
        write_json(
            progress_path,
            {
                "status": "中断",
                "error": f"{type(exc).__name__}: {exc}",
                "updated": datetime.now().isoformat(timespec="seconds"),
            },
        )
        write_ablation_report(options, root, manifest, models, f"中断：{type(exc).__name__}: {exc}")
        raise
