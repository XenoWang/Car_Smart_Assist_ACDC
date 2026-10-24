"""追加阶段完成后再核对结果、选择默认模型并保留只读对照。"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import stat

import numpy as np
import torch
import yaml

from car_smart_assist.perception.weather_attention_ablation import assert_protected_files
from car_smart_assist.perception.weather_attention_training import (
    attention_sampling_weights,
    prepare_attention_datasets,
    write_json,
)


def choose_extension(result, policy):
    """只用验证评分判断 SWAD 是否胜过 A 组及同轮单点。"""
    candidate = result["models"].get(policy["candidate"])
    reference = result["models"][policy["reference"]]["selection"]["score"]
    single = result["models"]["single_best"]["selection"]["score"]
    threshold = max(reference, single) if policy["require_beating_single_best"] else reference
    score = candidate["selection"]["score"] if candidate else None
    if (
        not all(math.isfinite(x) for x in (reference, single, threshold))
        or score is not None
        and not math.isfinite(score)
    ):
        raise ValueError("验证评分非有限，不能修改默认模型")
    success = score is not None and score > threshold + policy["min_validation_gain"]
    return {
        "swad_effective": success,
        "swad_validation_score": score,
        "a_validation_score": reference,
        "single_validation_score": single,
        "required_validation_score": threshold,
        "decision_basis": "validation_only",
        "needs_alternative_research": not success,
    }


def _samples(root, visibility):
    from scripts.run_pipeline import build_samples

    samples = build_samples(root, visibility)
    if len(samples) != 5:
        raise ValueError("默认切换验证需要正常参考和四类真实图片")
    return samples + [("全黑帧", np.zeros((1080, 1920, 3), np.uint8))]


def _check_pipeline(outputs):
    for output in outputs:
        if output.advisory is None or "gate" in output.skipped or "weather" in output.skipped:
            raise ValueError("新默认管线未通过实际运行验证")
        if not output.blocked and (output.weather is None or output.perception is None):
            raise ValueError("真实图片没有完成天气和目标感知")
    if not outputs[-1].blocked or not outputs[-1].advisory.should_takeover:
        raise ValueError("黑帧阻断或接管提示未通过")


def finish_extension(cfg, root, config_path, policy, *, stop_controller=None):
    from car_smart_assist.utils.stop import StopController

    controller = stop_controller or StopController(default_directory=root / cfg.checkpoint_root)
    with controller:
        controller.require_running()
        return _finish_extension(cfg, root, config_path, policy, controller)


def _finish_extension(cfg, root, config_path, policy, stop_controller):
    from car_smart_assist.inference.pipeline import InferencePipeline

    folder, checkpoints = root / cfg.report_dir, root / cfg.checkpoint_root
    result = json.loads((folder / "comparison.json").read_text(encoding="utf-8"))
    manifest = json.loads((folder / "extension_manifest.json").read_text(encoding="utf-8"))
    if result["status"] != "完成" or manifest["policy"] != policy:
        raise ValueError("追加阶段未完成或选优策略改变")
    assert_protected_files(root, manifest)
    for path, expected in manifest.get("experiment_hashes", {}).items():
        if hashlib.sha256((root / path).read_bytes()).hexdigest() != expected:
            raise ValueError(f"本阶段配置或代码已改变：{path}")
    last = torch.load(checkpoints / "last.pt", map_location="cpu", weights_only=True)
    stage = cfg.train["continuation"]
    origin = torch.load(root / stage["checkpoint"], map_location="cpu", weights_only=True)
    if last["epoch"] != cfg.train["epochs"] or not last["training_complete"]:
        raise ValueError("追加轮次未完成")
    history = last["history"]
    if history[: stage["source_epoch"]] != origin["history"] or [
        row["epoch"] for row in history
    ] != list(range(1, cfg.train["epochs"] + 1)):
        raise ValueError("源历史被改变或追加轮次不连续")
    if any(row["skipped_updates"] for row in history):
        raise ValueError("出现跳过更新，需复核后再切换默认模型")
    stop_controller.require_running()
    datasets, _ = prepare_attention_datasets(cfg, root)
    weights = attention_sampling_weights(datasets["train"], cfg)
    generator = torch.Generator()
    generator.set_state(origin["generator_state"])
    for row in history[stage["source_epoch"] :]:
        order = torch.multinomial(
            weights, len(datasets["train"]), replacement=True, generator=generator
        ).numpy()
        if row["sampling_sha256"] != hashlib.sha256(order.tobytes()).hexdigest():
            raise ValueError("追加阶段没有延续源采样序列")
    del datasets
    if last["initial_model_sha256"] != origin["initial_model_sha256"]:
        raise ValueError("追加阶段的原始初始化记录改变")
    if not math.isclose(
        history[stage["source_epoch"]]["head_lr"], origin["history"][-1]["head_lr"], rel_tol=1e-10
    ):
        raise ValueError("追加学习率发生跳变")
    meta = result["swad"]
    if meta["converged"] and (
        meta["start_epoch"] <= stage["source_epoch"]
        or meta["start_step"] <= origin["swad_state"]["step"]
        or meta["averaged_updates"] != meta["end_step"] - meta["start_step"] + 1
    ):
        raise ValueError("SWAD 没有使用新阶段的独立区间")
    for name, field in (("single_best.pt", "best_sha256"), ("best.pt", "export_best_sha256")):
        if hashlib.sha256((checkpoints / name).read_bytes()).hexdigest() != last[field]:
            raise ValueError("最终检查点与 last 记录不一致")
    verdict = choose_extension(result, policy)
    if verdict["swad_effective"]:
        desired = {
            "weather_default": {
                "config": config_path.relative_to(root).as_posix(),
                "checkpoint": (checkpoints / "swad.pt").relative_to(root).as_posix(),
            }
        }
    else:
        desired = {
            "weather_default": {
                "config": policy["fallback_config"],
                "checkpoint": policy["fallback_checkpoint"],
            }
        }
    visibility = yaml.safe_load(
        (root / "configs/model/visibility.yaml").read_text(encoding="utf-8")
    )["visibility"]
    samples = _samples(root, visibility)
    stop_controller.require_running()
    explicit = InferencePipeline.from_config(
        visibility,
        project_root=root,
        device="cpu",
        weather_config=desired["weather_default"]["config"],
        weather_checkpoint=desired["weather_default"]["checkpoint"],
    )
    stop_controller.require_running()
    before = explicit.run_batch([image for _, image in samples])
    _check_pipeline(before)
    del explicit
    stop_controller.require_running()
    route_path = root / "configs/model/weather_default.yaml"
    original_bytes = route_path.read_bytes()
    current = yaml.safe_load(original_bytes)
    if current != manifest["default_route_before"] and current != desired:
        raise ValueError("默认路由被其他操作改变；未覆盖该修改")
    # Once publication starts, finish or roll it back before honoring another stop.
    stop_controller.require_running()
    temporary = route_path.with_suffix(".tmp.yaml")
    temporary.write_text(
        yaml.safe_dump(desired, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    temporary.replace(route_path)
    try:
        default = InferencePipeline.from_config(visibility, project_root=root, device="cpu")
        after = default.run_batch([image for _, image in samples])
        _check_pipeline(after)
        for a, b in zip(before, after, strict=True):
            if (
                a.blocked != b.blocked
                or a.advisory.should_takeover != b.advisory.should_takeover
                or (a.weather and (not b.weather or a.weather.decisions != b.weather.decisions))
            ):
                raise ValueError("默认加载与显式权重的运行不一致")
    except Exception:
        temporary.write_bytes(original_bytes)
        temporary.replace(route_path)
        raise
    verdict.update(
        status="完成",
        source_epoch=stage["source_epoch"],
        additional_epochs=stage["additional_epochs"],
        completed_epochs=last["epoch"],
        source_history_preserved=True,
        sampling_sequence_continued=True,
        lr_continuous=True,
        initial_record_preserved=True,
        pipeline_verified=True,
        swad=meta,
        default_route=desired,
    )
    result["default_pipeline_promoted"] = verdict["swad_effective"]
    result["deployment"] = {
        "kind": "swad" if verdict["swad_effective"] else policy["reference"],
        "route": desired,
        "decision_basis": "validation_only",
    }
    write_json(folder / "comparison.json", result)
    write_json(
        folder / "pipeline_verification.json",
        {
            "status": "通过",
            "route": desired,
            "samples": [
                {"name": label, **output.to_dict()}
                for (label, _), output in zip(samples, after, strict=True)
            ],
        },
    )
    stop_controller.require_running()
    if not verdict["swad_effective"]:
        archive = (
            root
            / policy["readonly_archive"]
            / hashlib.sha256((checkpoints / "last.pt").read_bytes()).hexdigest()[:12]
        ).resolve()
        if not archive.is_relative_to(root.resolve()):
            raise ValueError("只读对照必须保存在项目内")
        archive.mkdir(parents=True, exist_ok=True)
        sources = {
            name: checkpoints / name
            for name in ("best.pt", "single_best.pt", "swad.pt", "last.pt")
            if (checkpoints / name).exists()
        }
        sources.update(
            {
                "config.yaml": config_path,
                "comparison.json": folder / "comparison.json",
                "REPORT.md": folder / "REPORT.md",
            }
        )
        hashes = {}
        for name, source in sources.items():
            target = archive / name
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() != digest:
                raise ValueError("只读对照已存在且内容改变")
            if not target.exists():
                shutil.copyfile(source, target)
            target.chmod(stat.S_IREAD)
            hashes[name] = digest
        verdict["readonly_archive"] = archive.relative_to(root).as_posix()
        write_json(
            folder / "readonly_manifest.json",
            {"folder": verdict["readonly_archive"], "hashes": hashes, "read_only": True},
        )
    stop_controller.require_running()
    assert_protected_files(root, manifest)
    write_json(folder / "decision.json", verdict)
    section = f"\n## 追加600轮后的部署选择\n\n累计{last['epoch']}轮；按验证评分判断SWAD有效：{verdict['swad_effective']}。SWAD评分{verdict['swad_validation_score']}，旧A组{verdict['a_validation_score']:.6f}，同轮单点{verdict['single_validation_score']:.6f}。测试分数不参与选择。\n\n默认配置：`{desired['weather_default']['config']}`；权重：`{desired['weather_default']['checkpoint']}`。原始初始化、300轮历史、续接采样和连续学习率检查通过，实际默认管线及黑帧验证通过。\n"
    for path in (root / cfg.report_file, root / cfg.evaluation["summary_file"]):
        text = path.read_text(encoding="utf-8")
        begin, end = (
            "<!-- WEATHER_SWAD_EXTEND_600_DECISION_BEGIN -->",
            "<!-- WEATHER_SWAD_EXTEND_600_DECISION_END -->",
        )
        block = begin + section + end + "\n"
        if begin in text and end in text:
            before, rest = text.split(begin, 1)
            _, after = rest.split(end, 1)
            text = before + block + after
        else:
            text += "\n" + block
        path.write_text(text, encoding="utf-8")
    return verdict
