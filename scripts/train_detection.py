"""在准备好的固定随机种子训练划分上训练 YOLO；选 epoch 只看验证集。"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import shutil
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from car_smart_assist.config.detection import load_yolo_config, resolve_yolo_device  # noqa: E402
from car_smart_assist.data.detection import validate_detection_export  # noqa: E402
from car_smart_assist.perception.detection_yolo import (  # noqa: E402
    configure_yolo_environment,
    load_yolo_model,
)
from car_smart_assist.perception.yolo_training import (  # noqa: E402
    attach_training_state_callbacks,
    detection_trainer_for_resume,
    load_training_state,
)


def resolve_training_plan(
    root, cfg, manifest, run, *, fresh=False, epochs=None, require_resume=False
):
    train = dict(cfg["train"])
    extra = train.pop("resume_extra_epochs", None)
    if epochs is not None:
        if epochs <= 0:
            raise ValueError("--epochs must be positive")
        train["epochs"] = epochs
    if train["seed"] != manifest["seed"]:
        raise ValueError("Training seed must match the prepared split seed")
    train["close_mosaic"] = min(train["close_mosaic"], train["epochs"])
    plan = {
        "train": train,
        "checkpoint": root / cfg["model"],
        "mode": "fresh",
        "resume_state": None,
        "best_fitness": None,
        "epochs_before_run": 0,
        "completed": False,
    }
    if fresh:
        return plan
    candidates = (run / "weights/last.pt", run / "weights/best.pt", root / cfg["checkpoint"])
    checkpoint = next((path for path in candidates if path.is_file()), None)
    if checkpoint is None:
        if require_resume:
            raise FileNotFoundError("No existing detection checkpoint to resume")
        return plan
    source_run = checkpoint.parent.parent
    previous = json.loads((source_run / "run_metadata.json").read_text(encoding="utf-8"))
    if (
        previous["dataset_signature"] != manifest["signature"]
        or previous["seed"] != train["seed"]
        or previous["classes"] != manifest["classes"]
    ):
        raise ValueError("Existing model dataset/seed/classes changed; use --fresh")
    summary_path = source_run / "validation_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
    # 反序列化 YOLO checkpoint 会导入 Ultralytics，所以要在这次读取之前就设好它的本地
    # 配置，而不是只在构建 predictor 之前设。
    configure_yolo_environment(root)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    completed = int(payload.get("epoch", -1)) < 0
    state_path = checkpoint.parent / "last_training_state.pt"
    state = None
    same_run = source_run.resolve() == run.resolve()
    if same_run and checkpoint.name == "last.pt":
        if not completed:
            state = load_training_state(state_path, checkpoint, manifest["signature"])
        elif state_path.is_file():
            candidate = torch.load(state_path, map_location="cpu", weights_only=True)
            if candidate.get("format_version", 1) >= 2:
                state = load_training_state(state_path, checkpoint, manifest["signature"])
    plan["checkpoint"] = checkpoint
    plan["completed"] = completed
    if state is not None:
        train = dict(previous["train_args"])
        train.pop("resume_extra_epochs", None)
        start = int(state["epoch"]) + 1
        target = epochs if epochs is not None else int(train["epochs"])
        if extra is not None:
            target = start + extra
        elif start >= target:
            target = start + (epochs if epochs is not None else cfg["train"]["epochs"])
        train["epochs"] = target
        plan.update(
            mode="resume",
            resume_state=state,
            train=train,
            epochs_before_run=previous.get("epochs_before_run", 0),
        )
    else:
        # 旧版已完成的 checkpoint 只含权重；续训从现有 best 权重开始，
        # 优化器重新初始化，此路径不加载 COCO 预训练权重。
        best = checkpoint.parent / "best.pt"
        plan["checkpoint"] = best if best.is_file() else checkpoint
        train["epochs"] = extra or train["epochs"]
        plan.update(
            mode="finetune",
            epochs_before_run=summary.get(
                "total_epochs_completed",
                previous.get("epochs_before_run", 0) + summary.get("epochs_completed", 0),
            ),
        )
        if same_run:
            plan["best_fitness"] = summary.get("validation_metrics", {}).get("fitness")
    plan["train"]["close_mosaic"] = min(plan["train"]["close_mosaic"], plan["train"]["epochs"])
    return plan


def archive_checkpoints(run):
    files = [
        run / name
        for name in ("run_metadata.json", "validation_summary.json", "args.yaml", "results.csv")
    ]
    files += [run / "weights" / name for name in ("last.pt", "best.pt", "last_training_state.pt")]
    files = [path for path in files if path.is_file()]
    if not files:
        return
    index = 1
    history = run / "history"
    while (history / f"round_{index:04d}").exists():
        index += 1
    target = history / f"round_{index:04d}"
    for path in files:
        destination = target / path.relative_to(run)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)


def main() -> int:
    parser = argparse.ArgumentParser(description="Fixed-seed ACDC YOLO training")
    parser.add_argument("--config", default="configs/model/yolo_detection.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--name", default=None)
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--resume",
        action="store_true",
        help="Explicitly require an existing model; default is automatic resume",
    )
    group.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore existing training state and restart from config model",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    try:
        cfg = load_yolo_config(root / args.config)
        data = root / cfg["data"]
        counts = validate_detection_export(data.parent)
        manifest = json.loads((data.parent / "manifest.json").read_text(encoding="utf-8"))
        name = args.name or cfg["name"]
        if Path(name).name != name or name in (".", ".."):
            raise ValueError("Run name must be a simple directory name")
        project = root / cfg["project"]
        run = project / name
        metadata_path = run / "run_metadata.json"
        plan = resolve_training_plan(
            root,
            cfg,
            manifest,
            run,
            fresh=args.fresh,
            epochs=args.epochs,
            require_resume=args.resume,
        )
        train = plan["train"]
        metadata = {
            "dataset_signature": manifest["signature"],
            "seed": train["seed"],
            "split_counts": counts,
            "classes": manifest["classes"],
            "train_args": train,
            "ultralytics_version": importlib.metadata.version("ultralytics"),
            "training_mode": plan["mode"],
            "source_checkpoint": str(plan["checkpoint"]),
            "epochs_before_run": plan["epochs_before_run"],
            "joint_foreground_exclusive": cfg.get("joint_foreground_exclusive", True),
        }
        run.mkdir(parents=True, exist_ok=True)
        model = load_yolo_model(root, plan["checkpoint"])
        if args.fresh or plan["mode"] == "finetune" or plan["completed"]:
            archive_checkpoints(run)
        attach_training_state_callbacks(
            model, manifest["signature"], plan["resume_state"], plan["best_fitness"]
        )

        def save_metadata(_trainer):
            temporary = metadata_path.with_suffix(".tmp")
            temporary.write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            temporary.replace(metadata_path)

        model.add_callback("on_model_save", save_metadata)
        print(
            f"YOLO training mode: {plan['mode']}; source: {plan['checkpoint']}; target epochs: {train['epochs']}"
        )
        trainer_class = detection_trainer_for_resume(plan["resume_state"])
        if manifest.get("joint_supervision"):
            from car_smart_assist.perception.joint_yolo_backend import joint_trainer

            trainer_class = joint_trainer(
                plan["resume_state"], cfg.get("joint_foreground_exclusive", True)
            )
        results = model.train(
            data=str(data),
            project=str(project),
            name=name,
            device=resolve_yolo_device(cfg["device"]),
            exist_ok=True,
            resume=plan["mode"] == "resume",
            trainer=trainer_class,
            **train,
        )
        summary = {
            **metadata,
            "best_checkpoint": str(model.trainer.best),
            "epochs_completed": model.trainer.epoch + 1,
            "total_epochs_completed": plan["epochs_before_run"] + model.trainer.epoch + 1,
            "validation_metrics": {
                key: float(value) for key, value in results.results_dict.items()
            },
        }
        (run / "validation_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    except (ImportError, OSError, ValueError, KeyError) as exc:
        print(f"YOLO training failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
