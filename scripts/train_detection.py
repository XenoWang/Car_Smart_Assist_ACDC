"""在准备好的固定随机种子训练划分上训练 YOLO；选 epoch 只看验证集。"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import importlib.metadata
import json
import os
import shutil
import sys
from pathlib import Path

import torch
from filelock import FileLock, Timeout

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from car_smart_assist.utils.stop import STOP_EXIT_CODE, StopController  # noqa: E402

from car_smart_assist.config.detection import (  # noqa: E402
    load_yolo_config,
    resolve_yolo_device,
    yolo_model_selection_contract,
)
from car_smart_assist.data.detection import validate_detection_export  # noqa: E402
from car_smart_assist.perception.detection_yolo import (  # noqa: E402
    configure_yolo_environment,
    load_yolo_model,
)
from car_smart_assist.perception.yolo_training import (  # noqa: E402
    EARLY_STOPPING_POLICY,
    attach_training_state_callbacks,
    detection_trainer_for_resume,
    load_training_state,
    require_model_selection_contract,
)


def resolve_training_plan(
    root, cfg, manifest, run, *, fresh=False, epochs=None, require_resume=False, continue_only=False
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
        "training_segment": {
            "start_epoch": 0,
            "target_epochs": train["epochs"],
            "resume_extra_epochs": None,
            "epochs_override": epochs,
            "early_stopping_policy": EARLY_STOPPING_POLICY,
        },
        "model_selection_contract": yolo_model_selection_contract(cfg, manifest),
    }
    if fresh:
        if continue_only:
            raise ValueError("--continue 与 --fresh 不能同时使用")
        return plan
    selection_contract = plan["model_selection_contract"]
    local_checkpoints = (run / "weights/last.pt", run / "weights/best.pt")
    if any(path.is_file() for path in local_checkpoints):
        metadata_path = run / "run_metadata.json"
        if not metadata_path.is_file():
            require_model_selection_contract({}, selection_contract)
        require_model_selection_contract(
            json.loads(metadata_path.read_text(encoding="utf-8")), selection_contract
        )
    if continue_only and not all(path.is_file() for path in local_checkpoints):
        raise ValueError("--continue requires this run's paired last.pt and best.pt")
    candidates = (local_checkpoints[0],) if continue_only else (*local_checkpoints, root / cfg["checkpoint"])
    checkpoint = next((path for path in candidates if path.is_file()), None)
    if checkpoint is None:
        if require_resume:
            raise FileNotFoundError("No existing detection checkpoint to resume")
        return plan
    source_run = checkpoint.parent.parent
    same_run = source_run.resolve() == run.resolve()
    previous = json.loads((source_run / "run_metadata.json").read_text(encoding="utf-8"))
    if (
        previous["dataset_signature"] != manifest["signature"]
        or previous["seed"] != train["seed"]
        or previous["classes"] != manifest["classes"]
    ):
        raise ValueError("Existing model dataset/seed/classes changed; use --fresh")
    summary_path = source_run / "validation_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
    if same_run and summary_path.is_file():
        require_model_selection_contract(summary, selection_contract)
    state = None
    completed = False
    if same_run:
        state_path = checkpoint.parent / "last_training_state.pt"
        if checkpoint.name != "last.pt":
            raise ValueError(
                "Same-directory resume requires paired last/best/FP32 state; "
                "use a new run directory (--name) to fine-tune best weights"
            )
        state = load_training_state(
            state_path, checkpoint, manifest["signature"], selection_contract
        )
        # 旧metadata在以上检查中拒绝；仅合法同目录恢复需要读取checkpoint的epoch。
        # 反序列化会导入Ultralytics，须先设好项目本地配置。
        configure_yolo_environment(root)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        completed = int(payload.get("epoch", -1)) < 0
    plan["checkpoint"] = checkpoint
    plan["completed"] = completed
    if state is not None:
        train = dict(previous["train_args"])
        train.pop("resume_extra_epochs", None)
        start = int(state["epoch"]) + 1
        saved_segment = state["training_segment"]
        # 请求先写metadata再启动训练；若首轮保存前中断，仍绑定旧last恢复本段目标。
        if previous.get("segment_resume_checkpoint_sha256") == state["checkpoint_sha256"]:
            saved_segment = previous["training_segment"]
        if continue_only:
            target = saved_segment["target_epochs"]
            if type(target) is not int or target <= 0 or start > target:
                raise ValueError("Saved continuation target is invalid")
            if epochs is not None and epochs != target:
                raise ValueError("--continue preserves the saved target; --epochs cannot change it")
            train["epochs"] = target
            plan.update(
                mode="resume", resume_state=state, train=train,
                training_segment=dict(saved_segment),
                epochs_before_run=previous.get("epochs_before_run", 0),
                no_op=start == target or bool(state.get("training_complete")) or completed,
            )
            return plan
        target = epochs if epochs is not None else int(train["epochs"])
        if extra is not None:
            if (
                saved_segment["resume_extra_epochs"] == extra
                and saved_segment["epochs_override"] == epochs
                and start < saved_segment["target_epochs"]
            ):
                target = saved_segment["target_epochs"]
            else:
                target = start + extra
        elif start >= target:
            target = start + (epochs if epochs is not None else cfg["train"]["epochs"])
        if (
            target == saved_segment["target_epochs"]
            and start < target
            and (
                extra is None
                or (
                    saved_segment["resume_extra_epochs"] == extra
                    and saved_segment["epochs_override"] == epochs
                )
            )
        ):
            segment = dict(saved_segment)
        else:
            segment = {
                "start_epoch": start,
                "target_epochs": target,
                "resume_extra_epochs": extra,
                "epochs_override": epochs,
                "early_stopping_policy": EARLY_STOPPING_POLICY,
            }
        train["epochs"] = target
        plan.update(
            mode="resume",
            resume_state=state,
            train=train,
            training_segment=segment,
            epochs_before_run=previous.get("epochs_before_run", 0),
        )
    else:
        # 新目录只微调权重，不继承不能核验的历史best分数。
        best = checkpoint.parent / "best.pt"
        plan["checkpoint"] = best if best.is_file() else checkpoint
        train["epochs"] = extra or train["epochs"]
        plan["training_segment"].update(
            target_epochs=train["epochs"], resume_extra_epochs=extra
        )
        plan.update(
            mode="finetune",
            epochs_before_run=summary.get(
                "total_epochs_completed",
                previous.get("epochs_before_run", 0) + summary.get("epochs_completed", 0),
            ),
        )
    plan["train"]["close_mosaic"] = min(plan["train"]["close_mosaic"], plan["train"]["epochs"])
    return plan


def archive_checkpoints(run, *, clear=False):
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
    originals = {path: path.read_bytes() for path in files}
    for path, original in originals.items():
        destination = target / path.relative_to(run)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        if destination.read_bytes() != original or path.read_bytes() != original:
            raise OSError(f"Training archive readback mismatch: {path}")
    if clear:
        if any(path.read_bytes() != original for path, original in originals.items()):
            raise OSError("Training files changed while archiving; originals retained")
        for path in originals:
            path.unlink()


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
    group.add_argument("--stop", action="store_true", help="Write a stop marker without starting training")
    group.add_argument("--continue", dest="continue_training", action="store_true", help="Resume paired last to its saved target")
    parser.add_argument("--stop-file", type=Path, default=None)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    try:
        cfg = load_yolo_config(root / args.config)
        name = args.name or cfg["name"]
        if Path(name).name != name or name in (".", ".."):
            raise ValueError("Run name must be a simple directory name")
        project = root / cfg["project"]
        run = project / name
        with StopController(args.stop_file, default_directory=run / "weights") as controller, ExitStack() as locks:
            if args.stop:
                controller.request_stop_file()
                print(f"Stop requested: {controller.path}")
                return 0
            run.mkdir(parents=True, exist_ok=True)
            locks.enter_context(FileLock(str((run / "run.lock").resolve()), timeout=0))
            if args.fresh or args.continue_training:
                controller.resume()
            controller.require_running()
            data = root / cfg["data"]
            counts = validate_detection_export(data.parent, verify_hashes=True)
            manifest = json.loads((data.parent / "manifest.json").read_text(encoding="utf-8"))
            metadata_path = run / "run_metadata.json"
            plan = resolve_training_plan(
                root,
                cfg,
                manifest,
                run,
                fresh=args.fresh,
                epochs=args.epochs,
                require_resume=args.resume,
                continue_only=args.continue_training,
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
                "model_selection_contract": plan["model_selection_contract"],
                "training_segment": plan["training_segment"],
                "segment_resume_checkpoint_sha256": (
                    plan["resume_state"]["checkpoint_sha256"] if plan["resume_state"] else None
                ),
            }
            if plan.get("no_op"):
                state = plan["resume_state"]
                summary_path = run / "validation_summary.json"
                summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
                summary.update(
                    no_op=True, training_complete=True, manual_stop=False,
                    termination_reason="budget_completed" if state["epoch"] + 1 == train["epochs"] else "early_stopping",
                    epochs_completed=state["epoch"] + 1,
                    best_checkpoint=str(run / "weights/best.pt"),
                    best_checkpoint_sha256=state["best_checkpoint_sha256"],
                    checkpoint_sha256=state["checkpoint_sha256"],
                    training_segment=plan["training_segment"],
                )
                print(json.dumps(summary, ensure_ascii=False, indent=2))
                return STOP_EXIT_CODE if controller.requested else 0
            run.mkdir(parents=True, exist_ok=True)
            model = load_yolo_model(root, plan["checkpoint"])
            controller.require_running()
            if args.fresh or plan["mode"] == "finetune" or plan["completed"]:
                archive_checkpoints(run, clear=args.fresh or plan["mode"] == "finetune")
            attach_training_state_callbacks(
                model, manifest["signature"], plan["resume_state"], plan["best_fitness"],
                selection_contract=plan["model_selection_contract"],
                training_segment=plan["training_segment"],
                stop_controller=controller,
            )

            def save_metadata(_trainer):
                temporary = metadata_path.with_suffix(".tmp")
                with temporary.open("w", encoding="utf-8") as stream:
                    stream.write(json.dumps(metadata, ensure_ascii=False, indent=2))
                    stream.flush()
                    os.fsync(stream.fileno())
                temporary.replace(metadata_path)

            model.add_callback("on_model_save", save_metadata)
            print(
                f"YOLO training mode: {plan['mode']}; source: {plan['checkpoint']}; target epochs: {train['epochs']}"
            )
            print(
                "YOLO model-selection contract (validation only): "
                + json.dumps(plan["model_selection_contract"], ensure_ascii=False, sort_keys=True)
            )
            trainer_class = detection_trainer_for_resume(
                plan["resume_state"], plan["model_selection_contract"]
            )
            if manifest.get("joint_supervision"):
                from car_smart_assist.perception.joint_yolo_backend import joint_trainer

                trainer_class = joint_trainer(
                    plan["resume_state"], cfg.get("joint_foreground_exclusive", True),
                    selection_contract=plan["model_selection_contract"],
                )
            if plan["resume_state"] is not None:
                save_metadata(None)
            controller.require_running()
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
            final_state = load_training_state(
                run / "weights/last_training_state.pt", model.trainer.last,
                manifest["signature"], plan["model_selection_contract"],
            )
            summary = {
                **metadata,
                "best_checkpoint": str(model.trainer.best),
                "best_checkpoint_sha256": final_state["best_checkpoint_sha256"],
                "epochs_completed": model.trainer.epoch + 1,
                "manual_stop": final_state.get("manual_stop", False),
                "training_complete": final_state.get("training_complete", False),
                "termination_reason": final_state.get("termination_reason"),
                "total_epochs_completed": plan["epochs_before_run"] + model.trainer.epoch + 1,
                "validation_metrics": {
                    key: float(value) for key, value in (
                        results.results_dict if results is not None else model.trainer.metrics
                    ).items()
                },
            }
            (run / "validation_summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            return STOP_EXIT_CODE if final_state.get("manual_stop") or controller.requested else 0
    except (ImportError, OSError, ValueError, KeyError, RuntimeError, Timeout) as exc:
        print(f"YOLO training failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
