"""先检查效率，再实际训练并把最佳模型对照写入 model_report.md。"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import logging
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from car_smart_assist.utils.stop import STOP_EXIT_CODE, StopController  # noqa: E402

from filelock import FileLock, Timeout  # noqa: E402

from car_smart_assist.config.weather_attention import (  # noqa: E402
    attention_model_config,
    load_weather_attention_config,
    validate_weather_attention_config,
)
from car_smart_assist.perception.weather_attention_training import (  # noqa: E402
    compare_attention,
    model_report,
    preflight_attention,
    prepare_attention_datasets,
    require_independent_weather_output,
    require_independent_weather_reports,
    train_attention,
    write_json,
)


def main(argv=None, *, stop_controller=None):
    parser = argparse.ArgumentParser(description="预训练属性注意力天气候选，默认恢复已有 last")
    parser.add_argument("--config", default="configs/model/weather_attention.yaml")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--preflight", action="store_true", help="只检查真实图像前后向和效率，不更新参数"
    )
    modes.add_argument("--fresh", action="store_true", help="保存已有 best/last 快照，再开新实验")
    parser.add_argument(
        "--total-epochs", type=int, default=None, help="新实验累计目标；普通恢复须与原配方一致，追加预算使用 continuation"
    )
    modes.add_argument("--stop", action="store_true", help="只写停止请求，不启动训练")
    modes.add_argument("--continue", dest="continue_training", action="store_true", help="恢复配对 last 与原累计目标")
    parser.add_argument("--stop-file", type=Path, default=None)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    root = Path(__file__).resolve().parents[1]
    cfg = load_weather_attention_config(root / args.config)
    if args.total_epochs is not None:
        cfg = replace(cfg, train={**cfg.train, "epochs": args.total_epochs})
        validate_weather_attention_config(cfg)
    if not args.stop:
        try:
            require_independent_weather_reports(cfg, root)
            if not args.preflight:
                require_independent_weather_output(root, root / cfg.checkpoint_root)
        except ValueError as exc:
            print(f"天气注意力训练未开始：{exc}", file=sys.stderr)
            return 2
    with (stop_controller or StopController(args.stop_file, default_directory=root / cfg.checkpoint_root)) as controller, ExitStack() as locks:
        if args.stop:
            controller.request_stop_file()
            print(f"已请求轮末停止：{controller.path}")
            return 0
        if args.continue_training and not (root / cfg.checkpoint_root / "last.pt").is_file():
            print("--continue 缺少本输出目录的 last.pt", file=sys.stderr)
            return 2
        lock_path = root / cfg.checkpoint_root / "run.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            locks.enter_context(FileLock(lock_path, timeout=0))
            if stop_controller is None and (args.fresh or args.continue_training):
                controller.resume()
            controller.require_running()
            if not args.preflight and cfg.train.get("swad", {}).get("enabled") and "continuation" not in cfg.train:
                from car_smart_assist.perception.weather_extension import require_run_manifest

                require_run_manifest(root, root / cfg.report_dir, "run_manifest.json")
            datasets, info = prepare_attention_datasets(cfg, root)
            if args.preflight:
                report = preflight_attention(cfg, root, datasets, info)
                print(json.dumps(report, ensure_ascii=False, indent=2))
            else:
                profile_path = root / cfg.report_dir / "preflight.json"
                if not profile_path.is_file():
                    raise ValueError("请先运行 --preflight 完成真实数据与效率检查")
                profile = json.loads(profile_path.read_text(encoding="utf-8"))
                if (
                    profile.get("model_config") != attention_model_config(cfg)
                    or profile["split_signature"] != info["signature"]
                    or profile["batch_size"] != cfg.train["batch_size"]
                    or profile["channels_last"] != cfg.train["channels_last"]
                    or profile.get("swad") != cfg.train.get("swad")
                    or profile.get("l2_sp_coefficient", 0) != cfg.train.get("l2_sp", 0)
                ):
                    raise ValueError("模型、批量或数据已改变，请重新运行 --preflight")
                trained_model, trained = train_attention(
                    cfg, root, datasets, info, fresh=args.fresh, stop_controller=controller,
                    continue_only=args.continue_training
                )
                if trained.get("manual_stop") or controller.requested:
                    print(json.dumps(trained, ensure_ascii=False, indent=2))
                    return STOP_EXIT_CODE
                report = compare_attention(
                    cfg, root, datasets, info, trained_model, trained, stop_controller=controller
                )
                controller.require_running()
                print(
                    json.dumps(
                        {
                            "status": report["status"],
                            "best_epoch": trained["best_epoch"],
                            "report": str(root / cfg.report_file),
                        },
                        ensure_ascii=False,
                    )
                )
        except Timeout:
            print("已有天气注意力任务运行中，请读取 progress.json", file=sys.stderr)
            return 2
        except SystemExit as exc:
            if exc.code != STOP_EXIT_CODE:
                raise
            path = root / cfg.report_dir / "progress.json"
            previous = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"epoch": 0}
            previous.update(status="手动停止", manual_stop=True, termination_reason="manual_stop")
            write_json(path, previous)
            model_report(cfg, root, previous)
            return STOP_EXIT_CODE
        except Exception as exc:
            path = root / cfg.report_dir / "progress.json"
            previous = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"epoch": 0}
            previous.update(status="中断", error=f"{type(exc).__name__}: {exc}")
            write_json(path, previous)
            model_report(cfg, root, previous)
            logging.exception("天气注意力任务中断；已有检查点保留")
            return 2
        return STOP_EXIT_CODE if controller.requested else 0


if __name__ == "__main__":
    raise SystemExit(main())
