"能见度门控的无监督训练入口。"

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from car_smart_assist.utils.stop import STOP_EXIT_CODE, StopController  # noqa: E402

from car_smart_assist.perception.visibility import VisibilityTrainer  # noqa: E402
from car_smart_assist.perception.visibility.trainer import archive_visibility_checkpoints  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="能见度门控的无监督训练",
        epilog="默认行为是**续训**：若 checkpoint_dir 下已有 last.pt，"
               "会自动载入权重与优化器状态接着训练。要强制重训请加 --fresh。",
    )
    p.add_argument("--config", default="configs/model/visibility.yaml")
    p.add_argument("--epochs", type=int, default=None, help="覆盖配置中的轮数")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--device", default=None, help="auto（优先 CUDA GPU）| cpu | cuda | cuda:0")
    p.add_argument("--log-level", default="INFO")
    p.add_argument("--stop-file", type=Path, default=None)

    g = p.add_mutually_exclusive_group()
    g.add_argument(
        "--resume", nargs="?", const="auto", default="auto", metavar="PATH",
        help="从检查点续训。不带参数时自动取 checkpoint_dir/last.pt（默认）。",
    )
    g.add_argument(
        "--fresh", action="store_true",
        help="先归档已有 last/best 和训练记录，再从头训练。",
    )
    g.add_argument("--stop", action="store_true", help="只写停止请求，不启动训练")
    g.add_argument("--continue", dest="continue_training", action="store_true", help="从配对 last 继续原阶段")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    root = Path(__file__).resolve().parents[1]
    cfg_path = root / args.config
    if not cfg_path.exists():
        print(f"配置文件不存在: {cfg_path}", file=sys.stderr)
        return 2

    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg = raw.get("visibility", raw)

    # 命令行覆盖（--set 那套等 config/loader.py 完成后再统一）
    if args.epochs is not None:
        cfg.setdefault("train", {})["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg.setdefault("train", {})["batch_size"] = args.batch_size
    if args.device is not None:
        cfg["device"] = args.device

    directory = root / cfg.get("train", {}).get("checkpoint_dir", "artifacts/checkpoints/visibility")
    with StopController(args.stop_file, default_directory=directory) as controller:
        if args.stop:
            controller.request_stop_file()
            print(f"已请求轮末停止：{controller.path}")
            return 0
        if args.fresh or args.continue_training:
            controller.resume()
        controller.require_running()
        resume = "none" if args.fresh else (
            directory / "last.pt" if args.continue_training else args.resume
        )
        if args.continue_training and not resume.is_file():
            logging.getLogger(__name__).error("--continue 缺少配对 last.pt：%s", resume)
            return 2

        try:
            if args.fresh:
                archive_visibility_checkpoints(
                    directory,
                    root / cfg.get("report", {}).get("output_dir", "artifacts/reports/visibility") / "train_history.json",
                )
            trainer = VisibilityTrainer(cfg, project_root=root, resume=resume)
        except ValueError as exc:
            logging.getLogger(__name__).error("无法初始化训练设备或检查点：%s", exc)
            return 2
        if trainer.resume_path is not None:
            print(f"续训来源: {trainer.resume_path}")
        else:
            print("未找到已有检查点，从头训练")

        try:
            history = trainer.fit(stop_controller=controller, continue_only=args.continue_training)
        except ValueError as exc:
            logging.getLogger(__name__).error("无法继续训练：%s", exc)
            return 2

        print("\n" + "=" * 62)
        print("手动停止，检查点已保留" if history.manual_stop else "训练完成")
        print("=" * 62)
        for k, v in history.to_dict().items():
            print(f"  {k:18s} {v}")
        if trainer.calibration:
            print("\n  零校准统计（正常天气参考图，未参与训练）:")
            for k, v in trainer.calibration.to_dict().items():
                print(f"    {k:8s} {v:.6f}" if isinstance(v, float) else f"    {k:8s} {v}")

        out = root / cfg.get("report", {}).get("output_dir", "artifacts/reports/visibility")
        out.mkdir(parents=True, exist_ok=True)
        (out / "train_history.json").write_text(
            json.dumps(
                {
                    "history": history.to_dict(),
                    "calibration": trainer.calibration.to_dict() if trainer.calibration else None,
                    "seed": trainer.seed,
                    "split_signature": trainer.split_signature,
                    "split_sizes": trainer.split_stats,
                },
                ensure_ascii=False, indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\n  训练记录: {out / 'train_history.json'}")
        if history.manual_stop or controller.requested:
            return STOP_EXIT_CODE
        print("\n  下一步: python scripts/evaluate_visibility.py   （验证召回率与单调性）")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
