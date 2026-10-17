"""历史金字塔模型复现；当前训练使用 train_weather_attention.py。"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from car_smart_assist.config.weather_pyramid import load_weather_pyramid_config  # noqa: E402
from car_smart_assist.data.weather_pyramid import prepare_pyramid_splits  # noqa: E402
from car_smart_assist.perception.weather_pyramid_training import (
    compare_pyramid_models,  # noqa: E402
)


def main():
    parser = argparse.ArgumentParser(
        description="历史金字塔复现实验；当前模型请使用 train_weather_attention.py"
    )
    parser.add_argument("--config", default="configs/model/weather_pyramid.yaml")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--fresh", action="store_true", help="保留历史快照，新开一轮训练；默认恢复已有 last"
    )
    modes.add_argument(
        "--evaluate-only", action="store_true", help="只读取已有权重比较，不追加训练"
    )
    parser.add_argument("--prepare-only", action="store_true", help="只生成和检查四分清单，不训练")
    parser.add_argument(
        "--total-epochs", type=int, default=None, help="每个版本累计跑到指定轮数，期间不提前早停"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    root = Path(__file__).resolve().parents[1]
    try:
        if args.total_epochs is not None and (args.total_epochs <= 0 or args.evaluate_only):
            raise ValueError("累计目标必须为正数，且不能与只评估模式同时使用")
        cfg = load_weather_pyramid_config(root / args.config)
        if args.prepare_only:
            _, report = prepare_pyramid_splits(cfg, root)
            report["splits"] = {
                name: {key: value for key, value in row.items() if key != "records"}
                for name, row in report["splits"].items()
            }
        else:
            variants = ("focal", "asl")
            if args.evaluate_only:
                variants = tuple(
                    name
                    for name in variants
                    if (root / cfg.checkpoint_root / name / "best.pt").is_file()
                )
                if not variants:
                    raise FileNotFoundError("没有保留的金字塔对照权重")
            report = compare_pyramid_models(
                cfg,
                root,
                fresh=args.fresh,
                evaluate_only=args.evaluate_only,
                total_epochs=args.total_epochs,
                variants=variants,
            )
            for model in report["models"].values():
                model.pop("history", None)
        print(json.dumps(report, ensure_ascii=False, indent=2))
    except (ValueError, FileNotFoundError) as exc:
        print(f"历史天气对照未完成：{exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
