"""历史 CNN 复现实验入口；当前训练使用 train_weather_attention.py。"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from car_smart_assist.utils.stop import STOP_EXIT_CODE, StopController  # noqa: E402

from car_smart_assist.config.weather import load_weather_config  # noqa: E402
from car_smart_assist.perception.weather_training import train_weather  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="历史 CNN 复现实验；当前模型请使用 train_weather_attention.py"
    )
    parser.add_argument("--config", default="configs/model/weather_classifier.yaml")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--fresh",
        action="store_true",
        help="先将已有 best/last 归档到输出目录的 history，再重开旧 CNN 实验；部署目录拒绝写入",
    )
    modes.add_argument("--stop", action="store_true", help="只写停止请求，不启动训练")
    modes.add_argument("--continue", dest="continue_training", action="store_true", help="恢复配对 last 与原累计目标")
    parser.add_argument("--stop-file", type=Path, default=None)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    try:
        cfg = load_weather_config(root / args.config)
        with StopController(args.stop_file, default_directory=(root / cfg.checkpoint).parent) as controller:
            if args.stop:
                controller.request_stop_file()
                print(f"已请求轮末停止：{controller.path}")
                return 0
            if args.fresh or args.continue_training:
                controller.resume()
            controller.require_running()
            result = train_weather(
                cfg, root, resume=not args.fresh, stop_controller=controller,
                continue_only=args.continue_training,
            )
    except (FileNotFoundError, ValueError) as exc:
        print(f"天气模型训练未开始或中断：{exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {**asdict(result), "checkpoint": str(result.checkpoint)}, ensure_ascii=False, indent=2
        )
    )
    return STOP_EXIT_CODE if result.manual_stop or controller.requested else 0


if __name__ == "__main__":
    raise SystemExit(main())
