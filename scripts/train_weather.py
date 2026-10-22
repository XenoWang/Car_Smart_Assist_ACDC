"""历史 CNN 复现实验入口；当前训练使用 train_weather_attention.py。"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from car_smart_assist.config.weather import load_weather_config  # noqa: E402
from car_smart_assist.perception.weather_training import train_weather  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="历史 CNN 复现实验；当前模型请使用 train_weather_attention.py"
    )
    parser.add_argument("--config", default="configs/model/weather_classifier.yaml")
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="先将已有 best/last 归档到输出目录的 history，再重开旧 CNN 实验；部署目录拒绝写入",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    try:
        cfg = load_weather_config(root / args.config)
        result = train_weather(cfg, root, resume=not args.fresh)
    except (FileNotFoundError, ValueError) as exc:
        print(f"天气模型训练未开始或中断：{exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {**asdict(result), "checkpoint": str(result.checkpoint)}, ensure_ascii=False, indent=2
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
