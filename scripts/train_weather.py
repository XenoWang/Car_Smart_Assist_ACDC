"""训练 ACDC 四类条件小模型：.venv/Scripts/python.exe scripts/train_weather.py。"""

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
    parser = argparse.ArgumentParser(description="训练 ACDC fog/night/rain/snow 小模型")
    parser.add_argument("--config", default="configs/model/weather_classifier.yaml")
    parser.add_argument("--fresh", action="store_true", help="从头训练并覆盖天气模型检查点")
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
