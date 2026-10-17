"""两组各 300 轮，保留当前 best，自动写对照结果。"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from filelock import FileLock, Timeout  # noqa: E402

from car_smart_assist.perception.weather_attention_ablation import (  # noqa: E402
    load_ablation_config,
    run_ablation,
)


def main():
    parser = argparse.ArgumentParser(description="固定 seed 42，冻结范围与 L2-SP 单项对照")
    parser.add_argument("--config", default="configs/model/weather_attention_ablation.yaml")
    parser.add_argument(
        "--preflight", action="store_true", help="先检查两组真实前后向和效率，不正式训练"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    root = Path(__file__).resolve().parents[1]
    options = load_ablation_config(root / args.config)
    lock_path = root / options["report_dir"] / "suite.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with FileLock(lock_path, timeout=0):
            result = run_ablation(options, root, preflight_only=args.preflight)
        if result:
            print(
                json.dumps(
                    {"status": result["status"], "report": str(root / options["report_file"])},
                    ensure_ascii=False,
                )
            )
    except Timeout:
        print("已有同一对照任务运行，请查看 progress.json", file=sys.stderr)
        return 2
    except Exception:
        logging.exception("天气对照未完成；基线和已有检查点保留")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
