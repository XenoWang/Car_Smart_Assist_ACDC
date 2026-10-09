"""Build fixed-seed train/val/calibration/test ACDC data for YOLO detection."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from car_smart_assist.data.detection import (  # noqa: E402
    load_detection_data_config,
    prepare_detection_data,
    validate_detection_export,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare four-way ACDC YOLO detection data")
    parser.add_argument("--config", default="configs/data/acdc_detection.yaml")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--verify-hashes", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        raw = yaml.safe_load((root / args.config).read_text(encoding="utf-8"))
        joint = "joint_detection_data" in raw
        cfg = (
            raw["joint_detection_data"] if joint else load_detection_data_config(root / args.config)
        )
        if args.verify_only:
            result = validate_detection_export(root / cfg["output_root"], args.verify_hashes)
        else:
            if joint:
                from car_smart_assist.data.detection_joint import prepare_joint_detection

                result = prepare_joint_detection(root, cfg)
            else:
                result = prepare_detection_data(root, cfg)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Detection data preparation failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
