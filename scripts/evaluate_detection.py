"""Separate YOLO validation, confidence calibration and final held-out test reporting."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from car_smart_assist.config.detection import load_yolo_config, resolve_yolo_device  # noqa: E402
from car_smart_assist.data.detection import validate_detection_export  # noqa: E402
from car_smart_assist.perception.detection_yolo import load_yolo_model  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate prepared YOLO detection splits")
    parser.add_argument("--config", default="configs/model/yolo_detection.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split", choices=("val", "calibration", "test"), default="val")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--imgsz", type=int, default=None, help="Validation-only input size comparison"
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    try:
        cfg = load_yolo_config(root / args.config)
        if args.imgsz is not None:
            if args.split != "val" or args.imgsz <= 0:
                raise ValueError("--imgsz is positive and only available for val comparisons")
            cfg["inference"]["imgsz"] = args.imgsz
        checkpoint = root / (args.checkpoint or cfg["checkpoint"])
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Trained weights missing: {checkpoint}")
        data = root / cfg["data"]
        validate_detection_export(data.parent)
        manifest = json.loads((data.parent / "manifest.json").read_text(encoding="utf-8"))
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        output = root / (args.output_dir or f"artifacts/reports/detection/{cfg['name']}")
        output.mkdir(parents=True, exist_ok=True)
        calibration_file = output / "calibration.json"
        locked_confidence = None
        class_thresholds = {}
        if args.split == "test":
            calibration = json.loads(calibration_file.read_text(encoding="utf-8"))
            if (
                calibration["checkpoint_sha256"] != digest
                or calibration["dataset_signature"] != manifest["signature"]
            ):
                raise ValueError(
                    "Calibration weights/data differ from this final test; recalibrate first"
                )
            for arg, report_key in (("iou", "nms_iou"), ("imgsz", "imgsz"), ("max_det", "max_det")):
                if calibration[report_key] != cfg["inference"][arg]:
                    raise ValueError("Calibration inference settings changed; recalibrate first")
            locked_confidence = float(calibration["confidence_threshold"])
            class_thresholds = calibration.get("class_confidence_thresholds", {})
        model = load_yolo_model(root, checkpoint)
        validation_args = {}
        if manifest.get("joint_supervision"):
            from car_smart_assist.perception.joint_yolo_backend import JointValidator

            validation_args["validator"] = JointValidator
        elif cfg.get("evaluation_matches_prediction", False):
            from car_smart_assist.perception.joint_yolo_backend import DeploymentValidator

            validation_args["validator"] = DeploymentValidator
        metrics = model.val(
            data=str(data.parent / "calibration.yaml" if args.split == "calibration" else data),
            split="val" if args.split == "calibration" else args.split,
            device=resolve_yolo_device(cfg["device"]),
            imgsz=cfg["inference"]["imgsz"],
            conf=0.001,
            iou=cfg["inference"]["iou"],
            max_det=cfg["inference"]["max_det"],
            batch=cfg["train"]["batch"],
            workers=cfg["train"]["workers"],
            project=str(output),
            name=args.split + "_plots",
            exist_ok=True,
            plots=True,
            **validation_args,
        )
        if args.split == "calibration":
            curve = metrics.box.f1_curve.mean(axis=0)
            if not np.isfinite(curve).all() or float(curve.max()) <= 0:
                raise ValueError("No valid positive detections to calibrate confidence")
            index = int(np.argmax(np.where(metrics.box.px > 0, curve, -np.inf)))
            locked_confidence = float(metrics.box.px[index])
            if manifest.get("joint_supervision"):
                for row, label in enumerate(metrics.box.ap_class_index):
                    curve = metrics.box.f1_curve[row]
                    best = int(np.argmax(np.where(metrics.box.px > 0, curve, -np.inf)))
                    # A class with no true positives cannot be calibrated from this split.
                    class_thresholds[model.names[int(label)]] = (
                        float(metrics.box.px[best])
                        if float(curve.max()) > 0
                        else float(cfg["inference"]["confidence"])
                    )
        elif locked_confidence is None:
            locked_confidence = float(cfg["inference"]["confidence"])
        index = int(np.argmin(np.abs(metrics.box.px - locked_confidence)))
        precision = float(metrics.box.p_curve[:, index].mean())
        recall = float(metrics.box.r_curve[:, index].mean())
        report = {
            "split": args.split,
            "dataset_signature": manifest["signature"],
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": digest,
            "confidence_threshold": locked_confidence,
            "class_confidence_thresholds": class_thresholds,
            "nms_iou": cfg["inference"]["iou"],
            "max_det": cfg["inference"]["max_det"],
            "imgsz": cfg["inference"]["imgsz"],
            "mAP50": float(metrics.box.map50),
            "mAP50_95": float(metrics.box.map),
            "operating_precision_iou50": precision,
            "operating_recall_iou50": recall,
            "operating_macro_f1_iou50": float(metrics.box.f1_curve[:, index].mean()),
            "per_class_mAP50_95": {
                model.names[int(label)]: float(metrics.box.all_ap[row].mean())
                for row, label in enumerate(metrics.box.ap_class_index)
            },
            "per_class_operating_iou50": {
                model.names[int(label)]: {
                    "precision": float(metrics.box.p_curve[row, index]),
                    "recall": float(metrics.box.r_curve[row, index]),
                    "f1": float(metrics.box.f1_curve[row, index]),
                }
                for row, label in enumerate(metrics.box.ap_class_index)
            },
            "image_count": sum(row["split"] == args.split for row in manifest["images"]),
            "threshold_source": "calibration"
            if args.split in ("calibration", "test")
            else "config",
        }
        if manifest.get("joint_supervision"):
            from car_smart_assist.perception.joint_yolo_backend import source_results

            report["source_metrics"] = source_results(metrics.box, manifest["known_class_count"])
            report["evaluation_scope"] = (
                "Per-source annotated classes only; Lost & Found ignores unannotated ROI"
            )
            # Operating metrics must use the deployed per-class thresholds, not a new test optimum.
            per_class = {}
            for row, label in enumerate(metrics.box.ap_class_index):
                name = model.names[int(label)]
                threshold = class_thresholds.get(name, locked_confidence)
                point = int(np.argmin(np.abs(metrics.box.px - threshold)))
                per_class[name] = {
                    "confidence_threshold": threshold,
                    "precision": float(metrics.box.p_curve[row, point]),
                    "recall": float(metrics.box.r_curve[row, point]),
                    "f1": float(metrics.box.f1_curve[row, point]),
                }
            report["per_class_operating_iou50"] = per_class
            for metric in ("precision", "recall", "f1"):
                report["operating_" + ("macro_f1" if metric == "f1" else metric) + "_iou50"] = (
                    float(np.mean([value[metric] for value in per_class.values()]))
                )
        (output / (args.split + ".json")).write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    except (ImportError, OSError, ValueError, KeyError) as exc:
        print(f"YOLO evaluation failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
