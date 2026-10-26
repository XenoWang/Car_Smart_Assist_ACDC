"""分开报告低阈值AP、proposal校准和正式预算下的锁定操作点。"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from car_smart_assist.config.detection import (  # noqa: E402
    YOLO_AGNOSTIC_NMS,
    YOLO_AP_MODE,
    YOLO_LOCKED_MODE,
    YOLO_PROPOSAL_MODE,
    YOLO_SWEEP_CONFIDENCE,
    load_yolo_config,
    resolve_yolo_device,
)
from car_smart_assist.data.detection import validate_detection_export  # noqa: E402
from car_smart_assist.perception.detection_yolo import load_yolo_model  # noqa: E402
from car_smart_assist.perception.yolo_postprocess import (  # noqa: E402
    calibration_identity,
    calibration_path,
    load_calibration,
    ordered_class_names,
    validate_calibration,
)


def atomic_json(path, payload):
    """同目录临时文件写完后才替换，失败不发布半份校准。"""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=path.name + ".", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def select_thresholds(metrics, names, fallback, per_class):
    curve = metrics.box.f1_curve.mean(axis=0)
    if not np.isfinite(curve).all() or float(curve.max()) <= 0:
        raise ValueError("No valid positive detections to calibrate confidence")
    positive = metrics.box.px > 0
    index = int(np.argmax(np.where(positive, curve, -np.inf)))
    confidence = float(metrics.box.px[index])
    # 缺GT或无正确检出类不能从其他类曲线借阈值。
    thresholds = {name: float(fallback) for name in names}
    sources = {name: "config_fallback_no_supported_positive" for name in names}
    for row, label in enumerate(metrics.box.ap_class_index):
        name = names[int(label)]
        class_curve = metrics.box.f1_curve[row]
        if np.isfinite(class_curve).all() and float(class_curve.max()) > 0:
            best = int(np.argmax(np.where(positive, class_curve, -np.inf)))
            thresholds[name] = float(metrics.box.px[best]) if per_class else confidence
            sources[name] = (
                "calibration_class_f1_proposal"
                if per_class else "calibration_global_f1_proposal"
            )
    return confidence, thresholds, sources


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
        calibration_file = calibration_path(root, cfg, args.output_dir)
        if args.split == "calibration" and checkpoint.resolve() != (root / cfg["checkpoint"]).resolve():
            if args.output_dir is None or calibration_file.resolve() == calibration_path(root, cfg).resolve():
                raise ValueError("A non-default checkpoint requires an independent --output-dir for calibration")
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Trained weights missing: {checkpoint}")
        data = root / cfg["data"]
        validate_detection_export(data.parent, verify_hashes=True)
        manifest = json.loads((data.parent / "manifest.json").read_text(encoding="utf-8"))
        if args.split == "test" and not calibration_file.is_file():
            raise FileNotFoundError(
                f"Current detection calibration missing; recalibrate first: {calibration_file}"
            )
        output = calibration_file.parent
        model = load_yolo_model(root, checkpoint)
        names = ordered_class_names(model.names)
        identity = calibration_identity(root, cfg, model.names, checkpoint, manifest)
        confidence = float(cfg["inference"]["confidence"])
        thresholds = {name: confidence for name in names}
        threshold_sources = {name: "config" for name in names}
        if args.split == "test":
            calibration = load_calibration(
                root, cfg, model.names, output_dir=args.output_dir,
                checkpoint=checkpoint, manifest=manifest,
            )
            confidence = float(calibration["confidence_threshold"])
            thresholds = calibration["class_confidence_thresholds"]
            threshold_sources = calibration["class_threshold_sources"]
        # 必须先准备项目本地Ultralytics设置，再导入验证器。
        from car_smart_assist.perception.joint_yolo_backend import (
            DeploymentValidator,
            JointValidator,
        )

        base = JointValidator if manifest.get("joint_supervision") else DeploymentValidator
        output.mkdir(parents=True, exist_ok=True)

        def validate(mode, locked_confidence=None, class_thresholds=None):
            class Validator(base):
                pass

            Validator.postprocess_mode = mode
            Validator.locked_confidence = locked_confidence
            Validator.class_thresholds = class_thresholds
            return model.val(
                data=str(
                    data.parent / "calibration.yaml" if args.split == "calibration" else data
                ),
                split="val" if args.split == "calibration" else args.split,
                validator=Validator,
                device=resolve_yolo_device(cfg["device"]),
                imgsz=cfg["inference"]["imgsz"],
                conf=YOLO_SWEEP_CONFIDENCE,
                iou=cfg["inference"]["iou"],
                max_det=cfg["inference"]["max_det"],
                agnostic_nms=YOLO_AGNOSTIC_NMS,
                single_cls=False,
                batch=cfg["train"]["batch"],
                workers=cfg["train"]["workers"],
                project=str(output),
                name=f"{args.split}_{mode}_plots",
                exist_ok=True,
                plots=True,
            )

        sweep_mode = YOLO_PROPOSAL_MODE if args.split == "calibration" else YOLO_AP_MODE
        sweep = validate(sweep_mode)
        if args.split == "calibration":
            confidence, thresholds, threshold_sources = select_thresholds(
                sweep, names, cfg["inference"]["confidence"],
                bool(manifest.get("joint_supervision")),
            )
        locked = validate(YOLO_LOCKED_MODE, confidence, thresholds)
        operating = locked.operating_iou50
        per_class = operating["per_class"]
        for name in names:
            per_class[name]["confidence_threshold"] = thresholds[name]
        report = {
            **identity,
            "split": args.split,
            "checkpoint": str(checkpoint),
            "confidence_threshold": confidence,
            "class_confidence_thresholds": thresholds,
            "class_threshold_sources": threshold_sources,
            "threshold_source": (
                "calibration" if args.split in ("calibration", "test") else "config"
            ),
            "passes": [sweep_mode, YOLO_LOCKED_MODE],
            "sweep": {
                "mode": sweep_mode,
                "confidence_floor": YOLO_SWEEP_CONFIDENCE,
                "quantity_budget": (
                    "unlimited" if args.split == "calibration" else "formal_max_nms_and_max_det"
                ),
                "mAP50": float(sweep.box.map50),
                "mAP50_95": float(sweep.box.map),
                "per_class_mAP50_95": {
                    names[int(label)]: float(sweep.box.all_ap[row].mean())
                    for row, label in enumerate(sweep.box.ap_class_index)
                },
            },
            "operating_iou50": operating,
            "operating_precision_iou50": operating["macro"]["precision"],
            "operating_recall_iou50": operating["macro"]["recall"],
            "operating_macro_f1_iou50": operating["macro"]["f1"],
            "per_class_operating_iou50": per_class,
            "image_count": sum(row["split"] == args.split for row in manifest["images"]),
            "threshold_selection_scope": (
                "config" if args.split == "val"
                else "F1 proposal curves; not a global optimum under formal budgets"
            ),
            "evaluation_scope": (
                "Per-source annotated classes; Lost & Found annotated ROI only"
                if manifest.get("joint_supervision") else "Exported annotated classes"
            ),
        }
        if args.split != "calibration":
            # 顶层AP仅来自正式预算的ap_sweep，不能把锁定阈值后的AP当常规mAP。
            report.update(
                mAP50=float(sweep.box.map50), mAP50_95=float(sweep.box.map),
                mAP_mode=YOLO_AP_MODE,
            )
            report["per_class_mAP50_95"] = report["sweep"]["per_class_mAP50_95"]
        if manifest.get("joint_supervision"):
            from car_smart_assist.perception.joint_yolo_backend import source_results

            report["sweep"]["source_metrics"] = source_results(
                sweep.box, manifest["known_class_count"]
            )
            if args.split != "calibration":
                report["source_metrics"] = report["sweep"]["source_metrics"]
        report_file = output / (
            "calibration_report.json" if args.split == "calibration" else args.split + ".json"
        )
        if args.split == "calibration":
            validate_calibration(report, identity, model.names)
        atomic_json(report_file, report)
        if args.split == "calibration":
            # locked遍历、精确操作点报告和身份校验全部成功后再发布。
            atomic_json(calibration_file, report)
        print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (ImportError, OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        print(f"YOLO evaluation failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
