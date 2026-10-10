"""在留出数据上评测天气/光照多属性标签。"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from car_smart_assist.config.weather import WeatherConfig, load_weather_config  # noqa: E402
from car_smart_assist.data.preprocessing import load_invalid_entries  # noqa: E402
from car_smart_assist.perception.weather import WeatherPredictor  # noqa: E402
from car_smart_assist.perception.weather_training import (  # noqa: E402
    WeatherDataset,
    _acdc_split,
    attribute_metrics,
    list_condition_images,
    list_pixel_accurate_images,
    split_by_sequence,
)


def evaluate_samples(
    cfg: WeatherConfig,
    checkpoint: Path,
    samples,
    cache_file: Path,
    batch_size: int,
    teacher_checkpoint: Path | None = None,
) -> tuple[dict[str, Any], str]:
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Weather checkpoint not found: {checkpoint}")
    predictor = WeatherPredictor.from_checkpoint(checkpoint, cfg)
    dataset = WeatherDataset(samples, cfg, cache_file)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=predictor.device.type == "cuda",
    )
    all_targets: list[np.ndarray] = []
    all_probabilities: list[np.ndarray] = []
    with torch.inference_mode():
        for images, cues, labels in loader:
            logits = predictor.model(
                images.to(predictor.device, non_blocking=True),
                cues.to(predictor.device, non_blocking=True),
            )
            all_targets.append(labels.numpy())
            all_probabilities.append(torch.sigmoid(logits.float()).cpu().numpy())
    values = np.concatenate(all_probabilities)
    if not np.isfinite(values).all():
        raise ValueError("Weather probabilities are not finite")
    metrics = attribute_metrics(
        np.concatenate(all_targets),
        values,
        cfg.attributes,
        cfg.decision_thresholds,
    )
    if "enhanced" in cfg.train:
        from car_smart_assist.perception.weather_enhanced_training import metrics_by_source

        metrics["by_source"] = metrics_by_source(dataset, values, cfg)
    if teacher_checkpoint is not None:
        from car_smart_assist.perception.weather_enhanced_training import (
            metrics_by_source,
            probabilities,
        )

        teacher = WeatherPredictor.from_checkpoint(teacher_checkpoint, replace(cfg, rain_adapter_channels=0))
        old_values = probabilities(teacher.model, dataset, teacher.device, batch_size)
        metrics["teacher"] = attribute_metrics(dataset.targets, old_values, cfg.attributes, cfg.decision_thresholds)
        metrics["teacher"]["by_source"] = metrics_by_source(dataset, old_values, cfg)
        protected = [cfg.attributes.index(name) for name in ("fog", "snow", "night")]
        metrics["protected_max_probability_difference"] = float(np.abs(values[:, protected] - old_values[:, protected]).max())
    return metrics, str(predictor.device)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the weather attribute model")
    parser.add_argument("--config", default="configs/model/weather_classifier.yaml")
    parser.add_argument(
        "--dataset", choices=("acdc", "validation", "calibration", "pixel-accurate", "unseen-recordings"), default="acdc"
    )
    parser.add_argument("--data-root", default=None, help="ACDC root; defaults to train config")
    parser.add_argument("--input-zip", default=None, help="Pixel Accurate RGB archive")
    parser.add_argument("--checkpoint", default=None, help="defaults to configured checkpoint")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", default=None, help="JSON report path")
    parser.add_argument("--compare-teacher", action="store_true", help="增强配置下，使用同一批样本比较原模型与候选")
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]

    try:
        cfg = load_weather_config(project_root / args.config)
        checkpoint = Path(args.checkpoint) if args.checkpoint else Path(cfg.checkpoint)
        if not checkpoint.is_absolute():
            checkpoint = project_root / checkpoint
        cache_root = project_root / cfg.train["cache_dir"] / "evaluation"
        enhanced = cfg.train.get("enhanced")
        report_root = enhanced["report_dir"] if enhanced else "artifacts/reports/weather_attributes"
        if args.compare_teacher and not enhanced:
            raise ValueError("--compare-teacher 需要天气增强配置")
        teacher_checkpoint = project_root / enhanced["teacher_checkpoint"] if args.compare_teacher else None
        if args.dataset in ("acdc", "unseen-recordings"):
            data_root = Path(args.data_root) if args.data_root else Path(cfg.train["acdc_root"])
            if not data_root.is_absolute():
                data_root = project_root / data_root
            invalid = load_invalid_entries(project_root / cfg.train.get(
                "acdc_cleaning_manifest", "data/processed/manifests/acdc.json"), project_root)
            samples = _acdc_split(data_root, "test", cfg.attributes, invalid)
            if args.dataset == "unseen-recordings":
                development = {s.group for part in ("train", "val") for s in
                    list_condition_images(data_root, part, cfg.attributes, invalid)}
                samples = [s for s in samples if s.group not in development]
            if not samples:
                raise FileNotFoundError("ACDC official test images not found")
            metrics, device = evaluate_samples(
                cfg, checkpoint, samples, cache_root / "acdc_test.npy", args.batch_size, teacher_checkpoint
            )
            dataset_name = "ACDC"
            split = "official_test" if args.dataset == "acdc" else "official_test_unseen_recording_subset"
            filename = "acdc_test_metrics.json" if args.dataset == "acdc" else "unseen_recordings_metrics.json"
            output = args.output or f"{report_root}/{filename}"
            dataset_counts = {
                "per_attribute_positive": {
                    name: int(sum(sample.labels[index] for sample in samples))
                    for index, name in enumerate(cfg.attributes)
                }
            }
        elif args.dataset in ("validation", "calibration") and "enhanced" in cfg.train:
            from car_smart_assist.perception.weather_enhanced_training import enhanced_splits

            samples = enhanced_splits(cfg, project_root)[args.dataset]
            metrics, device = evaluate_samples(cfg, checkpoint, samples,
                project_root / cfg.train["cache_dir"] / f"enhanced_{args.dataset}.npy",
                args.batch_size, teacher_checkpoint)
            dataset_name, split = "ACDC+PixelAccurateDepthBenchmark", f"enhanced_{args.dataset}"
            output = args.output or f"{cfg.train['enhanced']['report_dir']}/{args.dataset}_metrics.json"
            dataset_counts = {"samples": len(samples), "recording_groups": len({s.group for s in samples})}
        elif args.dataset == "calibration":
            raise ValueError("普通天气配置没有独立校准划分；请使用 weather_enhanced.yaml")
        elif args.dataset == "validation":
            data_root = Path(args.data_root) if args.data_root else Path(cfg.train["acdc_root"])
            if not data_root.is_absolute():
                data_root = project_root / data_root
            official = [
                *list_condition_images(data_root, "train", cfg.attributes, load_invalid_entries(
                    project_root / cfg.train.get("acdc_cleaning_manifest", "data/processed/manifests/acdc.json"), project_root)),
                *list_condition_images(data_root, "val", cfg.attributes, load_invalid_entries(
                    project_root / cfg.train.get("acdc_cleaning_manifest", "data/processed/manifests/acdc.json"), project_root)),
            ]
            _, acdc_val = split_by_sequence(
                official,
                float(cfg.train["validation_fraction"]),
                int(cfg.train["seed"]),
            )
            samples = list(acdc_val)
            pixel_val_count = 0
            if cfg.train.get("use_pixel_accurate", True):
                archive_path = project_root / cfg.train["pixel_accurate_zip"]
                _, pixel_val = list_pixel_accurate_images(
                    archive_path,
                    cfg.attributes,
                    int(cfg.train["pixel_accurate_validation_scene"]),
                    load_invalid_entries(project_root / cfg.train.get("pixel_cleaning_manifest",
                        "data/processed/manifests/pixel_accurate_benchmark.json"), project_root),
                )
                samples.extend(pixel_val)
                pixel_val_count = len(pixel_val)
            metrics, device = evaluate_samples(
                cfg, checkpoint, samples, cache_root / "development_validation.npy", args.batch_size
            )
            dataset_name, split = "ACDC+PixelAccurateDepthBenchmark", "development_validation"
            output = args.output or (
                "artifacts/reports/weather_attributes/development_validation_metrics.json"
            )
            dataset_counts = {
                "acdc_sequence_validation_samples": len(acdc_val),
                "pixel_accurate_validation_scene_samples": pixel_val_count,
                "positive_by_attribute": {
                    name: int(sum(sample.labels[index] for sample in samples))
                    for index, name in enumerate(cfg.attributes)
                },
            }
        else:
            archive_path = Path(args.input_zip) if args.input_zip else Path(
                cfg.train["pixel_accurate_zip"]
            )
            if not archive_path.is_absolute():
                archive_path = project_root / archive_path
            validation_scene = int(cfg.train["pixel_accurate_validation_scene"])
            _, samples = list_pixel_accurate_images(
                archive_path, cfg.attributes, validation_scene,
                load_invalid_entries(project_root / cfg.train.get("pixel_cleaning_manifest",
                    "data/processed/manifests/pixel_accurate_benchmark.json"), project_root),
            )
            metrics, device = evaluate_samples(
                cfg,
                checkpoint,
                samples,
                cache_root / f"pixel_scene{validation_scene}_validation.npy",
                args.batch_size,
                teacher_checkpoint,
            )
            dataset_name, split = "PixelAccurateDepthBenchmark", f"scene_{validation_scene}_validation"
            output = args.output or (
                f"{report_root}/"
                f"pixel_scene{validation_scene}_validation_metrics.json"
            )
            dataset_counts = {
                "samples": len(samples),
                "positive_by_attribute": {
                    name: int(sum(sample.labels[index] for sample in samples))
                    for index, name in enumerate(cfg.attributes)
                },
                "scenes_used": [validation_scene],
            }
    except (FileNotFoundError, KeyError, OSError, ValueError) as exc:
        print(f"Weather evaluation failed: {exc}", file=sys.stderr)
        return 2

    report = {
        "dataset": dataset_name,
        "split": split,
        "checkpoint": str(checkpoint),
        "device": device,
        "attributes": list(cfg.attributes),
        "decision_thresholds": dict(cfg.decision_thresholds),
        "dataset_counts": dataset_counts,
        "metrics": metrics,
        "evaluation_limit": (
            "新训练／验证／校准互斥，但旧原模型曾接触部分开发留出数据；开发指标不能当全新场景泛化证明。"
            if "enhanced" in cfg.train else "按数据集协议评估；录制组重叠需与独立录制泛化区别。"
        ),
    }
    output_path = Path(output)
    if not output_path.is_absolute():
        output_path = project_root / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nReport saved: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
