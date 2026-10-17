"""新天气模型的四分、未知标签、损失、保存和续训回归。"""

from dataclasses import replace
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from car_smart_assist.config.weather import load_weather_config
from car_smart_assist.config.weather_pyramid import (
    load_weather_pyramid_config,
    pyramid_model_config,
)
from car_smart_assist.data.weather_pyramid import (
    PyramidDataset,
    known_mask,
    prepare_pyramid_splits,
    split_pyramid_samples,
)
from car_smart_assist.perception.weather import WeatherClassifier
from car_smart_assist.perception.weather_pyramid import (
    WeatherPyramid,
    WeatherPyramidPredictor,
    masked_weather_loss,
    prepare_pyramid_image,
)
from car_smart_assist.perception.weather_pyramid_training import (
    calibrate_thresholds,
    compare_pyramid_models,
    masked_metrics,
    train_pyramid_variant,
)
from car_smart_assist.perception.weather_training import WeatherSample, checkpoint_model_config


def test_legacy_readonly_cli_evaluates_retained_asl_when_focal_was_retired(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace

    import scripts.compare_weather_models as command

    checkpoint = tmp_path / "legacy/asl/best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"retained readonly fixture")
    monkeypatch.setattr(
        command,
        "load_weather_pyramid_config",
        lambda _: SimpleNamespace(checkpoint_root=str(tmp_path / "legacy")),
    )
    seen = {}

    def compare(cfg, root, **kwargs):
        seen.update(kwargs)
        return {"models": {"asl": {}}}

    monkeypatch.setattr(command, "compare_pyramid_models", compare)
    monkeypatch.setattr(sys, "argv", ["compare_weather_models.py", "--evaluate-only"])
    assert command.main() == 0
    assert seen["evaluate_only"] is True and seen["variants"] == ("asl",)


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def tiny_config():
    cfg = load_weather_pyramid_config(
        Path(__file__).resolve().parents[2] / "configs/model/weather_pyramid.yaml"
    )
    return replace(
        cfg,
        image_size=(64, 96),
        channels=(4, 8, 8, 8, 8),
        fpn_channels=4,
        pool_grid=2,
        device="cpu",
        checkpoint_root="checkpoints",
        report_dir="reports",
        data={
            **cfg.data,
            "acdc_root": "acdc",
            "acdc_splits": ["train", "val"],
            "include_normal_references": False,
            "acdc_cleaning_manifest": "absent-acdc.json",
            "pixel_zip": "pixel.zip",
            "pixel_cleaning_manifest": "absent-pixel.json",
            "cache_dir": "cache",
        },
        train={
            **cfg.train,
            "epochs": 1,
            "min_epochs": 1,
            "resume_extra_epochs": 1,
            "patience": 4,
            "batch_size": 4,
            "amp": False,
            "force_epochs": False,
        },
        evaluation={
            **cfg.evaluation,
            "batch_size": 8,
            "benchmark_iterations": 1,
            "benchmark_warmup": 1,
            "old_config": "old.yaml",
            "old_checkpoint": "old.pt",
            "cpu_threads": 2,
        },
    )


def write_data(root, cfg):
    import io

    for condition_index, condition in enumerate(cfg.attributes):
        for group in range(4):
            folder = (
                root
                / "acdc/rgb_anon"
                / condition
                / ("train" if group < 3 else "val")
                / f"GOPR{condition_index * 10 + group:04d}"
            )
            folder.mkdir(parents=True)
            for frame in range(2):
                image = np.zeros((80, 120, 3), np.uint8)
                image[:, :, condition_index % 3] = 50 + group * 30 + frame
                Image.fromarray(image).save(folder / f"frame_{frame}_rgb_anon.png")
    with ZipFile(root / "pixel.zip", "w") as archive:
        for scene in range(1, 5):
            for lighting, condition in (
                ("day", "clear"),
                ("night", "clear"),
                ("day", "fog20"),
                ("night", "fog20"),
                ("day", "rain15"),
                ("night", "rain55"),
            ):
                stream = io.BytesIO()
                Image.fromarray(np.full((80, 120, 3), scene * 40, np.uint8)).save(
                    stream, format="PNG"
                )
                archive.writestr(f"scene{scene}_{lighting}_{condition}_0.png", stream.getvalue())


def build_sets(root, cfg):
    write_data(root, cfg)
    splits, info = prepare_pyramid_splits(cfg, root)
    return {
        name: PyramidDataset(records, cfg, root / "cache") for name, records in splits.items()
    }, info


def test_config_has_five_stages_and_requested_protocol():
    cfg = load_weather_pyramid_config(
        Path(__file__).resolve().parents[2] / "configs/model/weather_pyramid.yaml"
    )
    assert len(cfg.channels) == 5
    assert cfg.image_size == (384, 672)
    assert list(cfg.data["split_ratios"].values()) == [0.6, 0.1, 0.1, 0.2]


@pytest.mark.parametrize(
    "section,key,value",
    [
        (None, "image_size", [128, 223]),
        (None, "channels", [4, 8, 16]),
        ("loss", "attribute_weights", {"rain": 1}),
        ("data", "acdc_splits", ["train", "train"]),
        ("train", "num_workers", 1),
    ],
)
def test_invalid_configuration(tmp_path, section, key, value):
    path = Path(__file__).resolve().parents[2] / "configs/model/weather_pyramid.yaml"
    values = yaml.safe_load(path.read_text(encoding="utf-8"))
    row = values["weather_pyramid"] if section is None else values["weather_pyramid"][section]
    row[key] = value
    source = tmp_path / "bad.yaml"
    source.write_text(yaml.safe_dump(values), encoding="utf-8")
    with pytest.raises(ValueError):
        load_weather_pyramid_config(source)


def test_letterbox_preserves_shape_and_validates_input():
    image = np.full((80, 200, 3), 120, np.uint8)
    result = prepare_pyramid_image(image, (64, 96))
    assert result.shape == (64, 96, 3)
    assert not result[0].any()
    assert (result[32] == 120).all()
    with pytest.raises(ValueError):
        prepare_pyramid_image(image.astype(np.float32), (64, 96))


def test_pyramid_uses_all_five_stages_and_four_logits():
    cfg = tiny_config()
    model = WeatherPyramid(cfg)
    sizes = []
    handles = [
        stage.register_forward_hook(lambda _, __, output: sizes.append(output.shape[-2:]))
        for stage in model.stages
    ]
    result = model(torch.rand(2, 3, 64, 96))
    result.sum().backward()
    for handle in handles:
        handle.remove()
    assert result.shape == (2, 4)
    assert sizes == [(32, 48), (16, 24), (8, 12), (4, 6), (2, 3)]
    assert all(stage[0].weight.grad is not None for stage in model.stages)


def test_pipeline_predictor_rejects_uncalibrated_candidate(tmp_path):
    cfg = tiny_config()
    path = tmp_path / "uncalibrated.pt"
    torch.save(
        {
            "format_version": 3,
            "model_config": pyramid_model_config(cfg),
            "model_state": WeatherPyramid(cfg).state_dict(),
        },
        path,
    )
    with pytest.raises(ValueError, match="校准"):
        WeatherPyramidPredictor.from_checkpoint(path, cfg)


@pytest.mark.parametrize("name", ["focal", "asl"])
def test_loss_masks_unknown_labels_and_is_finite(name):
    cfg = tiny_config()
    logits = torch.tensor([[0.5, -0.7, 20.0, -20.0]], requires_grad=True)
    labels = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
    mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    value = masked_weather_loss(logits, labels, mask, name, cfg)
    value.backward()
    assert torch.isfinite(value)
    assert logits.grad[0, :2].abs().sum() > 0
    assert torch.equal(logits.grad[0, 2:], torch.zeros(2))
    assert masked_weather_loss(logits.detach(), labels, torch.zeros_like(mask), name, cfg) == 0


def test_focal_matches_binary_formula_and_asl_reference():
    cfg = tiny_config()
    labels = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
    logits, mask = torch.zeros_like(labels), torch.ones_like(labels)
    focal = masked_weather_loss(logits, labels, mask, "focal", cfg)
    assert focal.item() == pytest.approx(0.5 * 0.5**2 * np.log(2))
    asl = masked_weather_loss(logits, labels, mask, "asl", cfg)
    expected = (np.log(2) * 0.5 - np.log(0.55) * 0.45**4) / 2
    assert asl.item() == pytest.approx(expected)


def test_split_is_seeded_complete_and_group_disjoint(tmp_path):
    cfg = tiny_config()
    write_data(tmp_path, cfg)
    splits, info = prepare_pyramid_splits(cfg, tmp_path)
    repeated, again = prepare_pyramid_splits(cfg, tmp_path)
    assert info == again
    assert splits == repeated
    assert sum(map(len, splits.values())) == 56
    groups = [set(s.group for s in rows) for rows in splits.values()]
    assert all(not groups[i] & groups[j] for i in range(4) for j in range(i))
    assert all(sum(s.path is None for s in rows) == 6 for rows in splits.values())
    for rows in splits.values():
        mask = known_mask(rows)
        for known in mask:
            assert known.tolist() == [1, 1, 1, 1]
    combined = [s for rows in splits.values() for s in rows]
    changed = split_pyramid_samples(combined, cfg.data["split_ratios"], cfg.data["seed"] + 1)
    assert changed != splits


def test_partial_metrics_do_not_call_unknown_predictions_false_positives():
    labels = np.array([[0, 1, 0, 0], [0, 0, 0, 0]], np.float32)
    mask = np.array([[0, 1, 0, 0], [1, 1, 1, 1]], np.float32)
    values = np.array([[0.9, 0.9, 0.9, 0.9], [0.1, 0.1, 0.1, 0.1]])
    metrics = masked_metrics(
        labels, mask, values, tiny_config().attributes, dict.fromkeys(tiny_config().attributes, 0.5)
    )
    assert metrics["per_attribute"]["snow"]["fp"] == 0
    assert metrics["per_attribute"]["snow"]["unknown"] == 1
    assert metrics["per_attribute"]["rain"]["f1"] == 1


def test_calibration_requires_confirmed_positive_and_negative():
    class Empty:
        targets = np.ones((2, 4))
        mask = np.ones((2, 4))

    with pytest.raises(ValueError):
        calibrate_thresholds(Empty(), np.ones((2, 4)) * 0.8, tiny_config())


def test_calibration_uses_middle_of_optimal_plateau():
    class Separable:
        targets = np.array([[0] * 4, [1] * 4], np.float32)
        mask = np.ones((2, 4))

    cfg = tiny_config()
    values = np.array([[0.1] * 4, [0.9] * 4])
    thresholds = calibrate_thresholds(Separable(), values, cfg)
    assert all(t == pytest.approx(0.5) for t in thresholds.values())
    conservative = replace(cfg, evaluation={**cfg.evaluation, "threshold_tie_break": "highest"})
    assert all(
        t == pytest.approx(0.9)
        for t in calibrate_thresholds(Separable(), values, conservative).values()
    )


def test_cache_changes_when_resolution_changes(tmp_path):
    cfg = tiny_config()
    path = tmp_path / "raw.png"
    Image.fromarray(np.full((80, 120, 3), 100, np.uint8)).save(path)
    sample = WeatherSample((0, 1, 0, 0), "acdc:one", "rain", path=path)
    first = PyramidDataset([sample], cfg, tmp_path / "cache")
    second = PyramidDataset([sample], replace(cfg, image_size=(96, 128)), tmp_path / "cache")
    assert first.signature != second.signature
    assert first.images.shape == (1, 64, 96, 3)
    assert second.images.shape == (1, 96, 128, 3)


def test_normal_reference_pairs_and_duplicate_files_stay_together(tmp_path):
    import shutil

    cfg = tiny_config()
    cfg = replace(cfg, data={**cfg.data, "include_normal_references": True})
    write_data(tmp_path, cfg)
    references = []
    for condition_index, condition in enumerate(cfg.attributes):
        for group in range(4):
            split = "train" if group < 3 else "val"
            folder = (
                tmp_path
                / "acdc/rgb_anon"
                / condition
                / f"{split}_ref"
                / f"GOPR{condition_index * 10 + group:04d}"
            )
            folder.mkdir(parents=True)
            for frame in range(2):
                path = folder / f"frame_{frame}_rgb_ref_anon.png"
                Image.fromarray(
                    np.full((80, 120, 3), 10 + condition_index * 40 + group * 3 + frame, np.uint8)
                ).save(path)
                references.append(path)
    shutil.copy2(references[0], references[16])
    splits, info = prepare_pyramid_splits(cfg, tmp_path)
    assert info["total_samples"] == 88
    assert info["normal_references"] == 32
    assert info["cross_recording_duplicate_links"]
    membership = {
        str(s.path): name for name, records in splits.items() for s in records if s.path is not None
    }
    assert membership[str(references[0])] == membership[str(references[16])]
    for records in splits.values():
        groups = {
            s.group for s in records if s.path is not None and not s.stratum.startswith("normal:")
        }
        assert all(
            s.group in groups and not any(s.labels)
            for s in records
            if s.stratum.startswith("normal:")
        )


@pytest.mark.parametrize("name", ["focal", "asl"])
def test_training_checkpoint_predict_and_resume_matches_continuous(tmp_path, name):
    cfg = tiny_config()
    sets, info = build_sets(tmp_path, cfg)
    continuous_root = tmp_path / "continuous"
    first, _ = train_pyramid_variant(cfg, tmp_path, sets, info, name, fresh=True)
    saved = tmp_path / cfg.checkpoint_root / name / "best.pt"
    predictor = WeatherPyramidPredictor.from_checkpoint(saved, cfg)
    prediction = predictor.predict(np.full((80, 120, 3), 80, np.uint8))
    assert set(prediction.probabilities) == set(cfg.attributes)
    train_pyramid_variant(cfg, tmp_path, sets, info, name)
    train_pyramid_variant(
        replace(cfg, train={**cfg.train, "epochs": 2}),
        continuous_root,
        sets,
        info,
        name,
        fresh=True,
    )
    resumed = torch.load(tmp_path / cfg.checkpoint_root / name / "last.pt", weights_only=True)
    continuous = torch.load(
        continuous_root / cfg.checkpoint_root / name / "last.pt", weights_only=True
    )
    assert resumed["epoch"] == 2
    assert all(
        torch.equal(resumed["model_state"][k], v) for k, v in continuous["model_state"].items()
    )
    bad_cfg = replace(cfg, fpn_channels=cfg.fpn_channels + 1)
    with pytest.raises(ValueError):
        WeatherPyramidPredictor.from_checkpoint(saved, bad_cfg)
    assert pyramid_model_config(cfg)["channels"] == [4, 8, 8, 8, 8]


def test_full_comparison_uses_same_initialization_and_preserves_original(tmp_path):
    cfg = tiny_config()
    write_data(tmp_path, cfg)
    legacy_path = Path(__file__).resolve().parents[2] / "configs/model/weather_classifier.yaml"
    old_values = yaml.safe_load(legacy_path.read_text(encoding="utf-8"))
    old_values["weather_classifier"].update(image_size=[32, 48], channels=[4, 8, 8], device="cpu")
    (tmp_path / "old.yaml").write_text(yaml.safe_dump(old_values), encoding="utf-8")
    old_cfg = load_weather_config(tmp_path / "old.yaml")
    old = WeatherClassifier(old_cfg)
    torch.save(
        {
            "format_version": 2,
            "model_config": checkpoint_model_config(old_cfg),
            "model_state": old.state_dict(),
        },
        tmp_path / "old.pt",
    )
    original = (tmp_path / "old.pt").read_bytes()
    result = compare_pyramid_models(cfg, tmp_path, fresh=True)
    assert (
        result["models"]["focal"]["initial_state_sha256"]
        == result["models"]["asl"]["initial_state_sha256"]
    )
    assert result["splits"]["test"]["samples"] == 14
    assert result["default_pipeline_promoted"] is False
    assert result["official_test_protocol"] is False
    assert (tmp_path / "old.pt").read_bytes() == original
    assert (tmp_path / "reports/test_predictions.csv").exists()
    last = tmp_path / "checkpoints/focal/last.pt"
    before = last.read_bytes()
    evaluated = compare_pyramid_models(cfg, tmp_path, evaluate_only=True)
    assert evaluated["evaluation_only"] is True
    assert last.read_bytes() == before
    assert evaluated["models"]["focal"]["test"] == result["models"]["focal"]["test"]


def test_new_candidate_is_explicitly_selectable_in_pipeline(tmp_path):
    from dataclasses import asdict

    from car_smart_assist.inference.pipeline import InferencePipeline

    cfg = tiny_config()
    sets, info = build_sets(tmp_path, cfg)
    train_pyramid_variant(cfg, tmp_path, sets, info, "asl", fresh=True)
    source = tmp_path / "new.yaml"
    source.write_text(yaml.safe_dump({"weather_pyramid": asdict(cfg)}), encoding="utf-8")
    pipeline = InferencePipeline.from_config(
        {},
        project_root=tmp_path,
        require_visibility=False,
        weather_config=source,
        weather_loss="asl",
        device="cpu",
    )
    assert isinstance(pipeline.weather_predictor, WeatherPyramidPredictor)
    result = pipeline.run(np.full((80, 120, 3), 100, np.uint8))
    assert result.weather is not None
    assert set(result.weather.probabilities) == set(cfg.attributes)
    with pytest.raises(FileNotFoundError):
        InferencePipeline.from_config(
            {},
            project_root=tmp_path,
            require_visibility=False,
            weather_config=source,
            weather_checkpoint="missing.pt",
        )


def test_requested_total_epochs_overrides_early_stopping_and_resumes(tmp_path):
    cfg = tiny_config()
    cfg = replace(cfg, train={**cfg.train, "patience": 1})
    sets, info = build_sets(tmp_path, cfg)
    train_pyramid_variant(cfg, tmp_path, sets, info, "focal", fresh=True)
    train_pyramid_variant(cfg, tmp_path, sets, info, "focal", total_epochs=4)
    last = tmp_path / cfg.checkpoint_root / "focal/last.pt"
    payload = torch.load(last, weights_only=True)
    assert payload["epoch"] == payload["planned_epochs"] == 4
    assert payload["force_epochs"] is True
    assert len(payload["history"]) == 4
    before = payload["model_state"]
    train_pyramid_variant(cfg, tmp_path, sets, info, "focal", total_epochs=4)
    repeated = torch.load(last, weights_only=True)
    assert repeated["epoch"] == 4
    assert all(torch.equal(value, repeated["model_state"][key]) for key, value in before.items())
    with pytest.raises(ValueError, match="累计"):
        train_pyramid_variant(cfg, tmp_path, sets, info, "focal", total_epochs=3)
