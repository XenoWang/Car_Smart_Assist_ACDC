"""Focused synthetic regressions for PRD V0.0.4 data and policy fixes."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest
import yaml
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]


def _load_helper(relative: str, name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _joint_fixture(root: Path):
    helper = _load_helper("tests/unit/test_p2_pipeline_followup.py", "prd004_joint_fixture")
    helper._write_base_export(root)
    helper._write_laf_fixture(root)
    return helper, helper._joint_config()


def test_joint_staging_interruption_keeps_final_absent_then_reuses_published_bytes(
    tmp_path, monkeypatch
):
    import car_smart_assist.data.detection_joint as joint

    _, cfg = _joint_fixture(tmp_path)
    output = tmp_path / cfg["output_root"]
    source_bytes = {
        path: path.read_bytes()
        for path in (tmp_path / "data/raw").rglob("*")
        if path.is_file()
    }
    label_text = joint._label_text

    def interrupt_label_write(_boxes):
        raise OSError("synthetic label write interruption")

    monkeypatch.setattr(joint, "_label_text", interrupt_label_write)
    with pytest.raises(OSError, match="synthetic label write interruption"):
        joint.prepare_joint_detection(tmp_path, cfg)
    assert not output.exists()
    assert not list(output.parent.glob(f".{output.name}.staging-*"))
    assert all(path.read_bytes() == value for path, value in source_bytes.items())

    monkeypatch.setattr(joint, "_label_text", label_text)
    first = joint.prepare_joint_detection(tmp_path, cfg)
    published = {
        path.relative_to(output): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in output.rglob("*")
        if path.is_file()
    }
    second = joint.prepare_joint_detection(tmp_path, cfg)
    reused = {
        path.relative_to(output): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in output.rglob("*")
        if path.is_file()
    }
    assert first["signature"] == second["signature"]
    assert reused == published
    data_yaml = yaml.safe_load((output / "data.yaml").read_text(encoding="utf-8"))
    assert data_yaml["path"] == output.resolve().as_posix()
    assert Path(data_yaml["joint_manifest"]) == output / "manifest.json"
    assert not list(output.parent.glob(f".{output.name}.staging-*"))
    assert all(path.read_bytes() == value for path, value in source_bytes.items())


@pytest.mark.parametrize(
    "weight_key,weights",
    [
        ("source_weights", {"acdc": 1.0}),
        ("source_weights", {"acdc": 0.8, "lost_and_found": 0.2, "extra": 0.1}),
        ("classification_loss_weights", {"lost_and_found": 1.0}),
        (
            "classification_loss_weights",
            {"acdc": 1.0, "lost_and_found": 1.0, "extra": 1.0},
        ),
    ],
)
def test_joint_source_weight_maps_require_exact_source_keys(tmp_path, weight_key, weights):
    from car_smart_assist.data.detection_joint import prepare_joint_detection

    helper = _load_helper("tests/unit/test_p2_pipeline_followup.py", "prd004_joint_key_fixture")
    cfg = helper._joint_config()
    cfg[weight_key] = weights
    with pytest.raises(ValueError, match=weight_key):
        prepare_joint_detection(tmp_path, cfg)
    assert not (tmp_path / cfg["output_root"]).exists()


def test_missing_acdc_mask_invalidates_mask_path_without_invalidating_rgb(tmp_path):
    from car_smart_assist.data.preprocessing import CleaningReport, SampleStatus, check_acdc

    acdc = tmp_path / "acdc"
    image = acdc / "rgb_anon/fog/train/scene/frame_rgb_anon.png"
    image.parent.mkdir(parents=True)
    Image.fromarray(np.full((12, 16, 3), 64, np.uint8)).save(image)
    report = CleaningReport()
    check_acdc(
        acdc,
        report,
        {"integrity": {"full_decode": True, "force_load": True}},
        {
            "image_integrity": True,
            "image_statistics": False,
            "acdc_image_mask_pairing": True,
            "acdc_mask_validity": True,
            "detection_annotations": False,
        },
        num_workers=1,
    )
    missing_label = acdc / "gt/fog/train/scene/frame_gt_labelIds.png"
    assert report.status_of(image) is SampleStatus.VALID
    assert str(image) not in report.invalid_paths()
    assert report.status_of(missing_label) is SampleStatus.INVALID
    assert str(missing_label) in report.invalid_paths()


def test_weather_pyramid_requires_configured_cleaning_manifest_but_accepts_empty_one(tmp_path):
    from car_smart_assist.config.weather_pyramid import load_weather_pyramid_config
    from car_smart_assist.data.weather_pyramid import collect_pyramid_samples

    config_path = ROOT / "configs/model/weather_pyramid.yaml"
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw["weather_pyramid"]["data"].update(
        acdc_root="acdc",
        acdc_splits=["train"],
        include_normal_references=False,
        acdc_cleaning_manifest="data/cleaning/acdc.json",
        pixel_zip="data/pixel.zip",
        pixel_cleaning_manifest="data/cleaning/pixel.json",
    )
    local_config = tmp_path / "weather.yaml"
    local_config.write_text(yaml.safe_dump(raw), encoding="utf-8")
    acdc = tmp_path / "acdc/rgb_anon/fog/train"
    acdc.mkdir(parents=True)
    Image.fromarray(np.full((8, 8, 3), 80, np.uint8)).save(acdc / "frame_rgb_anon.png")
    with pytest.raises(FileNotFoundError, match="清洗清单不存在"):
        collect_pyramid_samples(load_weather_pyramid_config(local_config), tmp_path)

    empty_zip = tmp_path / "data/pixel.zip"
    empty_zip.parent.mkdir(parents=True, exist_ok=True)
    import io

    with ZipFile(empty_zip, "w") as archive:
        for scene in (1, 4):
            image = io.BytesIO()
            Image.fromarray(np.full((8, 8, 3), scene * 40, np.uint8)).save(image, format="PNG")
            archive.writestr(f"scene{scene}_day_clear_0.png", image.getvalue())
    (tmp_path / "data/cleaning").mkdir(parents=True)
    (tmp_path / "data/cleaning/acdc.json").write_text('{"invalid": []}', encoding="utf-8")
    (tmp_path / "data/cleaning/pixel.json").write_text('{"invalid": []}', encoding="utf-8")
    samples = collect_pyramid_samples(load_weather_pyramid_config(local_config), tmp_path)
    acdc_samples = [sample for sample in samples if sample.path is not None]
    pixel_samples = [sample for sample in samples if sample.archive_path is not None]
    assert len(acdc_samples) == 1
    assert acdc_samples[0].path == acdc / "frame_rgb_anon.png"
    assert len(pixel_samples) == 2
    assert {sample.member for sample in pixel_samples} == {
        "scene1_day_clear_0.png", "scene4_day_clear_0.png"
    }


def test_policy_weather_handover_conditions_must_have_risk_mappings():
    from car_smart_assist.config.policy import PolicyConfig

    raw = yaml.safe_load((ROOT / "configs/model/advisory_llm.yaml").read_text(encoding="utf-8"))
    cfg = dict(raw["advisory"]["policy"])
    cfg["weather_risk_levels"] = dict(cfg["weather_risk_levels"])
    cfg["weather_risk_multiplier"] = dict(cfg["weather_risk_multiplier"])
    del cfg["weather_risk_levels"]["fog"]
    del cfg["weather_risk_multiplier"]["fog"]
    with pytest.raises(ValueError, match="weather_handover_conditions"):
        PolicyConfig.from_mapping(cfg)


@pytest.mark.parametrize("kind", ["fog", "darkness", "blur", "occlusion"])
def test_zero_degradation_returns_an_independent_identical_image(kind):
    from car_smart_assist.perception.visibility.degradation import degrade

    image = np.arange(18 * 24 * 3, dtype=np.uint8).reshape(18, 24, 3)
    result, spec = degrade(image, kind, 0.0, seed=9)
    assert np.array_equal(result, image)
    assert result is not image
    assert spec.severity == 0.0


def test_positive_blur_keeps_preexisting_sigma_curve():
    from car_smart_assist.perception.visibility.degradation import add_blur

    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[24:40, 24:40] = 255
    result, spec = add_blur(image, 0.4, np.random.default_rng(9))
    assert spec.params["sigma"] == pytest.approx(0.5 + (41 / 6.0 - 0.5) * 0.4)
    assert not np.array_equal(result, image)
