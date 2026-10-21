from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/run_pipeline.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("run_pipeline_demo_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _Pipeline:
    def __init__(self):
        self.images = None

    def run_batch(self, images):
        self.images = list(images)
        return [type("Result", (), {"to_dict": lambda self: {}})() for _ in images]


def test_explicit_image_skips_cleaning_manifest_enumeration(monkeypatch, tmp_path):
    module = _load_script()
    image = tmp_path / "explicit.png"
    image.write_bytes(b"user supplied image path")
    pipeline = _Pipeline()
    monkeypatch.setattr(module, "parse_args", lambda: type("Args", (), {
        "log_level": "ERROR", "config": "configs/model/visibility.yaml",
        "checkpoint": None, "device": None, "weather_config": None,
        "weather_loss": "focal", "weather_checkpoint": None,
        "image": [str(image)], "synthetic": False, "json": True,
    })())
    monkeypatch.setattr(module.InferencePipeline, "from_config", lambda *a, **k: pipeline)
    monkeypatch.setattr(module, "list_ref_images", lambda *a, **k: (_ for _ in ()).throw(AssertionError("enumerated refs")))
    monkeypatch.setattr(module, "list_adverse_images", lambda *a, **k: (_ for _ in ()).throw(AssertionError("enumerated adverse")))

    assert module.main() == 0
    assert pipeline.images[0] == image
    assert len(pipeline.images) == 2  # Explicit input plus the built-in black-frame control.


def test_auto_samples_apply_shared_cleaning_manifest_and_keep_raw(tmp_path):
    import copy
    import json
    from PIL import Image

    module = _load_script()
    acdc = tmp_path / "data/raw/acdc/rgb_anon"
    paths = []
    invalid = []
    reference = acdc / "fog/train_ref/seq/1keep_rgb_ref_anon.png"
    suspect = reference.with_name("2suspect_rgb_ref_anon.png")
    for path in (reference, suspect):
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 8), color=(100, 120, 140)).save(path)
        paths.append(path)
    bad_ref = reference.with_name("0bad_rgb_ref_anon.png")
    bad_ref.write_bytes(b"broken image retained")
    paths.append(bad_ref)
    invalid.append(bad_ref)
    expected = [reference]
    for condition in ("fog", "night", "rain", "snow"):
        path = acdc / condition / "train/seq/1keep_rgb_anon.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 8), color=(100, 120, 140)).save(path)
        bad = path.with_name("0bad_rgb_anon.png")
        bad.write_bytes(b"broken image retained")
        paths.extend((path, bad))
        invalid.append(bad)
        expected.append(path)
    manifest = tmp_path / "data/processed/manifests/acdc.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({
        "invalid": [str(path.relative_to(tmp_path)) for path in invalid],
        "suspect": [str(suspect.relative_to(tmp_path))],
    }), encoding="utf-8")
    cfg = {"data": {"acdc_root": "data/raw/acdc", "cleaning_manifest": "data/processed/manifests/acdc.json"}}
    before_cfg = copy.deepcopy(cfg)
    before = {path: path.read_bytes() for path in paths}
    samples = module.build_samples(tmp_path, cfg)
    assert [path for _, path in samples] == expected
    assert cfg == before_cfg
    assert {path: path.read_bytes() for path in paths} == before
    assert suspect.exists()


def test_auto_samples_missing_manifest_reports_cleaning_command_even_without_images(tmp_path):
    import pytest

    module = _load_script()
    with pytest.raises(FileNotFoundError) as error:
        module.build_samples(tmp_path, {"data": {"cleaning_manifest": "missing.json"}})
    message = str(error.value)
    assert str(tmp_path) in message and "missing.json" in message
    assert "先进行数据清洗" in message
    assert r"scripts\clean_data.py --only acdc --corrupt-images-only" in message
