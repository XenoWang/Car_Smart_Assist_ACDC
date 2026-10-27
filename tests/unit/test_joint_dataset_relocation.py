"""Read-only dataset relocation, supervision validity and empty-output identity."""

import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest
import yaml
from PIL import Image

from _model_test_helpers import load_module, snapshot
from car_smart_assist.data.detection import (
    _empty_output_identity, _publish_detection_staging, validate_detection_export,
)
from car_smart_assist.data.detection_joint import prepare_joint_detection
from car_smart_assist.data.joint_paths import detection_data_scope, read_detection_manifest


def _joint(root):
    fixtures = load_module("tests/unit/test_data_preparation_and_policy_guards.py", "joint_content_fixtures")
    _, cfg = fixtures._joint_fixture(root)
    return cfg


@pytest.mark.parametrize("kind", ["acdc", "joint"])
def test_original_empty_output_can_be_published_without_changing_source_files(tmp_path, kind):
    if kind == "joint":
        cfg = _joint(tmp_path)
        prepare, expected = prepare_joint_detection, 4
    else:
        fixtures = load_module("tests/unit/test_dataset_export_and_split_contracts.py", "acdc_content_fixtures")
        cfg = fixtures._full_detection_fixture(tmp_path)
        from car_smart_assist.data.detection import prepare_detection_data
        prepare, expected = prepare_detection_data, 32
    output = tmp_path / cfg["output_root"]
    output.mkdir(parents=True)
    before = snapshot(tmp_path / "data")
    result = prepare(tmp_path, cfg)
    assert (output / "data.yaml").is_file()
    if kind == "acdc":
        assert result["dataset_yaml"] == str(output / "data.yaml")
    assert sum(validate_detection_export(output, verify_hashes=True).values()) == expected
    after = snapshot(tmp_path / "data")
    assert all(after[path] == value for path, value in before.items())


@pytest.mark.parametrize("replacement", [False, True])
def test_publish_preserves_changed_or_replaced_original_empty_directory(tmp_path, replacement):
    output, source = tmp_path / "output", tmp_path / ".output.staging-safe"
    output.mkdir()
    identity = _empty_output_identity(output)
    source.mkdir()
    (source / "prepared.bin").write_bytes(b"prepared export")
    if replacement:
        output.rename(tmp_path / "original-preserved")
        output.mkdir()
    else:
        (output / "user.bin").write_bytes(b"new user content")
    before = snapshot(tmp_path)
    with pytest.raises((ValueError, OSError)):
        _publish_detection_staging(source, output, empty_identity=identity)
    assert output.is_dir()
    assert snapshot(tmp_path) == before
    assert (source / "prepared.bin").read_bytes() == b"prepared export"


@pytest.mark.parametrize("value", [None, [], 7, "not a manifest"])
def test_valid_json_with_invalid_manifest_structure_is_an_input_error(tmp_path, value):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    before = path.read_bytes()
    with pytest.raises(ValueError, match="manifest.*object"):
        validate_detection_export(tmp_path, verify_hashes=True)
    with pytest.raises(ValueError, match="manifest.*object"):
        read_detection_manifest(tmp_path, "data.yaml", verify_hashes=True)
    assert path.read_bytes() == before


@pytest.mark.parametrize("key", ["roi_mask", "roi_sha256", "instance_sha256"])
def test_lost_and_found_rows_cannot_silently_drop_roi_or_hashes(tmp_path, key):
    cfg = _joint(tmp_path)
    prepare_joint_detection(tmp_path, cfg)
    output = tmp_path / cfg["output_root"]
    path = output / "manifest.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    row = next(row for row in value["images"] if row["dataset"] == "lost_and_found")
    row.pop(key)
    path.write_text(json.dumps(value), encoding="utf-8")
    before = snapshot(tmp_path)
    with pytest.raises(ValueError, match=key):
        validate_detection_export(output, verify_hashes=True)
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize("field", ["name", "source_id"])
def test_joint_class_identity_is_unique_before_publication_and_on_read(tmp_path, field):
    cfg = _joint(tmp_path)
    prepare_joint_detection(tmp_path, cfg)
    output = tmp_path / cfg["output_root"]
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["classes"][1][field] = manifest["classes"][0][field]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    before = snapshot(tmp_path)
    with pytest.raises(ValueError, match="unique"):
        validate_detection_export(output, verify_hashes=True)
    assert snapshot(tmp_path) == before
    cfg["additional_class"][field] = manifest["classes"][0][field]
    cfg["output_root"] = "data/processed/another-export"
    with pytest.raises(ValueError, match="unique"):
        prepare_joint_detection(tmp_path, cfg)
    assert not (tmp_path / cfg["output_root"]).exists()


@pytest.mark.parametrize("tamper", [None, "weights", "split", "boxes", "roi", "image", "rows"])
def test_explicit_relocation_preserves_identity_only_for_equivalent_data(tmp_path, tamper):
    original = tmp_path / "original"
    cfg = _joint(original)
    prepare_joint_detection(original, cfg)
    output = original / cfg["output_root"]
    identity = json.loads((output / "manifest.json").read_text(encoding="utf-8"))["signature"]
    original_before = snapshot(original)
    moved = tmp_path / "moved"
    shutil.copytree(original, moved)
    (moved / "configs").mkdir()
    config_path = moved / "configs/joint.yaml"
    current = json.loads(json.dumps(cfg))
    manifest_path = moved / cfg["output_root"] / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if tamper == "weights":
        current["source_weights"] = {"acdc": 0.7, "lost_and_found": 0.3}
    elif tamper in ("split", "boxes"):
        manifest["images"][0][tamper] = "test" if tamper == "split" else [[0, 0.5, 0.5, 0.2, 0.2]]
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    elif tamper == "rows":
        manifest["images"] = [None]
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    elif tamper in ("roi", "image"):
        row = next(row for row in manifest["images"] if row["dataset"] == "lost_and_found")
        relative = Path(*Path(row["source_relative"]).parts[1:])
        if tamper == "image":
            path = moved / cfg["lost_and_found_root"] / "leftImg8bit" / relative
            Image.new("RGB", (4, 4), (99, 99, 99)).save(path)
        else:
            stem = relative.name.removesuffix("_leftImg8bit.png")
            path = moved / cfg["lost_and_found_root"] / "gtCoarse" / relative.parent / (stem + "_gtCoarse_labelTrainIds.png")
            Image.new("L", (4, 4), 0).save(path)
    config_path.write_text(yaml.safe_dump({"joint_detection_data": current}, sort_keys=False), encoding="utf-8")
    serialized = yaml.safe_load(config_path.read_text(encoding="utf-8"))["joint_detection_data"]
    assert tuple(serialized["development_ratios"]) == ("train", "val", "calibration")
    before = snapshot(moved)
    data = Path(cfg["output_root"]) / "data.yaml"
    if tamper is not None:
        reason = "images object list" if tamper == "rows" else "current content, splits or supervision"
        with pytest.raises(ValueError, match=reason):
            read_detection_manifest(moved, data, "configs/joint.yaml", verify_hashes=True)
    else:
        with pytest.raises(ValueError, match="explicit --joint-data-config"):
            read_detection_manifest(moved, data, verify_hashes=True)
        manifest, counts, mapping = read_detection_manifest(moved, data, "configs/joint.yaml", verify_hashes=True)
        assert manifest["signature"] == identity
        assert sum(counts.values()) == 4
        assert mapping == str((moved / cfg["lost_and_found_root"]).resolve())
        work = tmp_path / "run"
        work.mkdir()
        with detection_data_scope(moved, data, joint_data_config="configs/joint.yaml", directory=work) as (sidecar, resolved, _counts):
            assert sidecar.is_file() and resolved["signature"] == identity
            document = yaml.safe_load(sidecar.read_text(encoding="utf-8"))
            assert document["path"] == str((moved / cfg["output_root"]).resolve())
            from car_smart_assist.perception.joint_yolo_backend import load_joint_index
            indexed, index = load_joint_index(document)
            assert indexed["signature"] == identity and len(index) == 4
            for row in indexed["images"]:
                if row["dataset"] == "lost_and_found":
                    assert Path(row["roi_mask"]).is_relative_to(moved)
        assert not sidecar.exists() and not list(work.iterdir())
        # This is the selected path-rebasing metamorphic contract, not a new ID.
        result = prepare_joint_detection(moved, current)
        assert result["signature"] == identity
    assert snapshot(moved) == before
    assert snapshot(original) == original_before


def test_resume_uses_current_validated_data_even_if_vendor_restores_old_path(tmp_path, monkeypatch):
    monkeypatch.setenv("YOLO_CONFIG_DIR", str(tmp_path / "settings"))
    from ultralytics.models.yolo.detect import DetectionTrainer
    from car_smart_assist.perception.yolo_training import detection_trainer_for_resume

    def old_saved_args(self, _overrides):
        self.resume = True
        self.args = SimpleNamespace(data="obsolete/data.yaml")

    monkeypatch.setattr(DetectionTrainer, "check_resume", old_saved_args)
    selected = tmp_path / "current/data.yaml"
    trainer_type = detection_trainer_for_resume(None, {}, validated_data=selected)
    trainer = trainer_type.__new__(trainer_type)
    trainer.check_resume({"epochs": 7})
    assert trainer.args.epochs == 7 and trainer.args.data == str(selected)


def test_explicit_mapping_preserves_nested_location_subdirectories(tmp_path):
    from car_smart_assist.data.joint_paths import lost_and_found_paths

    row = {"source_relative": "lost_and_found/train/site/nested/frame_leftImg8bit.png"}
    source, roi, instances = lost_and_found_paths(row, tmp_path)
    assert source == tmp_path / "leftImg8bit/train/site/nested/frame_leftImg8bit.png"
    assert roi == tmp_path / "gtCoarse/train/site/nested/frame_gtCoarse_labelTrainIds.png"
    assert instances == tmp_path / "gtCoarse/train/site/nested/frame_gtCoarse_instanceIds.png"


@pytest.mark.parametrize("relative", ["lost_and_found/train/site/../frame_leftImg8bit.png",
                                      "C:/train/site/frame_leftImg8bit.png",
                                      "lost_and_found/train/site/frame.txt"])
def test_explicit_mapping_rejects_ambiguous_or_escaping_paths(tmp_path, relative):
    from car_smart_assist.data.joint_paths import lost_and_found_paths

    with pytest.raises(ValueError, match="source_relative"):
        lost_and_found_paths({"source_relative": relative}, tmp_path)
