from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]


def _helper(relative, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _full_detection_fixture(root):
    helper = _helper("tests/unit/test_detection_data.py", "remaining_detection_fixtures")
    cfg, _ = helper.write_detection_fixture(root)
    cfg["report_root"] = "artifacts/reports/remaining_data"
    cfg["image_mode"] = "copy"
    return cfg


def test_detection_export_failure_cleans_staging_and_does_not_publish(tmp_path, monkeypatch):
    from car_smart_assist.data import detection

    cfg = _full_detection_fixture(tmp_path)
    original = detection.validate_detection_export

    def fail_staging(path, verify_hashes=False):
        if ".staging-" in path.name:
            raise ValueError("injected final validation failure")
        return original(path, verify_hashes=verify_hashes)

    monkeypatch.setattr(detection, "validate_detection_export", fail_staging)
    with pytest.raises(ValueError, match="injected final validation failure"):
        detection.prepare_detection_data(tmp_path, cfg)
    output = tmp_path / cfg["output_root"]
    assert not output.exists()
    assert not list(output.parent.glob(f".{output.name}.staging-*"))


def test_detection_export_reuses_valid_output_without_rewriting_and_yaml_uses_final_path(tmp_path):
    from car_smart_assist.data.detection import prepare_detection_data

    cfg = _full_detection_fixture(tmp_path)
    result = prepare_detection_data(tmp_path, cfg)
    output = tmp_path / cfg["output_root"]
    before = {p.relative_to(output): (p.read_bytes(), p.stat().st_mtime_ns)
              for p in output.rglob("*") if p.is_file()}
    assert result["dataset_yaml"] == str(output.resolve() / "data.yaml")
    prepare_detection_data(tmp_path, cfg)
    after = {p.relative_to(output): (p.read_bytes(), p.stat().st_mtime_ns)
             for p in output.rglob("*") if p.is_file()}
    assert after == before
    import yaml
    document = yaml.safe_load((output / "data.yaml").read_text(encoding="utf-8"))
    assert document["path"] == output.resolve().as_posix()


@pytest.mark.parametrize("damage", ["json", "yaml", "image", "official", "partial"])
def test_detection_export_rejects_damaged_nonempty_output_without_overwrite(tmp_path, damage):
    from car_smart_assist.data.detection import prepare_detection_data

    cfg = _full_detection_fixture(tmp_path)
    prepare_detection_data(tmp_path, cfg)
    output = tmp_path / cfg["output_root"]
    if damage == "json":
        target = output / "manifest.json"
        target.write_text("{broken", encoding="utf-8")
    elif damage == "yaml":
        target = output / "data.yaml"
        target.write_text("path: [broken", encoding="utf-8")
    elif damage == "image":
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        target = output / manifest["images"][0]["image"]
        target.write_bytes(b"changed exported image")
    elif damage == "partial":
        target = output / "train.txt"
        target.unlink()
    else:
        target = output / "official_inference.txt"
        target.write_text("different official list\n", encoding="utf-8")
    snapshot = {p.relative_to(output): p.read_bytes() for p in output.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="Existing detection export"):
        prepare_detection_data(tmp_path, cfg)
    assert {p.relative_to(output): p.read_bytes() for p in output.rglob("*") if p.is_file()} == snapshot


def test_official_index_bad_paths_and_resolved_duplicates_fail_before_output(tmp_path):
    from car_smart_assist.data import detection

    for invalid in (
        "../outside.png", "nested/../outside.png", r"nested\\..\\outside.png",
        r"C:\\outside.png", "/rooted/outside.png",
    ):
        case = tmp_path / hashlib.sha256(invalid.encode()).hexdigest()[:8]
        cfg = _full_detection_fixture(case)
        index = case / cfg["acdc_root"] / cfg["official_test_index"]
        index.write_text(json.dumps({"images": [{"file_name": invalid}], "annotations": []}), encoding="utf-8")
        with pytest.raises(ValueError, match="Invalid official inference image path"):
            detection.prepare_detection_data(case, cfg)
        assert not (case / cfg["output_root"]).exists()


def test_official_index_rejects_duplicate_resolved_file_before_output(tmp_path):
    from car_smart_assist.data import detection

    cfg = _full_detection_fixture(tmp_path)
    index = tmp_path / cfg["acdc_root"] / cfg["official_test_index"]
    valid = json.loads(index.read_text(encoding="utf-8"))["images"][0]["file_name"]
    index.write_text(json.dumps({"images": [{"file_name": valid}, {"file_name": valid}],
                                 "annotations": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate official inference image"):
        detection.prepare_detection_data(tmp_path, cfg)
    assert not (tmp_path / cfg["output_root"]).exists()


@pytest.mark.parametrize("failure", ["annotations", "missing"])
def test_official_index_prevalidation_fails_without_publishing(tmp_path, failure):
    from car_smart_assist.data import detection

    cfg = _full_detection_fixture(tmp_path)
    acdc = tmp_path / cfg["acdc_root"]
    index = acdc / cfg["official_test_index"]
    if failure == "annotations":
        index.write_text(json.dumps({"images": [], "annotations": [{"id": 1}]}), encoding="utf-8")
        expected = "Official inference-only index"
    else:
        index.write_text(json.dumps({"images": [{"file_name": "fog/test/missing.png"}],
                                     "annotations": []}), encoding="utf-8")
        expected = "Official inference image missing or outside"
    with pytest.raises(ValueError, match=expected):
        detection.prepare_detection_data(tmp_path, cfg)
    assert not (tmp_path / cfg["output_root"]).exists()


def test_official_index_rejects_symlink_escape_when_supported(tmp_path):
    from car_smart_assist.data import detection

    cfg = _full_detection_fixture(tmp_path)
    acdc = tmp_path / cfg["acdc_root"]
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"outside image sentinel")
    link = acdc / "rgb_anon/fog/test/GOPR9999/link.png"
    link.parent.mkdir(parents=True, exist_ok=True)
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"Windows symlink creation unavailable: {exc}")
    index = acdc / cfg["official_test_index"]
    index.write_text(json.dumps({"images": [{"file_name": "fog/test/GOPR9999/link.png"}],
                                 "annotations": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="missing or outside"):
        detection.prepare_detection_data(tmp_path, cfg)
    assert not (tmp_path / cfg["output_root"]).exists()


def test_preprocessing_detection_json_path_schema_continues_to_next_document(tmp_path, monkeypatch):
    from car_smart_assist.data.preprocessing import CleaningReport, _check_detection_jsons

    det = tmp_path / "gt_detection"
    rgb = tmp_path / "rgb_anon"
    det.mkdir(); rgb.mkdir()
    outside = tmp_path / "outside-but-not-rgb.png"
    outside.write_bytes(b"outside sentinel")
    bad = {"images": [{"id": 1, "file_name": str(outside), "width": 10, "height": 10}],
           "annotations": [], "categories": [{"id": 1}]}
    good = {"images": [{"id": 2, "file_name": "fog/train/seq/frame.png", "width": 10, "height": 10}],
            "annotations": [], "categories": [{"id": 1}]}
    valid_image = rgb / "fog/train/seq/frame.png"
    valid_image.parent.mkdir(parents=True)
    valid_image.write_bytes(b"valid relative image sentinel")
    (det / "a_gt_detection.json").write_text("{broken", encoding="utf-8")
    (det / "b_gt_detection.json").write_text(json.dumps(bad), encoding="utf-8")
    (det / "c_gt_detection.json").write_text(json.dumps(good), encoding="utf-8")
    original_exists = Path.exists

    def forbid_outside_exists(path):
        if path.resolve() == outside.resolve():
            pytest.fail("checker called Path.exists on an absolute path outside rgb_anon")
        return original_exists(path)

    monkeypatch.setattr(Path, "exists", forbid_outside_exists)
    report = CleaningReport()
    _check_detection_jsons(det, rgb, report, {"detection": {}})
    schema = [item for item in report.issues if item.check == "detection/schema"]
    assert len(schema) == 1 and "相对路径" in schema[0].message
    assert sum(item.check == "detection/json" for item in report.issues) == 1
    assert outside.read_bytes() == b"outside sentinel"
    assert len(report.stats["detection_files"]) == 1


def test_detection_json_rejects_empty_absolute_drive_and_parent_paths(tmp_path):
    from car_smart_assist.data.preprocessing import CleaningReport, _check_detection_jsons

    cases = ["", "/absolute.png", r"C:\\absolute.png", "../escape.png", r"a\\..\\escape.png"]
    for index, file_name in enumerate(cases):
        det = tmp_path / str(index); det.mkdir()
        document = {"images": [{"id": 1, "file_name": file_name, "width": 10, "height": 10}],
                    "annotations": [], "categories": [{"id": 1}]}
        (det / "one_gt_detection.json").write_text(json.dumps(document), encoding="utf-8")
        report = CleaningReport()
        _check_detection_jsons(det, tmp_path / "rgb", report, {"detection": {}})
        assert any(item.check == "detection/schema" for item in report.issues), file_name


def test_large_group_split_fallback_covers_four_splits_reproducibly(monkeypatch):
    from car_smart_assist.data import detection

    real_random = detection.random.Random

    class EmptyCandidateRandom:
        def __init__(self, seed):
            self.real = real_random(seed)

        def choices(self, population, weights=None, k=1):
            return [population[0]] * k

        def shuffle(self, values):
            self.real.shuffle(values)

    monkeypatch.setattr(detection.random, "Random", EmptyCandidateRandom)
    rows = [detection.DetectionImage(
        relative_path=f"fog/train/seq-{i}/frame.png", source=Path(f"unused-{i}"),
        width=10, height=10, condition="fog", sequence=f"seq-{i}", group=f"g{i}",
        labels=(), annotation_ids=(), sha256=f"{i:064x}") for i in range(12)]
    ratios = {"train": .997, "val": .001, "calibration": .001, "test": .001}
    first = detection.stratified_group_split(rows, ratios, 0, 1)
    second = detection.stratified_group_split(rows, ratios, 0, 1)
    assert all(first[name] for name in ratios)
    assert {r.group for split in first.values() for r in split} == {r.group for r in rows}
    assert [[r.group for r in first[name]] for name in ratios] == [[r.group for r in second[name]] for name in ratios]
    groups = [{r.group for r in split} for split in first.values()]
    assert all(not (groups[a] & groups[b]) for a in range(4) for b in range(a + 1, 4))
    normal = detection.stratified_group_split(rows[:4], ratios, 23, 1)
    assert all(normal[name] for name in ratios)


def _joint_helpers():
    return _helper("tests/unit/test_p2_pipeline_followup.py", "remaining_joint_fixtures")


def _write_five_laf_train_locations(root, helper):
    helper._write_laf_fixture(root)
    dataset = root / "data/raw/lost_and_found"
    for index in (3, 4):
        relative = Path("train") / f"location-{index}"
        stem = f"frame-{index}"
        image = dataset / "leftImg8bit" / relative / f"{stem}_leftImg8bit.png"
        mask = dataset / "gtCoarse" / relative / f"{stem}_gtCoarse_labelTrainIds.png"
        instances = mask.with_name(f"{stem}_gtCoarse_instanceIds.png")
        for path in (image, mask, instances):
            path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (4, 4), (20 + index, 30, 40)).save(image)
        Image.fromarray(np.ones((4, 4), dtype=np.uint8)).save(mask)
        Image.fromarray(np.zeros((4, 4), dtype=np.uint16)).save(instances)


@pytest.mark.parametrize("invalid_form", ["relative", "absolute"])
def test_joint_invalid_manifest_is_project_root_resolved_and_filtered(tmp_path, monkeypatch, invalid_form):
    from car_smart_assist.data.detection_joint import prepare_joint_detection

    helper = _joint_helpers()
    helper._write_base_export(tmp_path)
    _write_five_laf_train_locations(tmp_path, helper)
    dataset = tmp_path / "data/raw/lost_and_found/leftImg8bit/train/location-0/frame-0_leftImg8bit.png"
    invalid_path = str(dataset.resolve()) if invalid_form == "absolute" else str(dataset.relative_to(tmp_path))
    mask = tmp_path / "data/raw/lost_and_found/gtCoarse/train/location-0/frame-0_gtCoarse_labelTrainIds.png"
    invalid_mask = str(mask.resolve()) if invalid_form == "absolute" else str(mask.relative_to(tmp_path))
    (tmp_path / "data/cleaning.json").write_text(
        json.dumps({"invalid": [invalid_path, invalid_mask]}), encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path.parent)
    result = prepare_joint_detection(tmp_path, helper._joint_config())
    manifest_path = tmp_path / helper._joint_config()["output_root"] / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["signature"] == result["signature"]
    assert any("location-0" in row for row in manifest["deferred_corrupt"])
    assert all("location-0" not in row["source"] for row in manifest["images"] if row["dataset"] == "lost_and_found")


def test_joint_missing_cleaning_manifest_fails(tmp_path):
    from car_smart_assist.data.detection_joint import prepare_joint_detection

    helper = _joint_helpers(); helper._write_base_export(tmp_path); _write_five_laf_train_locations(tmp_path, helper)
    (tmp_path / "data/cleaning.json").unlink()
    with pytest.raises(FileNotFoundError, match="Cleaning manifest"):
        prepare_joint_detection(tmp_path, helper._joint_config())


@pytest.mark.parametrize("raw_action", ["delete", "replace"], ids=["raw-deleted", "raw-replaced"])
def test_joint_verified_base_survives_raw_source_change_and_preserves_provenance(tmp_path, raw_action):
    from car_smart_assist.data.detection_joint import prepare_joint_detection

    helper = _joint_helpers(); base = helper._write_base_export(tmp_path); _write_five_laf_train_locations(tmp_path, helper)
    base_manifest = json.loads((base / "manifest.json").read_text(encoding="utf-8"))
    base_image = base / base_manifest["images"][0]["image"]
    oracle_bytes = base_image.read_bytes()
    raw_source = Path(base_manifest["images"][0]["source"])
    if raw_action == "delete":
        raw_source.unlink()
    else:
        raw_source.write_bytes(b"raw source changed after verified base export")
    result = prepare_joint_detection(tmp_path, helper._joint_config())
    manifest_path = tmp_path / helper._joint_config()["output_root"] / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["signature"] == result["signature"]
    row = next(item for item in manifest["images"] if item["dataset"] == "acdc")
    assert row["source"] == str(raw_source)
    joint_bytes = (manifest_path.parent / row["image"]).read_bytes()
    assert joint_bytes == oracle_bytes
    assert hashlib.sha256(joint_bytes).hexdigest() == row["sha256"]


def test_joint_rejects_invalid_laf_contract_and_keeps_valid_zero_box_image(tmp_path):
    from car_smart_assist.data.detection_joint import prepare_joint_detection

    helper = _joint_helpers(); helper._write_base_export(tmp_path); _write_five_laf_train_locations(tmp_path, helper)
    base_cfg = helper._joint_config()
    invalid = [{**base_cfg, "obstacle_train_id": value} for value in (True, "2", 0, 3)]
    invalid += [{**base_cfg, "valid_roi_ids": value} for value in ([], [1, 1], ["1"], [True], [0], [255], [3])]
    for cfg in invalid:
        with pytest.raises(ValueError, match="Lost & Found"):
            prepare_joint_detection(tmp_path, cfg)
    result = prepare_joint_detection(tmp_path, base_cfg)
    manifest_path = tmp_path / base_cfg["output_root"] / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["signature"] == result["signature"]
    laf = [row for row in manifest["images"] if row["dataset"] == "lost_and_found"]
    assert laf and all(row["boxes"] == [] for row in laf)


@pytest.mark.parametrize("roi_ids", [[1], [2], [1, 2]], ids=["roi-1", "roi-2", "roi-1-2"])
def test_joint_accepts_each_supported_roi_contract(tmp_path, roi_ids):
    from car_smart_assist.data.detection_joint import prepare_joint_detection

    root = tmp_path / str(roi_ids).replace(" ", "")
    helper = _joint_helpers(); helper._write_base_export(root); _write_five_laf_train_locations(root, helper)
    cfg = helper._joint_config(); cfg["valid_roi_ids"] = roi_ids
    result = prepare_joint_detection(root, cfg)
    manifest_path = root / cfg["output_root"] / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["signature"] == result["signature"]
    assert manifest["valid_roi_ids"] == roi_ids


def _weather_samples(tmp_path):
    from car_smart_assist.perception.weather_training import WeatherSample

    samples = []
    for index, content in enumerate((b"same bytes", b"same bytes", b"different")):
        path = tmp_path / f"source-{index}.bin"
        path.write_bytes(content)
        samples.append(WeatherSample((1, 0, 0, 0), f"group-{index}", "fog", path=path))
    return samples


def test_duplicate_recording_cache_reuses_and_repairs_bad_identity(tmp_path, monkeypatch):
    from car_smart_assist.data import weather_pyramid

    samples = _weather_samples(tmp_path); cache = tmp_path / "cache"
    expected = [["group-0", "group-1"]]
    assert weather_pyramid._duplicate_recording_links(samples, cache) == expected
    target = next(cache.glob("file_hashes_*.json"))
    before = (target.read_bytes(), target.stat().st_mtime_ns)
    assert weather_pyramid._duplicate_recording_links(samples, cache) == expected
    assert (target.read_bytes(), target.stat().st_mtime_ns) == before
    for bad in (b"{", b"[]", json.dumps({"signature": "wrong", "cross_recording_links": []}).encode(),
                json.dumps({"signature": json.loads(before[0])["signature"],
                            "cross_recording_links": [["unknown", "group-0"]]}).encode()):
        target.write_bytes(bad)
        assert weather_pyramid._duplicate_recording_links(samples, cache) == expected
        assert json.loads(target.read_text(encoding="utf-8"))["cross_recording_links"] == expected


def test_duplicate_recording_cache_resource_and_replace_failures_propagate_and_cleanup(tmp_path, monkeypatch):
    from car_smart_assist.data import weather_pyramid

    samples = _weather_samples(tmp_path); cache = tmp_path / "cache"
    weather_pyramid._duplicate_recording_links(samples, cache)
    target = next(cache.glob("file_hashes_*.json")); original = target.read_bytes()
    target.write_text("{}", encoding="utf-8")
    damaged = target.read_bytes()
    real_open = Path.open

    def fail_source_read(path, *args, **kwargs):
        mode = args[0] if args else kwargs.get("mode", "r")
        if path.name == "source-0.bin" and "r" in mode:
            raise PermissionError("injected source read denial")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_source_read)
    with pytest.raises(PermissionError, match="injected source read denial"):
        weather_pyramid._duplicate_recording_links(samples, cache)
    assert target.read_bytes() == damaged
    monkeypatch.setattr(Path, "open", real_open)
    target.write_bytes(original)
    real_replace = Path.replace

    real_loads = weather_pyramid.json.loads
    cache_bytes = original

    def force_rebuild(payload, *args, **kwargs):
        parsed = real_loads(payload, *args, **kwargs)
        if payload.encode("utf-8") == cache_bytes:
            return {"signature": "forced-mismatch", "cross_recording_links": []}
        return parsed

    monkeypatch.setattr(weather_pyramid.json, "loads", force_rebuild)

    def fail_replace(path, destination):
        if path.parent == cache and path != target:
            raise PermissionError("injected replace denial")
        return real_replace(path, destination)

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(PermissionError, match="injected replace denial"):
        weather_pyramid._duplicate_recording_links(samples, cache)
    assert target.read_bytes() == cache_bytes
    assert not list(cache.glob(target.name + ".*.tmp"))


@pytest.mark.parametrize("samples,pixel_count", [([], 0), ("pixel", 2)], ids=["empty", "pixel-only"])
def test_pyramid_split_rejects_empty_or_pixel_only_with_counts(tmp_path, samples, pixel_count):
    from car_smart_assist.data.weather_pyramid import split_pyramid_samples
    from car_smart_assist.perception.weather_training import WeatherSample

    rows = [] if not samples else [WeatherSample((1, 0, 0, 0), f"p{i}", "pixel", archive_path=tmp_path / "p.zip", member=str(i)) for i in range(pixel_count)]
    with pytest.raises(ValueError) as exc_info:
        split_pyramid_samples(rows, {"train": .7, "validation": .1, "calibration": .1, "test": .1}, 1)
    assert f"ACDC=0，pixel_accurate={pixel_count}，total={pixel_count}" in str(exc_info.value)


def test_pyramid_split_keeps_mixed_source_groups_and_duplicate_links_disjoint(tmp_path):
    from car_smart_assist.data.weather_pyramid import split_pyramid_samples
    from car_smart_assist.perception.weather_training import WeatherSample

    samples = []
    for index in range(8):
        path = tmp_path / f"acdc-{index}.png"; path.write_bytes(bytes([index]))
        samples.append(WeatherSample((1, 0, 0, 0), f"a{index}", "fog", path=path))
    for index in range(4):
        samples.append(WeatherSample((0, 1, 0, 0), f"pixel{index}", "pixel", archive_path=tmp_path / "pixel.zip", member=str(index)))
    ratios = {"train": .7, "validation": .1, "calibration": .1, "test": .1}
    splits = split_pyramid_samples(samples, ratios, 2, duplicate_links=(('a0', 'a1'),))
    assert set(splits) == {"train", "validation", "calibration", "test"}
    assert all(splits[name] for name in ratios)
    assignment = {row.group: name for name, rows in splits.items() for row in rows}
    assert assignment["a0"] == assignment["a1"]
    group_sets = [{row.group for row in rows} for rows in splits.values()]
    assert all(not group_sets[i] & group_sets[j] for i in range(4) for j in range(i + 1, 4))
    assert sum(map(len, splits.values())) == len(samples)
