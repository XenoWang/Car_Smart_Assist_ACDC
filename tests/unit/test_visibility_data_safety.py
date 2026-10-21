from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, UnidentifiedImageError

from car_smart_assist.perception.visibility.dataset import (
    VisibilityImageDataset,
    _cache_identity,
    ensure_cache,
    list_ref_images,
)


def _write_rgb(path: Path, color: tuple[int, int, int]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color).save(path)
    return path


def _ref(root: Path, condition: str, sequence: str, frame: int) -> Path:
    return _write_rgb(
        root
        / "data/raw/acdc/rgb_anon"
        / condition
        / "train_ref"
        / sequence
        / f"{sequence}_frame_{frame:06d}_rgb_ref_anon.png",
        (frame % 255, 40, 90),
    )


def test_invalid_manifest_filters_only_invalid_and_keeps_suspect(tmp_path):
    acdc_root = tmp_path / "data/raw/acdc"
    valid = _ref(tmp_path, "rain", "GOPR0001", 1)
    invalid = _ref(tmp_path, "rain", "GOPR0001", 2)
    suspect = _ref(tmp_path, "night", "GOPR0002", 3)
    manifest = tmp_path / "data/processed/manifests/acdc.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "invalid": [invalid.relative_to(tmp_path).as_posix()],
                "suspect": [suspect.relative_to(tmp_path).as_posix()],
            }
        ),
        encoding="utf-8",
    )

    paths = list_ref_images(
        acdc_root,
        cleaning_manifest="data/processed/manifests/acdc.json",
        project_root=tmp_path,
    )
    assert set(paths) == {valid, suspect}
    assert suspect in paths  # Suspect/night samples are retained.


def test_invalid_only_manifest_excludes_invalid_but_keeps_unlisted_suspect_and_raw(tmp_path):
    acdc_root = tmp_path / "data/raw/acdc"
    invalid = _ref(tmp_path, "rain", "GOPR0001", 1)
    suspect = _ref(tmp_path, "night", "GOPR0002", 2)
    manifest = tmp_path / "data/processed/manifests/acdc.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps({"invalid": [invalid.relative_to(tmp_path).as_posix()]}),
        encoding="utf-8",
    )

    paths = list_ref_images(
        acdc_root,
        cleaning_manifest="data/processed/manifests/acdc.json",
        project_root=tmp_path,
    )

    assert invalid.is_file()  # Filtering never deletes a raw input.
    assert paths == [suspect]


def test_missing_manifest_errors_even_when_sample_collection_is_empty(tmp_path):
    missing = "data/processed/manifests/not-created.json"
    acdc_root = tmp_path / "data/raw/acdc"
    with pytest.raises(FileNotFoundError) as exc_info:
        list_ref_images(
            acdc_root,
            cleaning_manifest=missing,
            project_root=tmp_path,
        )

    message = str(exc_info.value)
    assert str(tmp_path / missing) in message
    assert str(tmp_path) in message
    assert "先进行数据清洗" in message
    assert r".\.venv\Scripts\python.exe -X utf8 scripts\clean_data.py --only acdc --corrupt-images-only" in message


def test_none_manifest_preserves_legacy_sample_enumeration(tmp_path):
    acdc_root = tmp_path / "data/raw/acdc"
    paths = {_ref(tmp_path, "fog", "GOPR0001", 1), _ref(tmp_path, "snow", "GOPR0002", 2)}
    found = list_ref_images(acdc_root, cleaning_manifest=None, project_root=tmp_path)
    assert set(found) == paths


def test_invalid_manifest_json_fails_clearly_instead_of_ignoring_filter(tmp_path):
    acdc_root = tmp_path / "data/raw/acdc"
    _ref(tmp_path, "fog", "GOPR0001", 1)
    manifest = tmp_path / "broken.json"
    manifest.write_text("{invalid", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        list_ref_images(acdc_root, cleaning_manifest=manifest, project_root=tmp_path)


def test_corrupt_sample_is_not_replaced_with_a_zero_image(tmp_path):
    bad = tmp_path / "broken_rgb_ref_anon.png"
    bad.write_bytes(b"not a png")
    dataset = VisibilityImageDataset([bad], input_size=(4, 4), augment=False)
    with pytest.raises(UnidentifiedImageError):
        dataset[0]


def test_cache_reuses_matching_identity_and_preserves_rgb_uint8_shape(tmp_path, monkeypatch):
    import car_smart_assist.perception.visibility.dataset as module

    paths = [_write_rgb(tmp_path / f"{i}.png", (i * 50, 10, 20)) for i in range(2)]
    cache = tmp_path / "cache/images.npy"
    original_builder = module._build_cache
    builds = 0

    def counted_builder(source_paths, size, target):
        nonlocal builds
        builds += 1
        original_builder(source_paths, size, target)

    monkeypatch.setattr(module, "_build_cache", counted_builder)
    assert ensure_cache(paths, (4, 4), cache) == cache
    assert ensure_cache(paths, (4, 4), cache) == cache
    values = np.load(cache, allow_pickle=False)
    assert values.shape == (2, 4, 4, 3)
    assert values.dtype == np.uint8
    assert builds == 1


def test_cache_identity_binds_order_size_algorithm_and_file_content(tmp_path):
    paths = [_write_rgb(tmp_path / f"{i}.png", (i * 70, 30, 40)) for i in range(2)]
    identity = _cache_identity(paths, (4, 4))
    assert identity["files"][0][0] == str(paths[0].resolve())
    assert identity["input_size"] == [4, 4]
    assert identity["algorithm"]
    assert _cache_identity(list(reversed(paths)), (4, 4)) != identity
    assert _cache_identity(paths, (8, 8)) != identity

    before = paths[0].stat()
    _write_rgb(paths[0], (1, 250, 2))
    os.utime(paths[0], ns=(before.st_atime_ns, before.st_mtime_ns))
    assert _cache_identity(paths, (4, 4)) != identity


def test_cache_rebuilds_for_reordered_paths_and_new_dimensions(tmp_path):
    paths = [_write_rgb(tmp_path / f"{i}.png", (i * 90, 15, 25)) for i in range(2)]
    cache = tmp_path / "cache/images.npy"
    ensure_cache(paths, (4, 4), cache)
    ensure_cache(list(reversed(paths)), (4, 4), cache)
    reordered = np.load(cache, allow_pickle=False)
    assert tuple(reordered[0, 0, 0]) == (90, 15, 25)

    ensure_cache(paths, (6, 5), cache)
    resized = np.load(cache, allow_pickle=False)
    assert resized.shape == (2, 6, 5, 3)
    assert resized.dtype == np.uint8


def test_cache_rejects_wrong_shape_or_dtype_even_when_manifest_hash_matches(tmp_path):
    import car_smart_assist.perception.visibility.dataset as module

    paths = [_write_rgb(tmp_path / "one.png", (5, 6, 7))]
    cache = tmp_path / "cache/images.npy"
    ensure_cache(paths, (4, 4), cache)
    manifest_path = cache.with_suffix(cache.suffix + ".json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    with cache.open("wb") as stream:
        np.save(stream, np.zeros((1, 3, 4, 3), dtype=np.float32), allow_pickle=False)
    manifest["cache_sha256"] = module._file_sha256(cache)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    ensure_cache(paths, (4, 4), cache)
    rebuilt = np.load(cache, allow_pickle=False)
    assert rebuilt.shape == (1, 4, 4, 3)
    assert rebuilt.dtype == np.uint8


def test_failed_cache_build_keeps_previous_cache_and_manifest(tmp_path, monkeypatch):
    import car_smart_assist.perception.visibility.dataset as module

    paths = [_write_rgb(tmp_path / "one.png", (10, 20, 30))]
    cache = tmp_path / "cache/images.npy"
    ensure_cache(paths, (4, 4), cache)
    manifest_path = cache.with_suffix(cache.suffix + ".json")
    previous_cache = cache.read_bytes()
    previous_manifest = manifest_path.read_bytes()

    monkeypatch.setattr(module, "_build_cache", lambda *_: (_ for _ in ()).throw(OSError("decode failed")))
    with pytest.raises(OSError, match="decode failed"):
        ensure_cache(paths, (6, 6), cache)

    assert cache.read_bytes() == previous_cache
    assert manifest_path.read_bytes() == previous_manifest
    assert not list(cache.parent.glob(".*.tmp"))


def test_cache_build_failure_does_not_publish_partial_files(tmp_path):
    paths = [tmp_path / "corrupt.png"]
    paths[0].write_bytes(b"not an image")
    cache = tmp_path / "new/cache.npy"

    with pytest.raises(UnidentifiedImageError):
        ensure_cache(paths, (4, 4), cache)

    assert not cache.exists()
    assert not cache.with_suffix(cache.suffix + ".json").exists()
    assert not list(cache.parent.glob(".*.tmp"))


def test_source_change_during_cache_build_is_not_published(tmp_path, monkeypatch):
    import car_smart_assist.perception.visibility.dataset as module

    source = _write_rgb(tmp_path / "one.png", (10, 20, 30))
    cache = tmp_path / "cache/images.npy"
    original_builder = module._build_cache

    def change_source_during_build(paths, size, target):
        original_builder(paths, size, target)
        _write_rgb(source, (200, 30, 40))

    monkeypatch.setattr(module, "_build_cache", change_source_during_build)
    with pytest.raises(ValueError, match="源图片发生变化"):
        ensure_cache([source], (4, 4), cache)
    assert not cache.exists()
    assert not cache.with_suffix(cache.suffix + ".json").exists()
    assert not list(cache.parent.glob(".*.tmp"))
