"""Bounded Windows publication recovery and safe rejection."""

from __future__ import annotations

import errno
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from car_smart_assist.data import detection


ROOT = Path(__file__).resolve().parents[2]


def _windows_error(code):
    error = PermissionError(errno.EACCES, "injected Windows publication denial")
    error.winerror = code
    return error


def _snapshot(root):
    return {path.relative_to(root): (path.read_bytes(), path.stat().st_mtime_ns)
            for path in root.rglob("*") if path.is_file()}


def _staging(tmp_path):
    source = tmp_path / ".export.staging-original"
    source.mkdir()
    (source / "payload.bin").write_bytes(b"original validated export")
    return source, tmp_path / "export"


@pytest.mark.parametrize("code", [5, 32, 33])
def test_temporary_windows_denial_publishes_original_bytes(tmp_path, monkeypatch, code):
    source, output = _staging(tmp_path)
    before = _snapshot(source)
    rename = Path.rename
    attempts, waits = [], []

    def deny_twice(path, target):
        if path == source:
            attempts.append(target)
            if len(attempts) <= 2:
                raise _windows_error(code)
        return rename(path, target)

    monkeypatch.setattr(Path, "rename", deny_twice)
    monkeypatch.setattr(detection, "time", SimpleNamespace(sleep=waits.append))
    detection._publish_detection_staging(source, output)
    assert len(attempts) == 3
    assert waits == [0.05, 0.1]
    assert not source.exists()
    assert _snapshot(output) == before


@pytest.mark.parametrize("code", [5, 32, 33])
def test_permanent_windows_denial_stops_after_bounded_attempts(tmp_path, monkeypatch, code):
    source, output = _staging(tmp_path)
    before = _snapshot(source)
    error = _windows_error(code)
    attempts, waits = [], []

    def denied(path, target):
        attempts.append((path, target))
        raise error

    monkeypatch.setattr(Path, "rename", denied)
    monkeypatch.setattr(detection, "time", SimpleNamespace(sleep=waits.append))
    with pytest.raises(PermissionError) as raised:
        detection._publish_detection_staging(source, output)
    assert raised.value is error
    assert len(attempts) == 6
    assert len(waits) == 5 and sum(waits) == pytest.approx(1.55)
    assert not output.exists()
    assert _snapshot(source) == before


@pytest.mark.parametrize("code", [None, 2, 3, 17, 80, 183])
def test_other_errors_are_not_retried_or_replaced(tmp_path, monkeypatch, code):
    source, output = _staging(tmp_path)
    before = _snapshot(source)
    error = _windows_error(code)
    attempts, waits = [], []

    def denied(path, target):
        attempts.append((path, target))
        raise error

    monkeypatch.setattr(Path, "rename", denied)
    monkeypatch.setattr(detection, "time", SimpleNamespace(sleep=waits.append))
    with pytest.raises(PermissionError) as raised:
        detection._publish_detection_staging(source, output)
    assert raised.value is error
    assert len(attempts) == 1 and waits == []
    assert not output.exists()
    assert _snapshot(source) == before


@pytest.mark.parametrize("nonempty", [False, True])
def test_output_appearing_between_attempts_is_preserved(tmp_path, monkeypatch, nonempty):
    source, output = _staging(tmp_path)
    before = _snapshot(source)
    attempts = []

    def competing_output(path, target):
        attempts.append((path, target))
        output.mkdir()
        if nonempty:
            (output / "competitor.bin").write_bytes(b"another producer")
        raise _windows_error(5)

    monkeypatch.setattr(Path, "rename", competing_output)
    monkeypatch.setattr(detection, "time", SimpleNamespace(sleep=lambda _delay: None))
    with pytest.raises(ValueError, match="output appeared"):
        detection._publish_detection_staging(source, output)
    assert len(attempts) == 1
    assert output.is_dir()
    assert _snapshot(source) == before
    assert sorted(path.name for path in output.iterdir()) == (["competitor.bin"] if nonempty else [])
    if nonempty:
        assert (output / "competitor.bin").read_bytes() == b"another producer"


def test_staging_replacement_between_attempts_is_rejected(tmp_path, monkeypatch):
    source, output = _staging(tmp_path)
    preserved = tmp_path / "original-preserved"
    before = _snapshot(source)
    rename = Path.rename
    attempts = []

    def replaced(path, target):
        attempts.append((path, target))
        rename(source, preserved)
        source.mkdir()
        (source / "unexpected.bin").write_bytes(b"replacement")
        raise _windows_error(5)

    monkeypatch.setattr(Path, "rename", replaced)
    monkeypatch.setattr(detection, "time", SimpleNamespace(sleep=lambda _delay: None))
    with pytest.raises(ValueError, match="staging identity changed"):
        detection._publish_detection_staging(source, output)
    assert len(attempts) == 1 and not output.exists()
    assert _snapshot(preserved) == before
    assert (source / "unexpected.bin").read_bytes() == b"replacement"


@pytest.mark.skipif(os.name != "nt", reason="requires a real Windows export-file handle")
def test_windows_open_export_file_is_released_then_publication_succeeds(tmp_path, monkeypatch):
    import ctypes
    from ctypes import wintypes

    source, output = _staging(tmp_path)
    before = _snapshot(source)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateFileW.argtypes = (
        wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p,
        wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE,
    )
    kernel.CreateFileW.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel.CloseHandle.restype = wintypes.BOOL
    # Hold the exported payload for real reading, without shared delete access.
    # An attributes-only directory handle does not reliably block parent rename.
    handle = kernel.CreateFileW(str(source / "payload.bin"), 0x80000000, 0x1, None, 3, 0x80, None)
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    released = False
    errors, waits = [], []
    rename = Path.rename

    def observe_rename(path, target):
        try:
            return rename(path, target)
        except OSError as exc:
            errors.append(exc.winerror)
            raise

    def release_handle(delay):
        nonlocal released
        waits.append(delay)
        if not released:
            if not kernel.CloseHandle(handle):
                raise ctypes.WinError(ctypes.get_last_error())
            released = True

    monkeypatch.setattr(Path, "rename", observe_rename)
    monkeypatch.setattr(detection, "time", SimpleNamespace(sleep=release_handle))
    try:
        detection._publish_detection_staging(source, output)
    finally:
        if not released and not kernel.CloseHandle(handle):
            raise ctypes.WinError(ctypes.get_last_error())
    assert errors and all(code in (5, 32, 33) for code in errors)
    assert released and 1 <= len(waits) <= 5 and len(waits) == len(errors)
    assert not source.exists() and _snapshot(output) == before
    print(f"WINDOWS_RENAME_BLOCK_CODES={errors}")


def _load_helper(relative, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("kind", ["acdc", "joint"])
@pytest.mark.parametrize("permanent", [False, True])
def test_export_entry_points_retry_or_clean_up_without_touching_sources(
    tmp_path, monkeypatch, kind, permanent,
):
    if kind == "acdc":
        fixtures = _load_helper("tests/unit/test_dataset_export_and_split_contracts.py", "publish_acdc_fixture")
        cfg = fixtures._full_detection_fixture(tmp_path)
        prepare = detection.prepare_detection_data
    else:
        fixtures = _load_helper("tests/unit/test_data_preparation_and_policy_guards.py", "publish_joint_fixture")
        _, cfg = fixtures._joint_fixture(tmp_path)
        from car_smart_assist.data.detection_joint import prepare_joint_detection
        prepare = prepare_joint_detection
    source_before = _snapshot(tmp_path / "data")
    output = tmp_path / cfg["output_root"]
    rename = Path.rename
    attempts, waits = [], []
    error = _windows_error(5)

    def denial(path, target):
        if target == output:
            attempts.append(path)
            if permanent or len(attempts) == 1:
                raise error
        return rename(path, target)

    monkeypatch.setattr(Path, "rename", denial)
    monkeypatch.setattr(detection, "time", SimpleNamespace(sleep=waits.append))
    if permanent:
        with pytest.raises(PermissionError) as raised:
            prepare(tmp_path, cfg)
        assert raised.value is error
        assert len(attempts) == 6 and sum(waits) == pytest.approx(1.55)
        assert not output.exists()
        assert _snapshot(tmp_path / "data") == source_before
    else:
        prepare(tmp_path, cfg)
        assert len(attempts) == 2 and waits == [0.05]
        assert sum(detection.validate_detection_export(output, verify_hashes=True).values()) > 0
        published = _snapshot(output)
        prepare(tmp_path, cfg)
        assert len(attempts) == 2
        assert _snapshot(output) == published
        after = _snapshot(tmp_path / "data")
        assert all(after[path] == value for path, value in source_before.items())
    assert not list(output.parent.glob(f".{output.name}.staging-*"))
