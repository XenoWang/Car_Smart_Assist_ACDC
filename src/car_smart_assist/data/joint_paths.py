"""Read-only current-root resolution for existing joint detection exports."""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path, PurePosixPath, PureWindowsPath

import yaml


def remove_original_empty_output(output, expected_identity):
    """Remove the recorded empty directory; Windows holds its identity until close."""
    import stat

    def require_original(current, *, check_empty=True):
        identity = current.st_dev, current.st_ino, getattr(current, "st_birthtime_ns", None)
        if (
            identity != expected_identity or not stat.S_ISDIR(current.st_mode)
            or getattr(current, "st_file_attributes", 0) & 0x400
            or (check_empty and (output.is_symlink() or any(output.iterdir())))
        ):
            raise ValueError("Original empty detection output changed; preserve it and choose a new output_root")

    if os.name != "nt":
        # Compatibility check only: POSIX path rmdir has no Windows sharing lock.
        require_original(output.lstat())
        output.rmdir()
        return

    import ctypes
    import msvcrt
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    create = kernel.CreateFileW
    create.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    create.restype = wintypes.HANDLE
    dispose = kernel.SetFileInformationByHandle
    dispose.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    dispose.restype = wintypes.BOOL
    close = kernel.CloseHandle
    close.argtypes = [wintypes.HANDLE]
    close.restype = wintypes.BOOL
    # DELETE | FILE_READ_ATTRIBUTES; share READ/WRITE, deliberately not DELETE.
    handle = create(str(output), 0x10000 | 0x80, 0x1 | 0x2, None, 3, 0x02000000 | 0x00200000, None)
    if handle == wintypes.HANDLE(-1).value:
        error = ctypes.WinError(ctypes.get_last_error())
        raise OSError(f"Cannot lock original empty output {output}: {error}; close other directory handles or choose a new output_root") from error
    fd = None
    try:
        fd = msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY)
        # open_osfhandle owns the native handle after successful transfer.
        handle = None
        # The held handle fixes identity; kernel disposition rejects nonempty
        # directories atomically without reopening the namespace for enumeration.
        require_original(os.fstat(fd), check_empty=False)

        class FileDispositionInfo(ctypes.Structure):
            _fields_ = [("DeleteFile", ctypes.c_ubyte)]  # Win32 BOOLEAN, one byte.

        info = FileDispositionInfo(1)
        if not dispose(msvcrt.get_osfhandle(fd), 4, ctypes.byref(info), ctypes.sizeof(info)):
            error = ctypes.WinError(ctypes.get_last_error())
            raise OSError(f"Cannot remove original empty output {output}: {error}; preserve it, close other handles or choose a new output_root") from error
    finally:
        if fd is not None:
            os.close(fd)
        elif handle is not None:
            close(handle)


def validate_joint_classes(classes):
    if not isinstance(classes, list) or not classes:
        raise ValueError("Joint classes must be a nonempty ordered list")
    ids, names = [], []
    for item in classes:
        if not isinstance(item, dict):
            raise ValueError("Joint classes must define source_id and name")
        source_id, name = item.get("source_id"), item.get("name")
        if (
            type(source_id) not in (int, str) or source_id == ""
            or not isinstance(name, str) or not name.strip()
        ):
            raise ValueError("Joint classes require nonempty source IDs and names")
        ids.append(source_id)
        names.append(name)
    if len(set(ids)) != len(ids) or len(set(names)) != len(names):
        raise ValueError("Joint classes must have unique source IDs and names")


def _lost_and_found_relative(row):
    value = row["source_relative"]
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError("Lost & Found source_relative must be a nonempty relative path")
    relative = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if (
        not isinstance(value, str) or "\\" in value or relative.is_absolute()
        or windows.drive or windows.root or ".." in relative.parts
        or len(relative.parts) < 4 or relative.parts[0] != "lost_and_found"
        or relative.parts[1] not in ("train", "test")
        or not relative.name.endswith("_leftImg8bit.png")
    ):
        raise ValueError("Invalid Lost & Found source_relative; explicit mapping cannot be resolved")
    return relative


def lost_and_found_paths(row, dataset):
    relative = _lost_and_found_relative(row)
    suffix = Path(*relative.parts[1:])
    prefix = relative.name.removesuffix("_leftImg8bit.png")
    source = dataset / "leftImg8bit" / suffix
    roi = dataset / "gtCoarse" / suffix.parent / (prefix + "_gtCoarse_labelTrainIds.png")
    instances = roi.with_name(prefix + "_gtCoarse_instanceIds.png")
    if any(not path.resolve().is_relative_to(dataset.resolve()) for path in (source, roi, instances)):
        raise ValueError("Lost & Found mapped path escapes the explicit dataset root")
    return source, roi, instances


def resolve_joint_manifest(manifest, lost_and_found_root=None):
    """Resolve only an explicit root, retaining the original identity and metadata."""
    if not isinstance(manifest, dict):
        raise ValueError("Detection manifest must be a JSON object")
    if not manifest.get("joint_supervision"):
        return manifest
    validate_joint_classes(manifest["classes"])
    known = manifest["known_class_count"]
    if type(known) is not int or known < 1 or len(manifest["classes"]) != known + 1:
        raise ValueError("Joint known_class_count does not match ordered classes")
    roi_ids = manifest["valid_roi_ids"]
    if (
        not isinstance(roi_ids, list) or not roi_ids
        or any(type(value) is not int or value not in (1, 2) for value in roi_ids)
        or len(set(roi_ids)) != len(roi_ids)
    ):
        raise ValueError("Joint valid_roi_ids must be unique integers drawn from 1/2")
    for key in ("source_weights", "classification_loss_weights"):
        values = manifest[key]
        if (
            not isinstance(values, dict) or set(values) != {"acdc", "lost_and_found"}
            or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0 for v in values.values())
        ):
            raise ValueError(f"Invalid joint {key}")
    if not math.isclose(sum(manifest["source_weights"].values()), 1.0, abs_tol=1e-9):
        raise ValueError("Joint source weights must sum to one")
    if (not isinstance(manifest.get("images"), list)
            or any(not isinstance(row, dict) for row in manifest["images"])):
        raise ValueError("Joint manifest images must be a list of objects")
    result = deepcopy(manifest)
    for row in result["images"]:
        source = row["dataset"]
        if source not in ("acdc", "lost_and_found"):
            raise ValueError("Unknown joint annotation source")
        expected = list(range(known)) if source == "acdc" else [known]
        if row["supervised_classes"] != expected or any(int(box[0]) not in expected for box in row["boxes"]):
            raise ValueError("Joint annotation coverage changed")
        if source == "acdc":
            if not isinstance(row.get("source"), str) or not row["source"]:
                raise ValueError("ACDC joint rows require their source location")
            if row.get("roi_mask") is not None:
                raise ValueError("ACDC joint rows must retain full-image supervision")
            continue
        for key in ("sha256", "roi_sha256", "instance_sha256"):
            if not isinstance(row.get(key), str) or re.fullmatch(r"[0-9a-f]{64}", row[key]) is None:
                raise ValueError(f"Lost & Found row lacks a valid {key}")
        if any(not isinstance(row.get(key), str) or not row[key] for key in ("source", "roi_mask")):
            raise ValueError("Lost & Found row requires source and roi_mask; full-image fallback is refused")
        # Check the fixed naming contract even when no relocation was requested.
        _lost_and_found_relative(row)
        if lost_and_found_root is not None:
            current_source, roi, _ = lost_and_found_paths(row, Path(lost_and_found_root).resolve())
            row.update(source=str(current_source), roi_mask=str(roi))
    return result


def load_joint_data_config(path):
    try:
        document = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        cfg = document["joint_detection_data"]
        if not isinstance(cfg, dict):
            raise TypeError("joint_detection_data must be a mapping")
        if not isinstance(cfg.get("lost_and_found_root"), str) or not cfg["lost_and_found_root"]:
            raise ValueError("An explicit current lost_and_found_root is required")
        return cfg
    except (KeyError, TypeError, yaml.YAMLError) as exc:
        raise ValueError(f"Invalid joint detection configuration {path}: {exc}") from exc


def read_joint_yaml(path):
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            raise ValueError("Joint data YAML must be a mapping")
        return document
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid joint data YAML {path}: {exc}") from exc


def read_detection_manifest(root, data, joint_data_config=None, *, verify_hashes=False):
    """Check a current export without preparing, publishing or writing reports."""
    from car_smart_assist.data.detection import validate_detection_export

    data = (root / data).resolve()
    manifest = json.loads((data.parent / "manifest.json").read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("Detection manifest must be a JSON object")
    mapping = None
    if joint_data_config is not None:
        if not manifest.get("joint_supervision"):
            raise ValueError("--joint-data-config requires a joint detection export")
        cfg = load_joint_data_config(root / joint_data_config)
        from car_smart_assist.data.detection_joint import validate_existing_joint_export

        manifest, counts = validate_existing_joint_export(root, cfg, data.parent)
        mapping = str((root / cfg["lost_and_found_root"]).resolve())
    else:
        manifest = resolve_joint_manifest(manifest)
        counts = validate_detection_export(data.parent, verify_hashes) if verify_hashes else None
    if manifest.get("joint_supervision"):
        document = read_joint_yaml(data)
        expected = {
            "train": "train.txt", "val": "calibration.txt" if data.name == "calibration.yaml" else "val.txt",
            "test": "test.txt", "names": {i: item["name"] for i, item in enumerate(manifest["classes"])},
        }
        semantic = {key: value for key, value in document.items() if key not in ("path", "joint_manifest")}
        if semantic != expected:
            raise ValueError("Joint data YAML differs from the validated manifest")
        if mapping is None and (
            Path(document["path"]).resolve() != data.parent
            or Path(document["joint_manifest"]).resolve() != data.parent / "manifest.json"
        ):
            raise ValueError("Relocated joint YAML requires explicit --joint-data-config with current roots")
    return manifest, counts, mapping


@contextmanager
def detection_data_scope(root, data, *, joint_data_config=None, directory=None):
    """Keep vendor YAML adaptations local to this one running operation."""
    data = (root / data).resolve()
    manifest, counts, mapping = read_detection_manifest(root, data, joint_data_config, verify_hashes=True)
    if mapping is None:
        yield data, manifest, counts
        return
    document = read_joint_yaml(data)
    document.update(path=str(data.parent), joint_manifest=str(data.parent / "manifest.json"), joint_lost_and_found_root=mapping)
    with tempfile.TemporaryDirectory(prefix=".joint-data-", dir=directory) as temporary:
        sidecar = Path(temporary) / data.name
        content = yaml.safe_dump(document).encode("utf-8")
        with sidecar.open("xb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        if sidecar.read_bytes() != content:
            raise OSError("Joint data sidecar readback mismatch")
        yield sidecar, manifest, counts
