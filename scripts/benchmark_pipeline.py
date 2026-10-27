"""Explicit CLI benchmark of the existing pipeline; no models run on import."""

from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import sys
import time
from collections import Counter
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CSV_FIELDS = [
    "scope", "input_mode", "phase", "iteration", "batch_index", "input_start",
    "requested_batch_size", "actual_batch_size", "call_count", "attempted_frame_count",
    "route_success_frame_count", "full_perception_frame_count", "blind_frame_count",
    "unavailable_frame_count", "wall_ms", "attempted_throughput_fps",
    "complete_route_throughput_fps", "full_perception_throughput_fps", "error",
]
LIMITATIONS = [
    "Desktop hardware measurements do not establish a vehicle hardware FPS target.",
    "Python startup, imports and OS cache clearing are outside the cold-start definition.",
    "The same pipeline runs paths first, then arrays; array first calls are already model-warm.",
    "File/OS caches are uncontrolled. No CUDA allocator cache is cleared.",
    "Decode timing includes RGB conversion, validation and array materialization via _to_array.",
    "Path-minus-array latency is not a measurement of pure decoding or any network stage.",
    "Array inputs still undergo run_batch input validation and copying.",
    "timings_ms is internal diagnostic data: gate/advisory times are batch amortizations; "
    "CUDA stages are not independently synchronized. Missing stages are not filled with zero.",
    "Resize, CPU features, host/device transfers, individual networks and NMS are not measured "
    "independently through the current interface.",
    "Batch wall time is batch response latency; dividing by batch size is not frame response latency.",
    "RAM peaks are process-lifetime peaks, including imports and retained arrays/results, "
    "not peaks reset for a benchmark round.",
    "CUDA allocated/reserved and their reset peaks concern this process's PyTorch allocator, "
    "not whole-device memory consumption.",
    "A BLIND route intentionally skips perception. Its throughput cannot describe full perception.",
]


def _count(text: str, *, allow_zero: bool = False) -> int:
    try:
        value = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if value < (0 if allow_zero else 1):
        raise argparse.ArgumentTypeError("must be non-negative" if allow_zero else "must be positive")
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", action="append", required=True, help="Explicit image path; repeat in order")
    parser.add_argument("--image-range", nargs=2, type=float, metavar=("MIN", "MAX"),
                        help="Fixed input code range, e.g. 0 4095 for processed 12-bit images")
    parser.add_argument("--config", default="configs/model/visibility.yaml")
    parser.add_argument("--checkpoint", help="Visibility checkpoint; otherwise use the pipeline default")
    parser.add_argument("--weather-config")
    parser.add_argument("--weather-checkpoint")
    parser.add_argument("--weather-loss", choices=("focal", "asl"), default="focal")
    parser.add_argument("--device", default="auto", help="auto, cpu or cuda[:index]; applies to all models")
    parser.add_argument("--batch-size", type=_count, default=1)
    parser.add_argument("--warmup", type=lambda x: _count(x, allow_zero=True), default=1,
                        help="Additional full sample-set warmup passes per input mode")
    parser.add_argument("--repeats", type=_count, default=5,
                        help="Stable full sample-set passes per input mode")
    parser.add_argument("--output-dir", "--output", dest="output_dir", required=True,
                        help="A new directory; existing directories are refused")
    args = parser.parse_args(argv)
    if args.image_range is not None:
        low, high = args.image_range
        if not all(math.isfinite(value) for value in (low, high, high - low)) or high <= low:
            parser.error("--image-range must be a finite increasing code range")
    return args


def _path(text: str | Path) -> Path:
    path = Path(text)
    return (ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def _stat(path: Path) -> dict[str, int] | None:
    if not path.is_file():
        return None
    stat = path.stat()
    return {"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _fingerprint(path: Path, initial: dict[str, int] | None) -> dict[str, Any]:
    record: dict[str, Any] = {"path": str(path), "before": initial}
    try:
        before_hash = _stat(path)
        digest = hashlib.sha256()
        if before_hash is not None:
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
        after_hash = _stat(path)
        record.update(
            exists=after_hash is not None,
            sha256=digest.hexdigest() if before_hash is not None else None,
            after=after_hash,
            changed_during_run=initial != before_hash or before_hash != after_hash,
        )
    except OSError as exc:
        record.update(sha256=None, error=f"{type(exc).__name__}: {exc}",
                      changed_during_run=True)
    return record


def _artifact_paths(args: argparse.Namespace, visibility: dict, yaml: Any) -> dict[str, Path]:
    """Record the existing pipeline routes without substituting a different route."""
    from car_smart_assist.perception.yolo_postprocess import calibration_path

    paths = {
        "benchmark_script": Path(__file__).resolve(),
        "visibility_config": _path(args.config),
        "visibility_checkpoint": _path(args.checkpoint) if args.checkpoint else
        ROOT / visibility.get("train", {}).get("checkpoint_dir", "artifacts/checkpoints/visibility")
        / "best.pt",
    }
    weather_config = args.weather_config
    weather_checkpoint = args.weather_checkpoint
    route_path = ROOT / "configs/model/weather_default.yaml"
    if weather_config is None and route_path.is_file():
        paths["weather_default_route"] = route_path
        route = yaml.safe_load(route_path.read_text(encoding="utf-8"))["weather_default"]
        weather_config = route["config"]
        if weather_checkpoint is None:
            weather_checkpoint = route["checkpoint"]
    if weather_config is not None:
        weather_path = _path(weather_config)
        paths["weather_config"] = weather_path
        raw = yaml.safe_load(weather_path.read_text(encoding="utf-8"))
        key = "weather_attention" if "weather_attention" in raw else "weather_pyramid"
        suffix = Path("best.pt") if key == "weather_attention" else Path(args.weather_loss) / "best.pt"
        paths["weather_checkpoint"] = _path(weather_checkpoint) if weather_checkpoint else (
            _path(raw[key]["checkpoint_root"]) / suffix
        )
    else:
        legacy_path = ROOT / "configs/model/weather_classifier.yaml"
        paths["weather_config"] = legacy_path
        if legacy_path.is_file():
            raw = yaml.safe_load(legacy_path.read_text(encoding="utf-8"))["weather_classifier"]
            paths["weather_checkpoint"] = _path(weather_checkpoint or raw["checkpoint"])
    for name in ("yolo_detection", "yolo_acdc_baseline"):
        path = ROOT / f"configs/model/{name}.yaml"
        paths[f"{name}_config_candidate"] = path
        if path.is_file():
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))["yolo_detection"]
            for key in ("checkpoint", "calibration_file"):
                if raw.get(key):
                    paths[f"{name}_{key}_candidate"] = _path(raw[key])
            paths[f"{name}_calibration"] = calibration_path(ROOT, raw)
    return {key: path.resolve() for key, path in paths.items()}


def _ram() -> dict[str, Any]:
    """Use available OS process readings, without installing a memory sampler."""
    result: dict[str, Any] = {
        "working_set_bytes": None, "process_lifetime_peak_working_set_bytes": None,
        "peak_scope": "Since process creation through this sample, including work before benchmark timing",
    }
    try:
        if sys.platform == "win32":
            from ctypes import wintypes

            class Counters(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = Counters()
            counters.cb = ctypes.sizeof(counters)
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            psapi = ctypes.WinDLL("psapi", use_last_error=True)
            kernel.GetCurrentProcess.restype = wintypes.HANDLE
            psapi.GetProcessMemoryInfo.argtypes = [
                wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD,
            ]
            psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
            if not psapi.GetProcessMemoryInfo(
                kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb
            ):
                raise ctypes.WinError(ctypes.get_last_error())
            result.update(working_set_bytes=counters.WorkingSetSize,
                          process_lifetime_peak_working_set_bytes=counters.PeakWorkingSetSize,
                          source="Windows GetProcessMemoryInfo")
        elif sys.platform.startswith("linux"):
            values = {}
            for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
                key, _, text = line.partition(":")
                if key in {"VmRSS", "VmHWM"}:
                    values[key] = int(text.split()[0]) * 1024
            result.update(working_set_bytes=values.get("VmRSS"),
                          process_lifetime_peak_working_set_bytes=values.get("VmHWM"),
                          source="/proc/self/status VmRSS/VmHWM")
            if not values:
                result["unavailable_reason"] = "VmRSS and VmHWM are unavailable"
        else:
            result["unavailable_reason"] = f"No process working-set reader for {sys.platform}"
    except (OSError, ValueError, AttributeError) as exc:
        result["unavailable_reason"] = f"{type(exc).__name__}: {exc}"
    return result


def _components(pipe: Any) -> dict[str, Any]:
    records = {}
    for name, component in (
        ("visibility", pipe.scorer), ("weather", pipe.weather_predictor),
        ("detection", pipe.predictor),
    ):
        if component is None:
            records[name] = {"available": False, "reason": (
                pipe.predictor_unavailable_reason if name == "detection" else
                "No component constructed by from_config"
            )}
            continue
        model = component.model
        models = [("model", model)]
        if getattr(model, "model", None) is not None:
            models.append(("model.model", model.model))
        predictor = getattr(model, "predictor", None)
        if getattr(predictor, "model", None) is not None:
            models.append(("model.predictor.model", predictor.model))
        observed = {}
        for label, candidate in models:
            if callable(getattr(candidate, "parameters", None)):
                observed[label] = sorted({
                    str(tensor.device)
                    for tensor in list(candidate.parameters()) + list(candidate.buffers())
                })
        cfg = getattr(component, "cfg", None)
        if is_dataclass(cfg):
            cfg = asdict(cfg)
        records[name] = {
            "available": True,
            "class": f"{type(component).__module__}.{type(component).__name__}",
            "declared_device": str(component.device),
            "observed_tensor_devices": observed,
            "configuration": cfg,
        }
    return records


def _cuda_devices(torch: Any, selected: Any, components: dict) -> list[int]:
    devices = set()
    candidates = [str(selected)]
    for record in components.values():
        if record["available"]:
            declared = record["declared_device"]
            candidates.append("cuda:" + declared if declared.isdigit() else declared)
            for observed in record["observed_tensor_devices"].values():
                candidates.extend(observed)
    for text in candidates:
        device = torch.device(text)
        if device.type == "cuda":
            devices.add(torch.cuda.current_device() if device.index is None else device.index)
    return sorted(devices)


def _sync(torch: Any, devices: list[int]) -> None:
    for index in devices:
        torch.cuda.synchronize(index)


def _gpu_memory(torch: Any, devices: list[int], *, round_peak: bool = True) -> dict[str, Any]:
    return {
        str(index): {
            "device_name": torch.cuda.get_device_name(index),
            "allocated_bytes": torch.cuda.memory_allocated(index),
            "reserved_bytes": torch.cuda.memory_reserved(index),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(index),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(index),
            "peak_scope": "Round since reset before construction" if round_peak else
            "Process allocator peak before this round reset",
        }
        for index in devices
    }


def _safe_result(value: Any, nonfinite: list[str], path: str = "$") -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        nonfinite.append(path)
        return None
    if isinstance(value, dict):
        return {key: _safe_result(item, nonfinite, f"{path}.{key}")
                for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_result(item, nonfinite, f"{path}[{index}]")
                for index, item in enumerate(value)]
    return value


def _branch_state(result: dict | None, result_error: str | None = None) -> dict:
    result = result or {}
    skipped = result.get("skipped", {})
    visibility = result.get("visibility") or {}
    perception = result.get("perception") or {}
    level = visibility.get("level", "unknown")
    blind = level == "blind" and "input" not in skipped and "gate" not in skipped
    advisory = result.get("advisory")
    branches = {
        "input": "available" if result and "input" not in skipped else "unavailable",
        "gate": "available" if level in {"visible", "degraded", "blind"} and
        "gate" not in skipped else "unavailable",
        "weather": "available" if result.get("weather") is not None and
        "weather" not in skipped else "unavailable",
        "object_detection": "available" if perception.get("object_detection_available") is True
        and "perception" not in skipped else "unavailable",
        "road_obstacle_detection": "available" if
        perception.get("road_obstacle_detection_available") is True and
        "perception" not in skipped else "unavailable",
        "advisory": "available" if isinstance(advisory, dict) and
        type(advisory.get("should_takeover")) is bool and advisory.get("text") else "unavailable",
    }
    if blind:
        for name in ("weather", "object_detection", "road_obstacle_detection"):
            branches[name] = "intentionally_skipped_blind"
    allowed_skips = {"perception", "advisory_perception_path"} if blind else set()
    successful = (
        result_error is None and not (set(skipped) - allowed_skips)
        and all(state != "unavailable" for state in branches.values())
    )
    return {
        "branches": branches, "visibility_level": level,
        "route_success": successful,
        "full_perception_success": successful and not blind,
        "blind": blind, "skipped": skipped,
    }


def _rate(frames: int, wall_ms: float) -> float | None:
    return frames * 1000.0 / wall_ms if wall_ms > 0 else None


def _row(scope: str, mode: str, phase: str, iteration: int, batch_size: int,
         count: int, wall_ms: float, states: list[dict], error: str | None,
         **extra: Any) -> dict:
    success = sum(state["route_success"] for state in states)
    full = sum(state["full_perception_success"] for state in states)
    return {
        "scope": scope, "input_mode": mode, "phase": phase, "iteration": iteration,
        "requested_batch_size": batch_size,
        "attempted_frame_count": count, "route_success_frame_count": success,
        "full_perception_frame_count": full,
        "blind_frame_count": sum(state["blind"] for state in states),
        "unavailable_frame_count": count - success, "wall_ms": wall_ms,
        "attempted_throughput_fps": _rate(count, wall_ms),
        "complete_route_throughput_fps": _rate(count, wall_ms) if success == count else None,
        "full_perception_throughput_fps": _rate(count, wall_ms) if full == count else None,
        "error": error, **extra,
    }


def _run_pass(pipe: Any, inputs: list, paths: list[Path], torch: Any, devices: list[int],
              mode: str, phase: str, iteration: int, batch_size: int,
              measurements: list[dict], frames: list[dict]) -> None:
    # Hold raw results only for this pass; conversion/JSON work is outside the pass timer.
    calls = []
    _sync(torch, devices)
    pass_started = time.perf_counter_ns()
    for start in range(0, len(inputs), batch_size):
        batch = inputs[start:start + batch_size]
        _sync(torch, devices)
        started = time.perf_counter_ns()
        error = None
        results = None
        try:
            results = pipe.run_batch(batch)
            if not isinstance(results, list) or len(results) != len(batch):
                raise RuntimeError("run_batch must return one result per input in order")
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
            results = None
        _sync(torch, devices)
        elapsed = (time.perf_counter_ns() - started) / 1_000_000
        calls.append((start, len(batch), elapsed, results, error))
    pass_ms = (time.perf_counter_ns() - pass_started) / 1_000_000
    all_states = []
    for batch_index, (start, count, wall_ms, results, error) in enumerate(calls):
        states = []
        for offset in range(count):
            result = None
            result_error = error
            nonfinite: list[str] = []
            if results is not None:
                try:
                    result = results[offset].to_dict()
                    result["timings_ms"] = dict(results[offset].timings_ms)
                    result = _safe_result(result, nonfinite)
                    # Refuse unsupported values before writing strict JSON artifacts.
                    json.dumps(result, ensure_ascii=False, allow_nan=False)
                    if nonfinite:
                        result_error = "Non-finite result values replaced with null; see field paths"
                except Exception as exc:  # noqa: BLE001
                    result_error = f"{type(exc).__name__}: {exc}"
                    result = None
            state = _branch_state(result, result_error)
            states.append(state)
            frames.append({
                **(result or {}),
                "input_mode": mode, "phase": phase, "iteration": iteration,
                "batch_index": batch_index, "input_index": start + offset,
                "input_name": str(paths[start + offset]),
                "batch_wall_ms": wall_ms, "actual_batch_size": count,
                "result_error": result_error,
                "nonfinite_fields_replaced_with_null": nonfinite, **state,
            })
        all_states.extend(states)
        measurements.append(_row(
            "batch", mode, phase, iteration, batch_size, count, wall_ms, states, error,
            batch_index=batch_index, input_start=start, actual_batch_size=count, call_count=1,
        ))
    measurements.append(_row(
        "sample_set", mode, phase, iteration, batch_size, len(inputs), pass_ms,
        all_states, "; ".join(call[4] for call in calls if call[4]) or None,
        batch_index=None, input_start=0, actual_batch_size=None, call_count=len(calls),
    ))


def _distribution(values: list[float], unit: str) -> dict:
    ordered = sorted(values)

    def quantile(q: float) -> float | None:
        if not ordered:
            return None
        position = (len(ordered) - 1) * q
        low, high = math.floor(position), math.ceil(position)
        return ordered[low] + (ordered[high] - ordered[low]) * (position - low)

    return {
        "sample_count": len(ordered), "unit": unit, "quantile_method": "linear (n-1)*q",
        "min": ordered[0] if ordered else None, "max": ordered[-1] if ordered else None,
        "mean": sum(ordered) / len(ordered) if ordered else None,
        "p50": quantile(0.5), "p95": quantile(0.95), "p99": quantile(0.99),
    }


def _aggregate(rows: list[dict]) -> dict:
    def totals(selected: list[dict]) -> dict:
        frames = sum(row["attempted_frame_count"] for row in selected)
        wall = sum(row["wall_ms"] for row in selected)
        return {
            "observation_count": len(selected), "frame_count": frames,
            "total_wall_ms": wall, "frames_per_second": _rate(frames, wall),
            "wall_ms": _distribution([row["wall_ms"] for row in selected], "ms"),
            "throughput_fps": _distribution(
                [row["attempted_throughput_fps"] for row in selected
                 if row["attempted_throughput_fps"] is not None], "frames/s"),
        }

    complete = [row for row in rows
                if row["route_success_frame_count"] == row["attempted_frame_count"]]
    full = [row for row in rows
            if row["full_perception_frame_count"] == row["attempted_frame_count"]]
    return {
        "all_attempts_including_unavailable": totals(rows),
        "complete_routes_only_including_blind": totals(complete),
        "full_perception_only_excluding_blind": totals(full),
        "unavailable_frame_count": sum(row["unavailable_frame_count"] for row in rows),
        "blind_frame_count": sum(row["blind_frame_count"] for row in rows),
        "eligibility": "Only whole observations with every frame successful enter complete/full "
        "throughput. All-attempt throughput includes failures and is not healthy-pipeline throughput.",
    }


def _summary(measurements: list[dict], frames: list[dict], construction_ms: float | None,
             fatal: str | None, array_error: str | None, files_changed: bool) -> dict:
    batches = [row for row in measurements if row["scope"] == "batch"]
    first = next((row for row in batches if row["input_mode"] == "paths"), None)
    modes = {}
    for mode in ("paths", "arrays"):
        modes[mode] = {}
        for phase in ("first_pass", "warmup", "steady"):
            selected = [row for row in measurements
                        if row["input_mode"] == mode and row["phase"] == phase]
            batch_rows = [row for row in selected if row["scope"] == "batch"]
            modes[mode][phase] = {
                "batch_observations": _aggregate(batch_rows),
                "whole_sample_set_observations": _aggregate(
                    [row for row in selected if row["scope"] == "sample_set"]),
                "by_actual_batch_size": {
                    str(size): _aggregate(
                        [row for row in batch_rows if row["actual_batch_size"] == size])
                    for size in sorted({row["actual_batch_size"] for row in batch_rows})
                },
            }
    branch_names = (
        "input", "gate", "weather", "object_detection", "road_obstacle_detection", "advisory",
    )
    unavailable = sum(not frame["route_success"] for frame in frames)
    return {
        "schema_version": 1,
        "status": "incomplete" if fatal or array_error or unavailable or files_changed else "complete",
        "fatal_error": fatal, "array_mode_unavailable_reason": array_error,
        "files_changed_or_unreadable": files_changed,
        "model_construction_wall_ms": construction_ms,
        "first_pipeline_call": first,
        "cold_start_model_construction_plus_first_batch_ms": (
            construction_ms + first["wall_ms"] if construction_ms is not None and first else None
        ),
        "cold_start_scope": "This process's construction + very first run_batch (one actual batch). "
        "First full-sample-set pass is separate. Imports, Python startup, CUDA instrumentation "
        "initialization and OS cache clearing are excluded. Arrays reuse the path-warmed model.",
        "modes": modes,
        "decode": {
            "per_image_attempts_including_failures": _distribution(
                [row["wall_ms"] for row in measurements if row["scope"] == "image_decode"], "ms"),
            "per_image_successful": _distribution(
                [row["wall_ms"] for row in measurements if row["scope"] == "image_decode"
                 and not row["error"]], "ms"),
            "whole_sample_set_wall_ms": next(
                (row["wall_ms"] for row in measurements if row["scope"] == "decode_sample_set"), None),
            "failed_image_count": sum(bool(row["error"]) for row in measurements
                                      if row["scope"] == "image_decode"),
            "definition": "Independent _to_array(path), including decode/RGB conversion, "
            "validation and materialization. No imputed or substituted images.",
        },
        "frame_observation_count": len(frames),
        "unavailable_frame_observation_count": unavailable,
        "branch_frame_counts": {
            branch: {"denominator": len(frames), "counts": dict(Counter(
                frame["branches"][branch] for frame in frames))}
            for branch in branch_names
        },
        "visibility_frame_counts": dict(Counter(frame["visibility_level"] for frame in frames)),
        "skipped_key_frame_counts": dict(Counter(
            key for frame in frames for key in frame["skipped"])),
        "branch_distribution_scope": "All first-pass, warmup and stable frame observations; "
        "use frames.json identifiers and mode/phase for a narrower view.",
        "unmeasured_stages": [
            "resize", "CPU information features", "host/device transfer",
            "individual network forward passes", "NMS",
        ],
        "limitations": LIMITATIONS,
    }


def _write_json(path: Path, payload: Any) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    images = [_path(text) for text in args.image]
    output = _path(args.output_dir)
    # Reserve a new directory before any model construction. Never replace prior evidence.
    if output.exists():
        raise FileExistsError(f"Output directory already exists: {output}")
    for path in [*images, _path(args.config)]:
        if not path.is_file():
            raise FileNotFoundError(f"Explicit input is not a file: {path}")
    output.mkdir(parents=True, exist_ok=False)
    metadata: dict[str, Any] = {
        "schema_version": 1, "started_utc": datetime.now(UTC).isoformat(),
        "project_root": str(ROOT), "cwd": str(Path.cwd()),
        "parameters": vars(args), "relative_path_base": str(ROOT),
        "input_count": len(images), "input_order": [str(path) for path in images],
        "input_mode_order": ["paths", "arrays"],
        "phase_order_per_mode": ["first_pass", "warmup", "steady"],
        "batch_wall_scope": "Synchronized run_batch call, including its input conversion and "
        "all executed branches. Frame conversion/JSON/CSV and file hashing are outside timers.",
        "sample_set_wall_scope": "All chunked run_batch calls, synchronizations and loop bookkeeping "
        "for the full ordered sample set; result conversion and artifact writing excluded.",
        "gpu_peak_scope": "Reset immediately before construction on the resolved CUDA device; "
        "through first calls, warmups and stable runs. No empty_cache call.",
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "platform": platform.platform(), "machine": platform.machine(),
        "processor": platform.processor(), "python": sys.version,
        "versions": {},
        "ram_samples": [{"phase": "before_dependency_imports", **_ram()}],
        "limitations": LIMITATIONS,
    }
    measurements: list[dict] = []
    frames: list[dict] = []
    refs = {"benchmark_script": Path(__file__).resolve(), "visibility_config": _path(args.config)}
    refs.update({f"input_{index}": path for index, path in enumerate(images)})
    initial = {role: _stat(path) for role, path in refs.items()}
    construction_ms = None
    fatal = None
    array_error = "Array mode has not run"
    torch = None
    pipe = None
    devices: list[int] = []
    try:
        import torch
        import yaml

        sys.path.insert(0, str(ROOT / "src"))
        from car_smart_assist.data.image_input import ImageFrame
        from car_smart_assist.inference.pipeline import InferencePipeline, _to_array
        from car_smart_assist.perception.weather import select_device

        value_range = getattr(args, "image_range", None)
        image_inputs = [
            ImageFrame(path, value_range=tuple(value_range)) if value_range else path for path in images
        ]

        for package in ("torch", "torchvision", "numpy", "Pillow", "PyYAML", "ultralytics",
                        "car-smart-assist"):
            try:
                metadata["versions"][package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                metadata["versions"][package] = None
        metadata["torch_cuda_runtime"] = torch.version.cuda
        metadata["torch_cpu_threads"] = torch.get_num_threads()
        visibility = yaml.safe_load(_path(args.config).read_text(encoding="utf-8"))["visibility"]
        if not isinstance(visibility, dict):
            raise ValueError("The visibility configuration must be a mapping")
        new_refs = _artifact_paths(args, visibility, yaml)
        refs.update(new_refs)
        for role, path in new_refs.items():
            initial.setdefault(role, _stat(path))
        selected = select_device(args.device.strip().lower())
        if selected.type == "cuda" and selected.index is None:
            selected = torch.device("cuda", torch.cuda.current_device())
        metadata["resolved_device_override"] = str(selected)
        devices = _cuda_devices(torch, selected, {})
        _sync(torch, devices)
        metadata["gpu_before_construction"] = _gpu_memory(torch, devices, round_peak=False)
        for index in devices:
            torch.cuda.reset_peak_memory_stats(index)
        metadata["ram_samples"].append({"phase": "before_construction", **_ram()})
        started = time.perf_counter_ns()
        pipe = InferencePipeline.from_config(
            visibility, project_root=ROOT,
            checkpoint=_path(args.checkpoint) if args.checkpoint else None,
            device=str(selected),
            weather_config=_path(args.weather_config) if args.weather_config else None,
            weather_checkpoint=_path(args.weather_checkpoint) if args.weather_checkpoint else None,
            weather_loss=args.weather_loss,
        )
        _sync(torch, devices)
        construction_ms = (time.perf_counter_ns() - started) / 1_000_000
        metadata["components_after_construction"] = _components(pipe)
        observed_devices = _cuda_devices(torch, selected, metadata["components_after_construction"])
        if set(observed_devices) - set(devices):
            raise RuntimeError("A constructed model uses a CUDA device outside the explicit override")
        metadata["synchronized_cuda_devices"] = devices
        metadata["ram_samples"].append({"phase": "after_construction", **_ram()})
        for phase, passes in (("first_pass", 1), ("warmup", args.warmup), ("steady", args.repeats)):
            for iteration in range(passes):
                _run_pass(pipe, image_inputs, images, torch, devices, "paths", phase, iteration,
                          args.batch_size, measurements, frames)
        metadata["components_after_path_calls"] = _components(pipe)
        observed_devices = _cuda_devices(torch, selected, metadata["components_after_path_calls"])
        if set(observed_devices) - set(devices):
            raise RuntimeError("A lazy model uses a CUDA device outside the explicit override")
        metadata["ram_samples"].append({"phase": "after_path_runs", **_ram()})
        arrays = []
        image_shapes = []
        decode_errors = []
        decode_started = time.perf_counter_ns()
        for index, path in enumerate(images):
            started = time.perf_counter_ns()
            error = None
            try:
                array = _to_array(image_inputs[index])
                arrays.append(array)
                image_shapes.append({"input_index": index, "path": str(path),
                                     "height": array.shape[0], "width": array.shape[1],
                                     "channels": array.shape[2], "dtype": str(array.dtype)})
            except Exception as exc:  # noqa: BLE001
                error = f"{type(exc).__name__}: {exc}"
                decode_errors.append({"input_index": index, "path": str(path), "error": error})
            measurements.append({
                "scope": "image_decode", "input_mode": "decode", "phase": "predecode",
                "iteration": 0, "input_start": index, "actual_batch_size": 1,
                "attempted_frame_count": 1,
                "wall_ms": (time.perf_counter_ns() - started) / 1_000_000, "error": error,
            })
        measurements.append({
            "scope": "decode_sample_set", "input_mode": "decode", "phase": "predecode",
            "iteration": 0, "attempted_frame_count": len(images),
            "wall_ms": (time.perf_counter_ns() - decode_started) / 1_000_000,
            "error": f"{len(decode_errors)} image(s) failed decoding" if decode_errors else None,
        })
        metadata["decoded_image_shapes"] = image_shapes
        metadata["decode_errors"] = decode_errors
        metadata["ram_samples"].append({"phase": "after_predecode_arrays_retained", **_ram()})
        if decode_errors:
            array_error = "At least one explicit image failed decoding; array mode was not run. "
            array_error += "No replacement image or black frame was inserted."
        else:
            array_error = None
            for phase, passes in (("first_pass", 1), ("warmup", args.warmup), ("steady", args.repeats)):
                for iteration in range(passes):
                    _run_pass(pipe, arrays, images, torch, devices, "arrays", phase, iteration,
                              args.batch_size, measurements, frames)
        metadata["components_after_all_calls"] = _components(pipe)
    except Exception as exc:  # noqa: BLE001
        fatal = f"{type(exc).__name__}: {exc}"
    metadata["ram_samples"].append({"phase": "after_runs_arrays_and_results_retained", **_ram()})
    if torch is not None:
        try:
            _sync(torch, devices)
            metadata["gpu_after_runs"] = _gpu_memory(torch, devices)
        except Exception as exc:  # noqa: BLE001
            metadata["gpu_memory_unavailable_reason"] = f"{type(exc).__name__}: {exc}"
            fatal = fatal or metadata["gpu_memory_unavailable_reason"]
    else:
        metadata["gpu_after_runs"] = None
        metadata["gpu_memory_unavailable_reason"] = "Torch was not imported successfully"
    # Hash after timing to avoid warming weight/image file caches with benchmark fingerprinting.
    metadata["files"] = {
        role: _fingerprint(path, initial.get(role)) for role, path in refs.items()
    }
    metadata["file_identity_scope"] = "SHA256 after timing; pre-run and pre/post-hash size/mtime "
    metadata["file_identity_scope"] += "are compared. No content changes are authorized."
    metadata["finished_utc"] = datetime.now(UTC).isoformat()
    files_changed = any(record["changed_during_run"] for record in metadata["files"].values())
    summary = _summary(measurements, frames, construction_ms, fatal, array_error, files_changed)
    metadata["status"] = summary["status"]
    _write_json(output / "metadata.json", metadata)
    _write_json(output / "summary.json", summary)
    _write_json(output / "frames.json", {"frames": frames})
    with (output / "measurements.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(measurements)
    print(json.dumps({"output_dir": str(output), "status": summary["status"],
                      "fatal_error": fatal}, ensure_ascii=False, allow_nan=False))
    return 0 if summary["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())

