from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/benchmark_pipeline.py"


def _module():
    spec = importlib.util.spec_from_file_location("pipeline_benchmark", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _evidence_module():
    path = ROOT / "scripts/summarize_pipeline_evidence.py"
    spec = importlib.util.spec_from_file_location("pipeline_evidence_from_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _Result:
    timings_ms = {"gate": 2.0}

    @staticmethod
    def to_dict():
        return {
            "input": {"available": True},
            "visibility": {"level": "visible"},
            "weather": {"available": True},
            "perception": {
                "object_detection_available": True,
                "road_obstacle_detection_available": True,
                "objects": [],
            },
            "advisory": {"should_takeover": False, "text": "ok"},
            "skipped": {},
        }


class _Pipe:
    def __init__(self):
        self.batches = []

    def run_batch(self, batch):
        self.batches.append(list(batch))
        return [_Result() for _ in batch]


class _Cuda:
    def __init__(self):
        self.calls = []

    def synchronize(self, device):
        self.calls.append(device)


def test_run_pass_retains_short_final_batch_order_and_mode_identity(monkeypatch):
    module = _module()
    pipe = _Pipe()
    torch = SimpleNamespace(cuda=_Cuda())
    inputs = ["a", "b", "c", "d", "e"]
    paths = [Path(f"{name}.png") for name in inputs]
    measurements, frames = [], []
    ticks = iter(range(0, 100_000_000, 1_000_000))
    monkeypatch.setattr(module.time, "perf_counter_ns", lambda: next(ticks))

    module._run_pass(pipe, inputs, paths, torch, [0], "arrays", "steady", 2, 2,
                     measurements, frames)

    assert pipe.batches == [["a", "b"], ["c", "d"], ["e"]]
    assert [row["actual_batch_size"] for row in measurements if row["scope"] == "batch"] == [2, 2, 1]
    assert [row["input_index"] for row in frames] == [0, 1, 2, 3, 4]
    assert {row["input_mode"] for row in frames} == {"arrays"}
    assert {row["phase"] for row in frames} == {"steady"}
    assert {row["iteration"] for row in frames} == {2}
    assert len(torch.cuda.calls) == 7  # pass start plus before/after each of three batches
    assert all(row["route_success"] and row["full_perception_success"] for row in frames)
    summary, review_rows, _ = _evidence_module().build_report(
        json.dumps({"frames": frames}).encode("utf-8")
    )
    assert summary["frames"]["total"] == len(review_rows) == 5
    assert summary["frames"]["successful_empty"] == 5


def test_branch_states_separate_blind_unavailable_and_failure():
    module = _module()
    blind = {
        "input": {}, "visibility": {"level": "blind"}, "skipped": {"perception": "blind"},
        "perception": {"object_detection_available": False},
        "advisory": {"should_takeover": True, "text": "stop"},
    }
    state = module._branch_state(blind)
    assert state["blind"] is True
    assert state["route_success"] is True
    assert state["full_perception_success"] is False
    assert state["branches"]["object_detection"] == "intentionally_skipped_blind"

    unavailable = module._branch_state({
        "input": {}, "visibility": {"level": "visible"}, "skipped": {},
        "perception": {"object_detection_available": False},
        "advisory": {"should_takeover": False, "text": "ok"},
    })
    assert unavailable["route_success"] is False
    assert unavailable["branches"]["object_detection"] == "unavailable"

    failed = module._row("batch", "paths", "steady", 0, 2, 2, 5.0,
                         [unavailable, unavailable], "RuntimeError: failed")
    assert failed["complete_route_throughput_fps"] is None
    assert failed["error"] == "RuntimeError: failed"


def test_throughput_uses_total_frames_over_total_wall_and_distribution_counts():
    module = _module()
    rows = [
        module._row("batch", "paths", "steady", 0, 2, 2, 10.0, [], None),
        module._row("batch", "paths", "steady", 0, 2, 1, 20.0, [], None),
    ]
    aggregate = module._aggregate(rows)["all_attempts_including_unavailable"]
    assert aggregate["frame_count"] == 3
    assert aggregate["total_wall_ms"] == 30.0
    assert aggregate["frames_per_second"] == pytest.approx(100.0)
    assert aggregate["wall_ms"]["sample_count"] == 2
    assert aggregate["wall_ms"]["p50"] == pytest.approx(15.0)
    assert aggregate["wall_ms"]["p95"] == pytest.approx(19.5)
    assert aggregate["wall_ms"]["unit"] == "ms"


def test_sync_uses_explicit_devices_without_environment_mutation(monkeypatch):
    module = _module()
    cuda = _Cuda()
    torch = SimpleNamespace(cuda=cuda)
    before = __import__("os").environ.get("CUDA_VISIBLE_DEVICES")
    module._sync(torch, [0, 2])
    assert cuda.calls == [0, 2]
    assert __import__("os").environ.get("CUDA_VISIBLE_DEVICES") == before


def test_ram_and_allocator_memory_report_their_actual_scopes():
    module = _module()
    ram = module._ram()
    assert "working_set_bytes" in ram
    assert "process_lifetime_peak_working_set_bytes" in ram
    assert "Since process creation" in ram["peak_scope"]
    assert "unavailable_reason" in ram or "source" in ram

    class FakeCuda:
        @staticmethod
        def get_device_name(index):
            return f"mock-{index}"

        @staticmethod
        def memory_allocated(index):
            return 100 + index

        @staticmethod
        def memory_reserved(index):
            return 200 + index

        @staticmethod
        def max_memory_allocated(index):
            return 300 + index

        @staticmethod
        def max_memory_reserved(index):
            return 400 + index

    memory = module._gpu_memory(SimpleNamespace(cuda=FakeCuda()), [0])
    assert memory["0"]["allocated_bytes"] == 100
    assert memory["0"]["reserved_bytes"] == 200
    assert memory["0"]["peak_scope"] == "Round since reset before construction"
    previous = module._gpu_memory(SimpleNamespace(cuda=FakeCuda()), [0], round_peak=False)
    assert previous["0"]["peak_scope"] == "Process allocator peak before this round reset"


def test_strict_result_json_replaces_nonfinite_with_null_and_tracks_field():
    module = _module()
    nonfinite = []
    sanitized = module._safe_result({"x": [float("inf"), 1.0]}, nonfinite)
    assert sanitized == {"x": [None, 1.0]}
    assert nonfinite == ["$.x[0]"]
    json.dumps(sanitized, allow_nan=False)


def test_refuses_existing_output_before_loading_models(tmp_path, monkeypatch):
    module = _module()
    image = tmp_path / "image.png"
    config = tmp_path / "config.yaml"
    image.write_bytes(b"input")
    config.write_text("visibility: {}\n", encoding="utf-8")
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("original", encoding="utf-8")
    monkeypatch.setattr(module, "parse_args", lambda _argv=None: SimpleNamespace(
        image=[str(image)], config=str(config), output_dir=str(output),
    ))

    with pytest.raises(FileExistsError):
        module.main([])
    assert marker.read_text(encoding="utf-8") == "original"
