"""Timing oracles use a manually advanced clock, never measured hardware timings."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from car_smart_assist.inference.profiling import PipelineStageProfiler

ROOT = Path(__file__).resolve().parents[2]


def benchmark():
    spec = importlib.util.spec_from_file_location("baseline_benchmark_test", ROOT / "scripts/benchmark_pipeline.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Clock:
    def __init__(self):
        self.ns = 0
    def advance(self, ms):
        self.ns += int(ms * 1_000_000)
    def __call__(self):
        return self.ns
    def sync(self):
        self.advance(1)


class Engine:
    def __init__(self, clock, fail=False):
        self.clock, self.fail = clock, fail
        self.output = object()
    def pre_transform(self, x):
        self.clock.advance(3)
        return x
    def preprocess(self, x):
        self.pre_transform(x)
        self.clock.advance(4)
        return SimpleNamespace(shape=(1, 3, 32, 64), dtype="float32", device="cpu")
    def inference(self, x):
        self.clock.advance(5)
        if self.fail:
            raise RuntimeError("deliberate inference failure")
        return x
    def postprocess(self, x):
        self.clock.advance(6)
        return self.output


class Predictor:
    def __init__(self, engine):
        self.model = SimpleNamespace(predictor=engine)
    def predict(self, x):
        e = self.model.predictor
        return e.postprocess(e.inference(e.preprocess(x)))


class Pipe:
    def __init__(self, engine):
        self.predictor = Predictor(engine)
    def run_batch(self, values):
        return [self.predictor.predict(x) for x in values]


def test_inclusive_synchronized_timing_preserves_results_and_restores_methods():
    clock = Clock()
    engine = Engine(clock)
    pipe = Pipe(engine)
    with PipelineStageProfiler(pipe, clock.sync, clock) as profiler:
        profiler.frame_index = 7
        profiler.iteration = 2
        result = pipe.run_batch(["frame"])
    assert result[0] is engine.output
    events = {e["stage"]: e for e in profiler.events}
    # Eighteen ms of work, twelve one-ms synchronizations; outer initial sync is excluded.
    assert clock.ns == 30_000_000
    assert events["pipeline_total"]["inclusive_wall_ms"] == 29
    assert events["yolo_total"]["inclusive_wall_ms"] == 27
    assert events["yolo_preprocess"]["inclusive_wall_ms"] == 10
    assert events["yolo_letterbox"]["inclusive_wall_ms"] == 4
    assert events["yolo_forward"]["inclusive_wall_ms"] == 6
    assert events["yolo_postprocess"]["inclusive_wall_ms"] == 7
    assert events["yolo_preprocess"]["input_shape"] == [1, 3, 32, 64]
    assert events["yolo_preprocess"]["input_dtype"] == "float32"
    assert events["yolo_forward"]["depth"] == 2
    assert all(e["frame_index"] == 7 and e["iteration"] == 2 for e in profiler.events)
    assert "run_batch" not in vars(pipe)
    assert "predict" not in vars(pipe.predictor)
    assert all(name not in vars(engine) for name in ("preprocess", "inference", "postprocess", "pre_transform"))
    assert pipe.run_batch(["again"])[0] is engine.output


def test_exception_restores_existing_instance_override_and_records_failure():
    clock = Clock()
    engine = Engine(clock, fail=True)
    pipe = Pipe(engine)
    own_predict = pipe.predictor.predict
    pipe.predictor.predict = own_predict
    profiler = PipelineStageProfiler(pipe, clock.sync, clock)
    with pytest.raises(RuntimeError, match="deliberate inference failure"):
        with profiler:
            pipe.run_batch(["frame"])
    assert pipe.predictor.predict is own_predict
    assert "run_batch" not in vars(pipe)
    errors = {e["stage"]: e["error"] for e in profiler.events if e["error"]}
    assert errors == {"yolo_forward": "RuntimeError", "yolo_total": "RuntimeError", "pipeline_total": "RuntimeError"}
    report = profiler.report(benchmark()._distribution)
    assert report["stages"]["yolo_forward"]["successful_ms"]["sample_count"] == 0
    assert report["stages"]["yolo_forward"]["failed_calls"] == 1


def test_image_manifest_is_explicit_ordered_and_requires_valid_paths_shape(tmp_path):
    module = benchmark()
    path = tmp_path / "images.json"
    path.write_text(json.dumps(["data/b.png", "data/a.png"]), encoding="utf-8")
    args = module.parse_args(["--image", "data/first.png", "--image-list", str(path),
                              "--output-dir", "unused", "--profile-stages"])
    assert args.image == ["data/first.png", "data/b.png", "data/a.png"]
    assert args.profile_stages is True
    path.write_text('{"images": []}', encoding="utf-8")
    with pytest.raises(SystemExit):
        module.parse_args(["--image-list", str(path), "--output-dir", "unused"])




def test_full_benchmark_keeps_diagnostic_passes_out_of_steady_statistics(tmp_path, monkeypatch):
    from PIL import Image
    from car_smart_assist.inference.pipeline import InferencePipeline

    module = benchmark()
    monkeypatch.setattr(module, "ROOT", tmp_path)
    config = tmp_path / "visibility.yaml"
    config.write_text("visibility: {}\n", encoding="utf-8")
    images = [tmp_path / "one.png", tmp_path / "two.png"]
    for path in images:
        Image.new("RGB", (64, 32), (100, 120, 140)).save(path)
    image_list = tmp_path / "images.json"
    image_list.write_text(json.dumps([str(p) for p in images]), encoding="utf-8")

    class Result:
        timings_ms = {}
        def to_dict(self):
            return {
                "visibility": {"level": "visible"}, "weather": {"available": True},
                "perception": {"object_detection_available": True, "road_obstacle_detection_available": True,
                               "objects": []},
                "advisory": {"should_takeover": False, "text": "fixture"}, "skipped": {},
            }

    class CompletePipe(Pipe):
        def __init__(self, engine):
            super().__init__(engine)
            self.weather_predictor = SimpleNamespace(predict=lambda value: None)
            self.generator = SimpleNamespace(generate=lambda: None)
        def _run_gate(self, images):
            return None
        def run_batch(self, images):
            self._run_gate(images)
            for value in images:
                self.weather_predictor.predict(value)
                self.predictor.predict(value)
                self.generator.generate()
            return [Result() for _ in images]

    engine = Engine(Clock())
    pipe = CompletePipe(engine)
    monkeypatch.setattr(InferencePipeline, "from_config", lambda *a, **k: pipe)
    monkeypatch.setattr(module, "_artifact_paths", lambda *a: {})
    monkeypatch.setattr(module, "_components", lambda *a: {})
    output = tmp_path / "benchmark"
    assert module.main(["--image-list", str(image_list), "--config", str(config), "--device", "cpu",
                        "--warmup", "1", "--repeats", "2", "--profile-stages", "--profile-repeats", "3",
                        "--output-dir", str(output)]) == 0
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    normal = json.loads((output / "frames.json").read_text(encoding="utf-8"))["frames"]
    diagnostic = json.loads((output / "stage_profile.json").read_text(encoding="utf-8"))
    assert len(normal) == 16  # Two modes * (first1 + warmup1 + steady2) * two images.
    assert len(diagnostic["frames"]) == 6  # Three separate diagnostic passes * two images.
    assert all(row["phase"] != "stage_diagnostic" for row in normal)
    assert {row["phase"] for row in diagnostic["frames"]} == {"stage_diagnostic"}
    assert summary["modes"]["arrays"]["steady"]["batch_observations"]["full_perception_only_excluding_blind"]["frame_count"] == 4
    assert diagnostic["missing_stages"] == []
    assert diagnostic["route_failures"] == 0
    assert "preprocess" not in vars(engine)


@pytest.mark.parametrize("value", ["nan", "inf", "0", "-1"])
def test_nonfinite_or_nonpositive_speed_target_is_rejected(value):
    with pytest.raises(SystemExit):
        benchmark().parse_args(["--image", "a.png", "--output-dir", "unused", "--target-fps", value])


@pytest.mark.parametrize("batch,count,fps,p95,healthy,expected", [
    (1, 200, 40., 30., True, "meets_on_observed_sample"),
    (1, 200, 40., 40., True, "does_not_meet_on_observed_sample"),
    (1, 200, 20., 30., True, "does_not_meet_on_observed_sample"),
    (2, 200, 80., 15., True, "inconclusive"),
    (1, 0, None, None, True, "inconclusive"),
    (1, 200, 80., 15., False, "inconclusive"),
])
def test_target_requires_single_frame_healthy_full_route_throughput_and_tail_latency(batch, count, fps, p95, healthy, expected):
    summary = {"status": "complete" if healthy else "incomplete", "modes": {
        "arrays": {"steady": {"batch_observations": {"full_perception_only_excluding_blind": {
            "frame_count": count, "frames_per_second": fps, "wall_ms": {"p95": p95},
        }}}}}}
    result = benchmark()._realtime_target(summary, batch, 30)
    assert result["status"] == expected
    assert result["p95_budget_ms"] == pytest.approx(100 / 3)
