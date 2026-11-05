"""Instance-local timing for a separate diagnostic pass, never a throughput benchmark."""
from __future__ import annotations

from contextlib import AbstractContextManager
from functools import wraps
from time import perf_counter_ns


class PipelineStageProfiler(AbstractContextManager):
    def __init__(self, pipe, synchronize, clock=perf_counter_ns):
        self.pipe, self.synchronize, self.clock = pipe, synchronize, clock
        self.events = []
        self._restore = []
        self._wrapped = set()
        self._depth = 0
        self.frame_index = None
        self.iteration = None

    def _wrap(self, obj, method, stage):
        if obj is None or not callable(getattr(obj, method, None)):
            return
        key = (id(obj), method)
        if key in self._wrapped:
            return
        original = getattr(obj, method)
        existed = method in vars(obj)
        own_value = vars(obj).get(method)

        @wraps(original)
        def timed(*args, **kwargs):
            if stage == "yolo_total":
                self._install_yolo()
            self.synchronize()
            started = self.clock()
            depth = self._depth
            self._depth += 1
            error = None
            output = None
            try:
                output = original(*args, **kwargs)
                return output
            except BaseException as exc:
                error = type(exc).__name__
                raise
            finally:
                self._depth -= 1
                try:
                    self.synchronize()
                finally:
                    elapsed = (self.clock() - started) / 1_000_000
                    event = {"stage": stage, "frame_index": self.frame_index, "iteration": self.iteration,
                             "depth": depth, "inclusive_wall_ms": elapsed, "error": error}
                    if stage == "yolo_preprocess" and output is not None:
                        event.update(input_shape=list(output.shape), input_dtype=str(output.dtype),
                                     input_device=str(output.device))
                    self.events.append(event)

        setattr(obj, method, timed)
        self._restore.append((obj, method, existed, own_value))
        self._wrapped.add(key)

    def _install_yolo(self):
        predictor = getattr(self.pipe, "predictor", None)
        engine = getattr(getattr(predictor, "model", None), "predictor", None)
        for method, stage in (
            ("pre_transform", "yolo_letterbox"),
            ("preprocess", "yolo_preprocess"),
            ("inference", "yolo_forward"),
            ("postprocess", "yolo_postprocess"),
        ):
            self._wrap(engine, method, stage)

    def __enter__(self):
        try:
            self._wrap(self.pipe, "run_batch", "pipeline_total")
            self._wrap(self.pipe, "_run_gate", "gate_total")
            self._wrap(self.pipe, "_resize_for_gate", "gate_resize")
            self._wrap(getattr(self.pipe, "scorer", None), "score_arrays", "gate_score")
            self._wrap(getattr(self.pipe, "weather_predictor", None), "predict", "weather_total")
            self._wrap(getattr(self.pipe, "predictor", None), "predict", "yolo_total")
            self._wrap(getattr(self.pipe, "generator", None), "generate", "advisory")
            self._install_yolo()
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *exc):
        for obj, method, existed, value in reversed(self._restore):
            if existed:
                setattr(obj, method, value)
            else:
                delattr(obj, method)
        self._restore.clear()
        self._wrapped.clear()
        return False

    def report(self, distribution):
        stages = sorted({event["stage"] for event in self.events})
        return {
            "scope": "Separate warmed batch=1 diagnostic pass over predecoded arrays",
            "timing": "CUDA synchronized inclusive wall time; nested measurements MUST NOT be summed",
            "limitations": [
                "Instrumentation and extra synchronization inflate diagnostic wall time; use uninstrumented steady FPS.",
                "YOLO preprocess includes letterbox, layout conversion, host/device transfer and normalization.",
                "YOLO postprocess includes filtering, NMS, coordinate scaling and result construction.",
                "Individual transfers, kernels and NMS are not separately measured.",
                "No acquisition, queueing, display or vehicle-hardware latency is measured.",
            ],
            "stages": {
                stage: {
                    "successful_ms": distribution([e["inclusive_wall_ms"] for e in self.events
                                                    if e["stage"] == stage and e["error"] is None], "ms"),
                    "failed_calls": sum(e["stage"] == stage and e["error"] is not None for e in self.events),
                } for stage in stages
            },
            "missing_stages": [stage for stage in ("gate_total", "weather_total", "yolo_total",
                                                   "yolo_preprocess", "yolo_forward", "yolo_postprocess")
                               if stage not in stages],
            "events": self.events,
        }
