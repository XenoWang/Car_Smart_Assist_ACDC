"""Synthetic regression coverage for the frozen P2/P3 control and evaluation fixes."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]


def _load(relative: str, name: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def small_cpu_pool():
    threads, rng = torch.get_num_threads(), torch.get_rng_state()
    torch.set_num_threads(2)
    try:
        yield
    finally:
        torch.set_num_threads(threads)
        torch.set_rng_state(rng)


class _Controller:
    def __init__(self, path: Path):
        self.path = path
        self.requested = False

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def require_running(self):
        if self.requested:
            raise SystemExit(75)

    def request(self):
        self.requested = True

    def resume(self):
        self.requested = False
        self.path.unlink(missing_ok=True)

    def watch_directory(self, _path):
        return None


def _write_yaml(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def test_fix_p3_01_calibration_pause_then_ordinary_resume_clears_stop(tmp_path, monkeypatch):
    helper = _load("tests/unit/test_training_stop_models.py", "control_stop_models")
    from car_smart_assist.utils.stop import StopController

    controller = StopController(tmp_path / "STOP", default_directory=tmp_path)
    training, cfg, datasets, split, training_batches = helper._attention_setup(
        tmp_path, monkeypatch, stop_controller=controller
    )
    original = training.balanced_thresholds
    pause_once = {"done": False, "calls": 0}

    def pause_during_calibration(dataset, *args, **kwargs):
        pause_once["calls"] += 1
        result = original(dataset, *args, **kwargs)
        if dataset is datasets["calibration"] and not pause_once["done"]:
            pause_once["done"] = True
            controller.request()
        return result

    monkeypatch.setattr(training, "balanced_thresholds", pause_during_calibration)
    _, paused = training.train_attention(
        cfg, tmp_path, datasets, split, fresh=True, stop_controller=controller
    )
    assert paused["manual_stop"] and paused["termination_reason"] == "manual_stop"
    stopped = torch.load(tmp_path / "checkpoints/last.pt", map_location="cpu", weights_only=True)
    assert stopped["manual_stop"] is True
    assert stopped["termination_reason"] == "manual_stop"
    assert stopped["epoch"] == 2 and stopped["planned_epochs"] == 3
    batches_at_pause = len(training_batches)
    assert batches_at_pause == 2

    controller.resume()
    _, resumed = training.train_attention(
        cfg, tmp_path, datasets, split, stop_controller=controller, continue_only=False
    )
    last = torch.load(tmp_path / "checkpoints/last.pt", map_location="cpu", weights_only=True)
    best = torch.load(tmp_path / "checkpoints/best.pt", map_location="cpu", weights_only=True)
    assert resumed["trained_through_epoch"] == 2
    assert resumed["training_complete"] is True
    assert resumed["manual_stop"] is False
    assert resumed["termination_reason"] == "early_stopping"
    assert resumed["thresholds"] == best["decision_thresholds"]
    assert best["calibration_signature"] == datasets["calibration"].signature
    assert "calibration" in best and best["calibration"]
    assert last["epoch"] == 2 and last["planned_epochs"] == 3
    assert last["training_complete"] is True
    assert last["manual_stop"] is False
    assert last["termination_reason"] == "early_stopping"
    assert len(training_batches) == batches_at_pause
    assert pause_once["calls"] == 4  # Two validation and two calibration passes.

def _weather_main_fixture(tmp_path, monkeypatch, *, compare):
    module = _load("scripts/train_weather_attention.py", "control_weather_cli")
    scripts = tmp_path / "scripts"
    scripts.mkdir(parents=True)
    monkeypatch.setattr(module, "__file__", str(scripts / "train_weather_attention.py"))
    root = tmp_path
    (root / "reports").mkdir(exist_ok=True)
    (root / "checkpoints").mkdir(exist_ok=True)
    cfg = SimpleNamespace(
        train={"swad": {"enabled": False}, "batch_size": 2, "channels_last": False, "l2_sp": 0},
        checkpoint_root="checkpoints", report_dir="reports", report_file="reports/REPORT.md",
        evaluation={}, backbone={"freeze_epochs": 0, "fine_tune_from": 0}, device="cpu",
    )
    module.main.__globals__["load_weather_attention_config"] = lambda *_: cfg
    module.main.__globals__["validate_weather_attention_config"] = lambda _cfg: None
    module.main.__globals__["require_independent_weather_reports"] = lambda *_: None
    module.main.__globals__["require_independent_weather_output"] = lambda *_: None
    module.main.__globals__["attention_model_config"] = lambda _cfg: {"synthetic": True}
    module.main.__globals__["prepare_attention_datasets"] = lambda *_: ({}, {"signature": "sig"})
    module.main.__globals__["train_attention"] = lambda *_args, **_kwargs: (
        object(), {"manual_stop": False, "best_epoch": 2}
    )
    (root / "reports/preflight.json").write_text(json.dumps({
        "model_config": {"synthetic": True}, "split_signature": "sig", "batch_size": 2,
        "channels_last": False, "swad": {"enabled": False}, "l2_sp_coefficient": 0,
    }), encoding="utf-8")
    controller = _Controller(root / "STOP")
    module.main.__globals__["compare_attention"] = compare(controller, root)
    module.main.__globals__["model_report"] = lambda *_: None
    return module, controller, root


def test_fix_p3_02_weather_main_late_stop_keeps_completed_comparison_report(tmp_path, monkeypatch):
    def compare(controller, root):
        def run(*_args, **_kwargs):
            (root / "reports/REPORT.md").write_text(
                "completed comparison\nnormal field: preserved\n", encoding="utf-8"
            )
            (root / "reports/progress.json").write_text(json.dumps({
                "status": "completed", "report": "REPORT.md", "epoch": 3,
            }), encoding="utf-8")
            controller.requested = True
            return {"status": "completed"}
        return run

    module, controller, root = _weather_main_fixture(tmp_path, monkeypatch, compare=compare)
    assert module.main(["--config", "unused.yaml"], stop_controller=controller) == 0
    assert (root / "reports/REPORT.md").read_text(encoding="utf-8") == (
        "completed comparison\nnormal field: preserved\n"
    )
    assert json.loads((root / "reports/progress.json").read_text(encoding="utf-8"))["status"] == "completed"


def test_fix_p3_02_weather_main_early_compare_stop_and_pipeline_boundary(tmp_path, monkeypatch):
    def compare(controller, _root):
        def run(*_args, **_kwargs):
            controller.request()
            raise SystemExit(75)
        return run

    module, controller, root = _weather_main_fixture(tmp_path, monkeypatch, compare=compare)
    assert module.main(["--config", "unused.yaml"], stop_controller=controller) == 75
    paused = json.loads((root / "reports/progress.json").read_text(encoding="utf-8"))
    assert paused["status"] == "手动停止" and paused["manual_stop"] is True
    assert paused["termination_reason"] == "manual_stop"

    pipeline = _load("scripts/train_pipeline.py", "control_pipeline_stop")
    work = tmp_path / "pipeline-work"
    work.mkdir()
    calls = []
    controller = _Controller(tmp_path / "pipeline-STOP")
    state = {"status": "RUNNING", "stages": [
        {"name": "weather", "status": "PENDING", "command": ["weather"], "attempts": [], "checkpoints": {}},
        {"name": "visibility", "status": "PENDING", "command": ["visibility"], "attempts": [], "checkpoints": {}},
    ]}
    class Child:
        pid = 123
        def wait(self):
            controller.requested = True
            return 0
    monkeypatch.setattr(pipeline, "save_state", lambda *_: None)
    monkeypatch.setattr(pipeline, "verify_identity", lambda *_: calls.append("identity"))
    monkeypatch.setattr(pipeline, "checkpoint_records", lambda *_: {"last": {"sha256": "synthetic"}})
    monkeypatch.setattr(pipeline, "subprocess", SimpleNamespace(Popen=lambda *_a, **_k: calls.append("spawn") or Child(), DEVNULL=object()))
    assert pipeline.execute(work, state, controller) == pipeline.STOP_EXIT_CODE
    assert calls.count("spawn") == 1
    assert state["stages"][0]["status"] == "COMPLETED"
    assert state["stages"][1]["status"] == "PENDING"


def test_fix_p3_03_cpu_yolo_state_saves_no_cuda_and_restores_without_gpu(tmp_path, monkeypatch):
    helper = _load("tests/unit/test_yolo_training_state.py", "control_yolo_state")
    from car_smart_assist.perception import yolo_training

    # Construct CPU optimizer state before simulating an available CUDA runtime.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    saved = helper.trainer(tmp_path)
    callbacks = {}
    model = SimpleNamespace(add_callback=lambda key, callback: callbacks.update({key: callback}))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: pytest.fail("CPU save queried CUDA RNG"))
    monkeypatch.setattr(yolo_training, "_cuda_device_ids", lambda: pytest.fail("CPU save queried CUDA UUIDs"))
    yolo_training.attach_training_state_callbacks(
        model, "synthetic-data", None, selection_contract=helper.SELECTION_CONTRACT,
        training_segment=helper.TRAINING_SEGMENT,
    )
    callbacks["on_model_save"](saved)
    state_path = tmp_path / "last_training_state.pt"
    disk_oracle = torch.load(state_path, map_location="cpu", weights_only=True)
    resume_state = torch.load(state_path, map_location="cpu", weights_only=True)
    assert disk_oracle["cuda_rng"] == resume_state["cuda_rng"] == []
    assert disk_oracle["cuda_device_ids"] == resume_state["cuda_device_ids"] == []
    assert disk_oracle["checkpoint_sha256"] == resume_state["checkpoint_sha256"]

    # Construct CPU optimizer state before simulating an available CUDA runtime.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    restored = helper.trainer(tmp_path)
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", lambda *_: pytest.fail("CPU restore queried CUDA RNG"))
    yolo_training.attach_training_state_callbacks(
        model, "synthetic-data", resume_state, selection_contract=helper.SELECTION_CONTRACT,
        training_segment=helper.TRAINING_SEGMENT,
    )
    callbacks["on_train_start"](restored)
    assert torch.equal(restored.model.weight, disk_oracle["model"]["weight"])
    assert restored._project_best_checkpoint_sha256 == disk_oracle["best_checkpoint_sha256"]


def test_fix_p3_04_split_identity_errors_are_distinct_and_match_still_scores(tmp_path, monkeypatch):
    module = _load("scripts/evaluate_visibility.py", "control_eval_visibility")
    import car_smart_assist.perception.visibility.dataset as dataset
    monkeypatch.setattr(dataset, "split_ref_indices", lambda *_a, **_k: tuple(
        np.asarray([i], dtype=np.int64) for i in range(4)
    ))
    monkeypatch.setattr(dataset, "reference_split_signature", lambda *_a, **_k: "expected")
    paths = [tmp_path / f"ref-{i}.png" for i in range(4)]
    for path in paths:
        path.write_bytes(b"synthetic")
    refs = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in paths]
    cfg = {"split_by_sequence": False}
    gate = SimpleNamespace(judge_many=lambda scores: ["ok" for _ in scores])
    monkeypatch.setattr(module, "summarize", lambda _values: {"blind": 0.0})

    for signature, phrase in ((None, "缺少 split_signature"), ("stale", "身份不一致")):
        scorer = SimpleNamespace(input_size=(4, 4), split_signature=signature,
                                 score_arrays=lambda *_: pytest.fail("scored before identity check"))
        with pytest.raises(module.SplitIdentityError, match=phrase):
            module.eval_fit_quality(scorer, refs, paths, cfg, gate, 1)

    calls = []
    scorer = SimpleNamespace(
        input_size=(4, 4), split_signature="expected",
        score_arrays=lambda *_: calls.append(1) or [SimpleNamespace(recon_mean=1.0, information=0.5)],
    )
    result = module.eval_fit_quality(scorer, refs, paths, cfg, gate, 1)
    assert result["test_false_positive_rate"] == 0.0 and len(calls) == 4


def test_fix_p3_04_cli_returns_two_without_traceback_or_report_on_missing_identity(tmp_path, monkeypatch, capsys):
    module = _load("scripts/evaluate_visibility.py", "control_eval_cli")
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    monkeypatch.setattr(module, "__file__", str(scripts / "evaluate_visibility.py"))
    _write_yaml(tmp_path / "configs/model/visibility.yaml", {"visibility": {
        "data": {"acdc_root": "raw", "split_by_sequence": False},
        "train": {"checkpoint_dir": "checkpoints"}, "report": {"output_dir": "reports"},
    }})
    ckpt = tmp_path / "checkpoint.pt"
    ckpt.write_bytes(b"synthetic checkpoint placeholder")
    ref_root = tmp_path / "raw"
    ref_root.mkdir()
    refs = [ref_root / f"ref-{i}.png" for i in range(20)]
    for path in refs:
        path.write_bytes(b"synthetic image")
    monkeypatch.setattr(module.VisibilityScorer, "from_checkpoint", classmethod(
        lambda _cls, *_a, **_k: SimpleNamespace(
            input_size=(4, 4), split_signature=None, scoring_cfg={}, calibration=None,
        )
    ))
    monkeypatch.setattr(module, "list_ref_images", lambda *_a, **_k: refs)
    monkeypatch.setattr(module, "load_images", lambda *_a, **_k: [np.zeros((4, 4, 3), np.uint8)] * len(refs))
    monkeypatch.setattr(sys, "argv", ["evaluate_visibility.py", "--config", "configs/model/visibility.yaml",
                                      "--checkpoint", str(ckpt)])
    assert module.main() == 2
    assert "Traceback" not in capsys.readouterr().err
    assert not (tmp_path / "reports").exists()


    monkeypatch.setattr(module.VisibilityScorer, "from_checkpoint", classmethod(
        lambda _cls, *_a, **_k: SimpleNamespace(
            input_size=(4, 4), split_signature="valid", scoring_cfg={}, calibration=None,
        )
    ))
    monkeypatch.setattr(module, "eval_fit_quality", lambda *_: (_ for _ in ()).throw(ValueError("unrelated program error")))
    with pytest.raises(ValueError, match="unrelated program error"):
        module.main()

def test_rev01_checkpoint_scoring_snapshot_fallback_and_explicit_empty_override(tmp_path):
    helper = _load("tests/unit/test_visibility_best_identity.py", "control_visibility_identity")
    from car_smart_assist.perception.visibility.trainer import (
        VisibilityTrainer, _atomic_save, _model_state_sha256,
    )
    from car_smart_assist.perception.visibility.scorer import VisibilityScorer

    cfg = helper._config(tmp_path)
    cfg["scoring"] = {"aggregation": {"power": -2.0}}
    trainer = VisibilityTrainer(cfg, project_root=tmp_path, resume="none")
    trainer._best_model_sha256 = _model_state_sha256(trainer.model.state_dict())
    checkpoint = tmp_path / "visibility-checkpoint.pt"
    _atomic_save(trainer._make_payload(1, None, None, 0.1), checkpoint)
    fallback = VisibilityScorer.from_checkpoint(checkpoint, cfg=trainer.model_cfg, scoring_cfg=None)
    explicit_empty = VisibilityScorer.from_checkpoint(checkpoint, cfg=trainer.model_cfg, scoring_cfg={})
    assert fallback.scoring_cfg == cfg["scoring"]
    assert explicit_empty.scoring_cfg == {}


def test_rev01_evaluate_visibility_main_passes_none_for_omission_and_empty_for_override(
    tmp_path, monkeypatch
):
    module = _load("scripts/evaluate_visibility.py", "control_eval_scoring_callers")
    scripts = tmp_path / "scripts"
    scripts.mkdir(parents=True)
    monkeypatch.setattr(module, "__file__", str(scripts / "evaluate_visibility.py"))
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"synthetic placeholder")
    refs = [tmp_path / f"ref-{i}.png" for i in range(4)]
    for ref in refs:
        ref.write_bytes(b"synthetic")
    seen = []
    snapshot = {"aggregation": {"power": -2.0}}

    def scorer_factory(_cls, _checkpoint, *, scoring_cfg, **_kwargs):
        seen.append(scoring_cfg)
        return SimpleNamespace(
            input_size=(4, 4), calibration=None, scoring_cfg=(snapshot if scoring_cfg is None else scoring_cfg),
        )

    monkeypatch.setattr(module.VisibilityScorer, "from_checkpoint", classmethod(scorer_factory))
    monkeypatch.setattr(module.VisibilityGate, "from_config", classmethod(
        lambda _cls, _cfg: SimpleNamespace(thresholds=SimpleNamespace())
    ))
    monkeypatch.setattr(module, "list_ref_images", lambda *_a, **_k: refs)
    monkeypatch.setattr(module, "load_images", lambda *_a, **_k: [np.zeros((4, 4, 3), np.uint8)] * len(refs))
    monkeypatch.setattr(module, "eval_fit_quality", lambda *_: {"test_train_ratio": 1.0, "verdict": "synthetic"})
    monkeypatch.setattr(module, "eval_clean_control", lambda *_: {"false_positive_rate": 0.0})
    monkeypatch.setattr(module, "eval_real_adverse", lambda *_a, **_k: {})
    monkeypatch.setattr(module, "eval_synthetic", lambda *_: {
        kind: {"information_monotonic_decreasing": True, "recall_at_max_severity": 0.5}
        for kind in ("fog", "darkness", "occlusion", "blur")
    })
    monkeypatch.setattr(module, "write_report", lambda *_: None)

    for scoring in (None, {}):
        visibility = {"data": {"acdc_root": "raw", "split_by_sequence": False},
                      "train": {}, "report": {"output_dir": "reports"}}
        if scoring is not None:
            visibility["scoring"] = scoring
        _write_yaml(tmp_path / "configs/model/visibility.yaml", {"visibility": visibility})
        monkeypatch.setattr(sys, "argv", ["evaluate_visibility.py", "--config", "configs/model/visibility.yaml",
                                           "--checkpoint", str(checkpoint), "--samples", "1"])
        assert module.main() == 0
    assert seen == [None, {}]

def _minimal_ablation_setup(tmp_path, monkeypatch, *, tags=("alpha", "beta")):
    module = _load("scripts/run_visibility_ablations.py", "control_visibility_ablations")
    scripts = tmp_path / "scripts"
    scripts.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(module, "__file__", str(scripts / "run_visibility_ablations.py"))
    base = {
        "data": {"split_mode": "fixed"}, "model": {"input_size": [4, 4]},
        "train": {"epochs": 1}, "thresholds": {"use_recon_z": True},
        "ablations": [{"tag": tag, "set": {}} for tag in tags],
    }
    _write_yaml(tmp_path / "configs/model/visibility.yaml", {"visibility": base})
    input_calls = []

    def prepare(cfg, root):
        input_calls.append(cfg)
        return {"paths": [], "size": (4, 4), "splits": tuple(np.array([i]) for i in range(4)),
                "data_identity": {"signature": "synthetic-data"},
                "protocol": {"signature": "synthetic-protocol"}, "refs": []}

    monkeypatch.setattr(module, "_prepare_inputs", prepare)
    monkeypatch.setattr(module, "StopController", lambda *_a, **_k: _Controller(tmp_path / "STOP"))
    monkeypatch.setattr(module, "write_report", lambda *_a, **_k: 0)
    return module, base, input_calls


def test_rev01_callers_pass_omitted_scoring_as_none_and_empty_as_override(tmp_path, monkeypatch):
    module, base, _ = _minimal_ablation_setup(tmp_path, monkeypatch, tags=("alpha",))
    seen = []

    class FakeTrainer:
        def __init__(self, cfg, project_root, resume):
            self.cfg = cfg
            self.project_root = project_root

        def fit(self, **_kwargs):
            directory = tmp_path / self.cfg["train"]["checkpoint_dir"]
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "last.pt").write_bytes(b"last synthetic")
            (directory / "best.pt").write_bytes(b"best synthetic")
            return SimpleNamespace(manual_stop=False, train_loss=[0.1], best_epoch=1)

    def scorer_factory(_cls, checkpoint, *, device, cfg, scoring_cfg):
        seen.append(scoring_cfg)
        return SimpleNamespace(input_size=(4, 4), scoring_cfg=(
            {"aggregation": {"power": -2.0}} if scoring_cfg is None else scoring_cfg
        ))

    monkeypatch.setattr(module, "VisibilityTrainer", FakeTrainer)
    monkeypatch.setattr(module.VisibilityScorer, "from_checkpoint", classmethod(scorer_factory))
    monkeypatch.setattr(module, "load_refs", lambda *_: [])
    monkeypatch.setattr(module, "evaluate_run", lambda *_: {
        "params_m": 0.1, "overfit_ratio": 1.0, "val_fpr": 0.0, "mean_recall": 0.5,
    })
    monkeypatch.setattr(module.VisibilityGate, "from_config", classmethod(
        lambda _cls, _cfg: SimpleNamespace(thresholds=SimpleNamespace())
    ))
    monkeypatch.setattr(sys, "argv", ["run_visibility_ablations.py", "--only", "alpha"])
    assert module.main() == 0
    assert seen == [None]
    # The explicit-empty path is a direct caller invocation in the same run shape;
    # it must remain distinguishable from omission at the checkpoint boundary.
    base["scoring"] = {}
    seen.clear()
    _write_yaml(tmp_path / "configs/model/visibility.yaml", {"visibility": base})
    monkeypatch.setattr(sys, "argv", ["run_visibility_ablations.py", "--only", "alpha", "--force"])
    assert module.main() == 0
    assert seen == [{}]


def test_rev02_subset_preserves_suite_successful_subset_continue_and_identity_rejection(tmp_path, monkeypatch):
    module, base, _ = _minimal_ablation_setup(tmp_path, monkeypatch)
    out = tmp_path / "artifacts/reports/visibility/ablations"
    out.mkdir(parents=True)
    ckpt_root = tmp_path / "artifacts/checkpoints/visibility_ablations"
    identities, results, groups = {}, {}, {}
    for tag in ("baseline", "alpha", "beta"):
        cfg = module.apply_config_overrides(base, {})
        cfg.setdefault("train", {})["checkpoint_dir"] = f"artifacts/checkpoints/visibility_ablations/{tag}"
        data_identity, protocol = {"signature": "synthetic-data"}, {"signature": "synthetic-protocol"}
        identity = {"version": 1, "config": cfg, "data": data_identity,
                    "evaluation_protocol": protocol}
        signature = module._signature(identity)
        identities[tag] = signature
        if tag != "beta":
            results[tag] = {"experiment_signature": signature, "experiment_identity": identity,
                            "val_fpr": 0.0, "synth": {}}
        directory = ckpt_root / tag
        directory.mkdir(parents=True)
        last, best = directory / "last.pt", directory / "best.pt"
        last.write_bytes(f"last-{tag}".encode())
        best.write_bytes(f"best-{tag}".encode())
        groups[tag] = {"status": "completed",
                       "last_sha256": hashlib.sha256(last.read_bytes()).hexdigest(),
                       "best_sha256": hashlib.sha256(best.read_bytes()).hexdigest()}
    groups["beta"]["status"] = "paused"
    (out / "results.json").write_text(json.dumps(results), encoding="utf-8")
    progress = out / "training_progress.json"
    progress.write_text(json.dumps({"identity": identities, "groups": groups}), encoding="utf-8")

    monkeypatch.setattr(sys, "argv", ["run_visibility_ablations.py", "--only", "alpha"])
    assert module.main() == 0
    after_subset = json.loads(progress.read_text(encoding="utf-8"))
    assert after_subset["identity"] == identities
    assert after_subset["groups"]["baseline"] == groups["baseline"]
    assert after_subset["groups"]["beta"] == groups["beta"]

    # A valid selected-group continue accepts the unchanged alpha identity.
    monkeypatch.setattr(sys, "argv", ["run_visibility_ablations.py", "--continue", "--only", "alpha"])
    assert module.main() == 0
    before_rejection = json.loads(progress.read_text(encoding="utf-8"))
    corrupted = json.loads(json.dumps(before_rejection))
    corrupted["identity"]["alpha"] = "wrong-experiment-identity"
    progress.write_text(json.dumps(corrupted), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["run_visibility_ablations.py", "--continue", "--only", "alpha"])
    assert module.main() == 2
    after_rejection = json.loads(progress.read_text(encoding="utf-8"))
    assert after_rejection["groups"] == corrupted["groups"]
    assert {k: v for k, v in after_rejection["identity"].items() if k != "alpha"} == {
        k: v for k, v in before_rejection["identity"].items() if k != "alpha"
    }

    # Restore the approved identity. With no beta results row, complete-suite continue
    # must enter the paused beta trainer instead of silently reusing/skipping it.
    progress.write_text(json.dumps(before_rejection), encoding="utf-8")
    class StopAtBetaTrainer(Exception):
        pass
    trained_tags = []
    class TrainerSentinel:
        def __init__(self, cfg, **_kwargs):
            self.tag = Path(cfg["train"]["checkpoint_dir"]).name
            trained_tags.append(self.tag)
        def fit(self, **_kwargs):
            raise StopAtBetaTrainer
    monkeypatch.setattr(module, "VisibilityTrainer", TrainerSentinel)
    monkeypatch.setattr(sys, "argv", ["run_visibility_ablations.py", "--continue"])
    with pytest.raises(StopAtBetaTrainer):
        module.main()
    assert trained_tags == ["beta"]

def test_rev06_empty_only_exits_two_before_input_preparation_and_list_still_works(tmp_path, monkeypatch):
    module, _base, input_calls = _minimal_ablation_setup(tmp_path, monkeypatch, tags=("alpha",))
    monkeypatch.setattr(module, "_prepare_inputs", lambda *_: pytest.fail("prepared inputs before parser rejection"))
    monkeypatch.setattr(sys, "argv", ["run_visibility_ablations.py", "--only"])
    with pytest.raises(SystemExit) as error:
        module.main()
    assert error.value.code == 2
    monkeypatch.setattr(sys, "argv", ["run_visibility_ablations.py", "--list"])
    assert module.main() == 0
    assert input_calls == []


def test_rev07_attention_fork_rng_receives_explicit_cuda_zero_before_model_factory(tmp_path, monkeypatch):
    helper = _load("tests/unit/test_training_stop_models.py", "control_stop_models_rev07")
    training, cfg, datasets, split, _ = helper._attention_setup(tmp_path, monkeypatch)
    monkeypatch.setattr(training, "select_device", lambda *_: torch.device("cuda:0"))
    monkeypatch.setattr(training.torch, "manual_seed", lambda *_: None)
    monkeypatch.setattr(training.torch.cuda, "current_device", lambda: 1)
    captured = []

    @contextmanager
    def fork_rng(*, devices):
        captured.append(list(devices))
        yield

    class StopBeforeFactory(Exception):
        pass

    monkeypatch.setattr(training.torch.random, "fork_rng", fork_rng)
    monkeypatch.setattr(training, "WeatherAttention", lambda *_a, **_k: (_ for _ in ()).throw(StopBeforeFactory()))
    with pytest.raises(StopBeforeFactory):
        training._train_attention(
            cfg, tmp_path, datasets, split, fresh=True, stop_controller=_Controller(tmp_path / "STOP")
        )
    assert captured == [[0]]


def test_rev09_weather_cli_invalid_epoch_budgets_return_two_and_valid_override_reaches_preflight(
    tmp_path, monkeypatch, capsys
):
    module = _load("scripts/train_weather_attention.py", "control_weather_epoch_cli")
    from car_smart_assist.config.weather_attention import load_weather_attention_config

    config_path = ROOT / "configs/model/weather_attention_a.yaml"
    cfg = load_weather_attention_config(config_path)
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    monkeypatch.setattr(module, "__file__", str(scripts / "train_weather_attention.py"))
    monkeypatch.setattr(module, "load_weather_attention_config", lambda *_: cfg)
    monkeypatch.setattr(module, "require_independent_weather_reports", lambda *_: None)
    monkeypatch.setattr(module, "require_independent_weather_output", lambda *_: None)
    monkeypatch.setattr(module, "prepare_attention_datasets", lambda *_: ({}, {"signature": "synthetic"}))
    monkeypatch.setattr(module, "preflight_attention", lambda *_: {"status": "synthetic-preflight"})
    for value in ("0", "600"):
        assert module.main(["--preflight", "--config", "config.yaml", "--total-epochs", value]) == 2
        assert "Traceback" not in capsys.readouterr().err
    assert module.main(["--preflight", "--config", "config.yaml", "--total-epochs", "450"]) == 0


def test_rev10_from_config_resolves_relative_absolute_and_default_visibility_paths(tmp_path, monkeypatch):
    from car_smart_assist.inference.pipeline import InferencePipeline
    from car_smart_assist.perception.visibility.scorer import VisibilityScorer

    project = tmp_path / "project"
    cwd = tmp_path / "other-cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    seen = []

    def factory(_cls, checkpoint, **kwargs):
        seen.append(Path(checkpoint).resolve())
        return SimpleNamespace(input_size=(4, 4), calibration=None, scoring_cfg={}, split_signature=None)

    monkeypatch.setattr(VisibilityScorer, "from_checkpoint", classmethod(factory))
    cfg = {"model": {"input_size": [4, 4]}, "train": {"checkpoint_dir": "default-vis"}}
    rel = project / "relative" / "vis.pt"
    absolute = tmp_path / "absolute-vis.pt"
    default = project / "default-vis/best.pt"
    for path in (rel, absolute, default, cwd / "relative/vis.pt"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(path.name.encode())
    decoy = cwd / "relative/vis.pt"
    for checkpoint, expected in (("relative/vis.pt", rel.resolve()),
                                 (absolute, absolute.resolve()), (None, default.resolve())):
        InferencePipeline.from_config(
            cfg, project_root=project, checkpoint=checkpoint, predictor=object(),
        )
        assert seen[-1] == expected
    assert seen[0] != decoy.resolve()