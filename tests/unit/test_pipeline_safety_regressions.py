"""数据准备、训练恢复、评估与停止控制的安全回归测试。"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import io
import json
import subprocess
import sys
import zipfile
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]


def load(relative):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location("pipeline_safety_" + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def small_cpu_pool():
    threads, rng = torch.get_num_threads(), torch.get_rng_state()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(threads)
    torch.set_rng_state(rng)


@pytest.mark.parametrize("malformed", [[], {}, {"categories": [{}]}, {"images": [None]},
                                           {"annotations": [{"bbox": [1, 2]}]},
                                           {"images": [{"id": 1, "width": 10, "height": 10}]}])
def test_bad_coco_schema_is_reported_and_next_file_checked(tmp_path, malformed):
    from car_smart_assist.data import preprocessing as pp

    valid = {"images": [], "annotations": [], "categories": []}
    bad = {**valid, **malformed} if isinstance(malformed, dict) and malformed else malformed
    first = tmp_path / "a_gt_detection.json"
    second = tmp_path / "b_gt_detection.json"
    first.write_text(json.dumps(bad), encoding="utf-8")
    second.write_text(json.dumps(valid), encoding="utf-8")
    report = pp.CleaningReport()
    pp._check_detection_jsons(tmp_path, tmp_path, report, {})
    assert any(issue.check == "detection/schema" for issue in report.issues)
    assert any(key.endswith(second.name) for key in report.stats["detection_files"])
    assert json.loads(first.read_text()) == bad


def test_zip_outside_project_keeps_absolute_member_identity(tmp_path):
    from car_smart_assist.data import preprocessing as pp

    buffer = io.BytesIO()
    Image.new("RGB", (4, 4)).save(buffer, format="PNG")
    archive = tmp_path / "external.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("frame.png", buffer.getvalue())
    report = pp.CleaningReport()
    paths = pp.check_zip_image_integrity([archive], report, check="pab/image_integrity",
                                        project_root=tmp_path / "project")
    assert paths == [f"{archive.resolve()}::frame.png"]
    assert not report.issues


def test_detection_single_weather_has_four_disjoint_partitions(tmp_path):
    from car_smart_assist.data.detection import DetectionImage, stratified_group_split

    rows = [DetectionImage(str(i), tmp_path / str(i), 4, 4, "fog", str(i), str(i),
                           ((0, .5, .5, .2, .2),), ()) for i in range(4)]
    parts = stratified_group_split(rows, dict(train=.7, val=.1, calibration=.1, test=.1), 42, 1)
    assert sorted(len(v) for v in parts.values()) == [1, 1, 1, 1]
    assert {r.group for values in parts.values() for r in values} == {str(i) for i in range(4)}


@pytest.mark.parametrize("corrupt", [False, True])
def test_detection_read_failure_only_excludes_confirmed_corruption(tmp_path, monkeypatch, corrupt):
    from car_smart_assist.data import detection

    monkeypatch.setattr(detection, "probe_image_integrity", lambda _p: {
        "ok": False, "corrupt": corrupt, "error": "unreadable"})
    row = SimpleNamespace(source=tmp_path / "frame.png")
    if corrupt:
        assert detection._verify_record(row) == (row, "unreadable")
    else:
        with pytest.raises(OSError, match="cannot exclude as corrupt"):
            detection._verify_record(row)


def test_empty_detection_export_fails_before_writing(tmp_path, monkeypatch):
    from car_smart_assist.data import detection

    monkeypatch.setattr(detection, "read_detection_annotations", lambda *_: ([], {}, {}))
    official_index = tmp_path / "data/raw/acdc/gt_detection/instancesonly_test_image_info.json"
    official_index.parent.mkdir(parents=True)
    official_index.write_text(json.dumps({"images": [], "annotations": []}), encoding="utf-8")
    cfg = {
        "acdc_root": "data/raw/acdc",
        "official_test_index": "gt_detection/instancesonly_test_image_info.json",
        "output_root": "data/processed/new",
        "integrity_workers": 1,
    }
    with pytest.raises(ValueError, match="No eligible"):
        detection.prepare_detection_data(tmp_path, cfg)
    assert not (tmp_path / "data/processed/new").exists()


def test_partial_information_weights_rejected_and_complete_weights_accepted():
    from car_smart_assist.perception.visibility.scorer import _aggregation_parameters

    with pytest.raises(ValueError, match="四个"):
        _aggregation_parameters({"aggregation": {"weights": {"contrast": .4}}})
    weights, _, _ = _aggregation_parameters({})
    assert len(weights) == 4


@pytest.mark.parametrize("change", ["split", "file", "missing"])
def test_visibility_evaluation_rejects_wrong_reference_identity_before_scoring(tmp_path, monkeypatch, change):
    from car_smart_assist.perception.visibility.dataset import reference_split_signature, split_ref_indices

    module = load("scripts/evaluate_visibility.py")
    monkeypatch.setattr(module, "__file__", str(tmp_path / "scripts/evaluate_visibility.py"))
    paths = [tmp_path / str(i) for i in range(20)]
    for p in paths:
        p.write_bytes(p.name.encode())
    cfg = {"acdc_root": str(tmp_path), "split_by_sequence": False}
    parts = split_ref_indices(20, seed=42, groups=None)
    signature = reference_split_signature(paths, tmp_path, (16, 16), None, *parts)
    if change == "split":
        cfg["split_seed"] = 99
    elif change == "file":
        paths[0].write_bytes(b"changed reference")
    else:
        signature = None
    scorer = SimpleNamespace(input_size=(16, 16), split_signature=signature,
                             score_arrays=lambda *_: pytest.fail("scoring before identity check"))
    message = "检查点缺少 split_signature" if change == "missing" else "身份不一致"
    with pytest.raises(module.SplitIdentityError, match=message):
        module.eval_fit_quality(scorer, [], paths, cfg, None, 1)


def test_empty_severity_cli_rejected_before_checkpoint_access(monkeypatch):
    module = load("scripts/evaluate_visibility.py")
    monkeypatch.setattr(sys, "argv", ["evaluate_visibility.py", "--severities"])
    with pytest.raises(SystemExit) as error:
        module.main()
    assert error.value.code == 2


@pytest.mark.parametrize("report_file", ["published/REPORT.md", "separate/REPORT.md"])
def test_preflight_cannot_overwrite_default_reports(tmp_path, monkeypatch, report_file):
    from car_smart_assist.perception import weather_attention_training as training

    config = tmp_path / "configs/model"
    config.mkdir(parents=True)
    (config / "weather_default.yaml").write_text(yaml.safe_dump({"weather_default": {
        "config": "configs/model/deployed.yaml", "checkpoint": "weights/best.pt"}}), encoding="utf-8")
    deployed = yaml.safe_load(
        (ROOT / "configs/model/weather_attention_a.yaml").read_text(encoding="utf-8")
    )
    deployed["weather_attention"].update(report_dir="published", report_file="published/REPORT.md")
    (config / "deployed.yaml").write_text(yaml.safe_dump(deployed), encoding="utf-8")
    cfg = SimpleNamespace(report_dir="published" if report_file.startswith("separate") else "separate",
                          report_file=report_file)
    monkeypatch.setattr(training, "select_device", lambda *_: pytest.fail("started preflight"))
    with pytest.raises(ValueError, match="受保护"):
        training.preflight_attention(cfg, tmp_path, {}, {})
    assert not (tmp_path / "published").exists()


def test_orphan_attention_best_is_preserved_before_initialization(tmp_path, monkeypatch):
    from car_smart_assist.perception import weather_attention_training as training

    folder = tmp_path / "checkpoints"
    folder.mkdir()
    best = folder / "best.pt"
    best.write_bytes(b"orphan original")
    cfg = SimpleNamespace(checkpoint_root="checkpoints", report_dir="reports", report_file="reports/REPORT.md", train={}, evaluation={})
    monkeypatch.setattr(training, "ensure_pretrained_weights", lambda *_: pytest.fail("initialized before orphan check"))
    with pytest.raises(ValueError, match="缺少配对 last"):
        training.train_attention(cfg, tmp_path, {}, {})
    assert best.read_bytes() == b"orphan original"


def test_orphan_pyramid_best_is_preserved(tmp_path):
    from car_smart_assist.perception.weather_pyramid_training import _train_variant

    folder = tmp_path / "checkpoints/asl"
    folder.mkdir(parents=True)
    (folder / "best.pt").write_bytes(b"orphan")
    with pytest.raises(ValueError, match="缺少配对 last"):
        _train_variant(SimpleNamespace(checkpoint_root="checkpoints"), tmp_path, {}, {}, "asl", "cpu", False)
    assert (folder / "best.pt").read_bytes() == b"orphan"


def test_run_manifest_missing_and_stale_are_rejected(tmp_path):
    from car_smart_assist.perception.weather_extension import require_run_manifest

    with pytest.raises(ValueError, match="运行前"):
        require_run_manifest(tmp_path, tmp_path, "run_manifest.json")
    source = tmp_path / "source.py"
    source.write_text("original", encoding="utf-8")
    manifest = {"protected_hashes": {}, "snapshot_hashes": {}, "snapshot_root": "snapshot",
                "source_hashes": {"source.py": hashlib.sha256(source.read_bytes()).hexdigest()}}
    (tmp_path / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert require_run_manifest(tmp_path, tmp_path, "run_manifest.json") == manifest
    source.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="过期"):
        require_run_manifest(tmp_path, tmp_path, "run_manifest.json")


def yolo_state(tmp_path, monkeypatch):
    helper = load("tests/unit/test_yolo_training_state.py")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    trainer = helper.trainer(tmp_path)
    callbacks = {}
    model = SimpleNamespace(add_callback=lambda k, v: callbacks.update({k: v}))
    helper.attach_training_state_callbacks(model, "data", None, selection_contract=helper.SELECTION_CONTRACT,
                                          training_segment=helper.TRAINING_SEGMENT)
    callbacks["on_train_start"](trainer)
    callbacks["on_model_save"](trainer)
    state = torch.load(tmp_path / "last_training_state.pt", weights_only=True)
    return helper, trainer, model, callbacks, state


def test_yolo_empty_disabled_scaler_can_resume_with_enabled_scaler(tmp_path, monkeypatch):
    helper, trainer, model, callbacks, state = yolo_state(tmp_path, monkeypatch)
    assert state["scaler"] == {} and state["scaler_enabled"] is False
    trainer.scaler = torch.amp.GradScaler("cpu", enabled=True)
    helper.attach_training_state_callbacks(model, "data", state, selection_contract=helper.SELECTION_CONTRACT,
                                          training_segment=helper.TRAINING_SEGMENT)
    callbacks["on_train_start"](trainer)
    assert trainer.scaler.is_enabled()


@pytest.mark.parametrize("ids", [["GPU-other"], None])
def test_yolo_changed_or_unverifiable_cuda_identity_refused(tmp_path, monkeypatch, ids):
    helper, trainer, model, callbacks, state = yolo_state(tmp_path, monkeypatch)
    state.update(cuda_rng=[torch.get_rng_state()], cuda_device_ids=ids)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _: SimpleNamespace(uuid="GPU-current"))
    helper.attach_training_state_callbacks(model, "data", state, selection_contract=helper.SELECTION_CONTRACT,
                                          training_segment=helper.TRAINING_SEGMENT)
    with pytest.raises(ValueError, match="CUDA device layout"):
        callbacks["on_train_start"](trainer)


@pytest.mark.parametrize("bad_loss", [float("nan"), float("inf")])
def test_ae_nonfinite_validation_does_not_replace_last_good_checkpoint(tmp_path, monkeypatch, bad_loss):
    from car_smart_assist.perception.visibility.trainer import VisibilityTrainer

    cfg = {"device": "cpu", "model": {"input_size": [16, 16], "base_channels": 2, "latent_dim": 4},
           "data": {"num_workers": 0}, "train": {"epochs": 2, "batch_size": 2, "amp": False,
           "augment": False, "checkpoint_dir": "checkpoints", "early_stopping": {"enabled": False}}}
    trainer = VisibilityTrainer(cfg, project_root=tmp_path, resume="none")
    samples = torch.ones((2, 3, 16, 16)) * .4
    monkeypatch.setattr(trainer, "_build_datasets", lambda: (samples, samples, samples, []))
    saved = {}

    def validation(_loader):
        path = tmp_path / "checkpoints/last.pt"
        if not path.exists():
            return 1.0
        saved["last"] = path.read_bytes()
        return bad_loss

    monkeypatch.setattr(trainer, "_eval_loss", validation)
    with pytest.raises(ValueError, match="非有限"):
        trainer.fit()
    assert (tmp_path / "checkpoints/last.pt").read_bytes() == saved["last"]
    assert torch.load(tmp_path / "checkpoints/last.pt", weights_only=False)["epoch"] == 1


@contextmanager
def external_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    command = "from filelock import FileLock; import sys; lock=FileLock(sys.argv[1]); lock.acquire(); print('ready',flush=True); sys.stdin.read(1); lock.release()"
    process = subprocess.Popen([sys.executable, "-c", command, str(path)], stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        assert process.stdout.readline().strip() == "ready"
        yield
    finally:
        process.communicate("x", timeout=30)
        assert process.returncode == 0


@pytest.mark.parametrize("kind", ["visibility", "detection", "weather_attention"])
def test_training_cli_lock_conflict_preserves_stop_and_artifacts(tmp_path, monkeypatch, kind):
    module = load(f"scripts/train_{kind}.py")
    monkeypatch.setattr(module, "__file__", str(tmp_path / "scripts" / f"train_{kind}.py"))
    if kind == "visibility":
        cfg = {"visibility": {"train": {"checkpoint_dir": "run"}}}
        (tmp_path / "cfg.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
        directory, marker = tmp_path / "run", tmp_path / "run/STOP"
    elif kind == "detection":
        monkeypatch.setattr(module, "load_yolo_config", lambda _: {"name": "run", "project": "runs"})
        directory, marker = tmp_path / "runs/run", tmp_path / "runs/run/weights/STOP"
    else:
        from car_smart_assist.config.weather_attention import load_weather_attention_config
        cfg = load_weather_attention_config(ROOT / "configs/model/weather_attention.yaml")
        cfg = replace(cfg, checkpoint_root="run", report_dir="reports", report_file="reports/REPORT.md")
        monkeypatch.setattr(module, "load_weather_attention_config", lambda _: cfg)
        monkeypatch.setattr(module, "model_report", lambda *_: pytest.fail("lock loser wrote report"))
        directory, marker = tmp_path / "run", tmp_path / "run/STOP"
    marker.parent.mkdir(parents=True)
    marker.write_text("requested", encoding="utf-8")
    sentinel = directory / "best.pt"
    sentinel.write_bytes(b"original best")
    monkeypatch.setattr(sys, "argv", ["train", "--config", "cfg.yaml", "--fresh"])
    with external_lock(directory / "run.lock"):
        assert module.main() == 2
    assert marker.read_text() == "requested"
    assert sentinel.read_bytes() == b"original best"
    assert not (directory / "history").exists()


@pytest.mark.parametrize("failure", [KeyboardInterrupt, OSError])
def test_runner_wait_failure_reaps_child_before_propagating(monkeypatch, failure):
    runner = load("AI_Skills/Test_Cases/_runner.py")
    calls = []
    outcomes = iter([failure(), 1])

    def wait(**_kwargs):
        calls.append("wait")
        outcome = next(outcomes)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    process = SimpleNamespace(wait=wait, terminate=lambda: calls.append("terminate"),
                              send_signal=lambda _: calls.append("signal"))
    with pytest.raises(failure):
        runner.wait_for_process(process)
    assert calls[-1] == "wait" and len(calls) == 3


def test_attention_stop_updates_markdown_and_progress(tmp_path, monkeypatch):
    helper = load("tests/unit/test_training_stop_models.py")
    from car_smart_assist.utils.stop import StopController

    controller = StopController(default_directory=tmp_path)
    training, cfg, datasets, split, _ = helper._attention_setup(
        tmp_path, monkeypatch, stop_controller=controller, pause_on_first=True)
    reports = []
    monkeypatch.setattr(training, "model_report", lambda _cfg, _root, progress: reports.append(copy.deepcopy(progress)))
    _, result = training.train_attention(cfg, tmp_path, datasets, split, fresh=True, stop_controller=controller)
    assert result["manual_stop"] and reports[-1]["status"] == "手动停止"
    assert json.loads((tmp_path / "reports/progress.json").read_text(encoding="utf-8"))["manual_stop"]


def test_ablation_reproducibility_rejects_skipped_updates(tmp_path, monkeypatch):
    from car_smart_assist.perception import weather_attention_ablation as ablation

    monkeypatch.setattr(ablation, "attention_sampling_weights", lambda *_: torch.ones(2))
    generator = torch.Generator().manual_seed(42)
    order = torch.multinomial(torch.ones(2), 2, replacement=True, generator=generator)
    payload = {"epoch": 1, "initial_model_sha256": "same", "history": [{
        "epoch": 1, "sampling_sha256": hashlib.sha256(order.numpy().tobytes()).hexdigest(), "skipped_updates": 1}]}
    path = tmp_path / "weights/a/last.pt"
    path.parent.mkdir(parents=True)
    torch.save(payload, path)
    with pytest.raises(ValueError, match="跳过"):
        ablation.verify_reproducibility(SimpleNamespace(data={"seed": 42}), {
            "training_epochs": 1, "experiments": ["a"], "checkpoint_root": "weights", "report_dir": "reports"},
            tmp_path, {"train": [0, 1]}, {"reconstructed_initial_sha256": "same"})
    assert not (tmp_path / "reports/reproducibility.json").exists()


def test_ablation_old_signature_does_not_compare_different_thresholds(tmp_path):
    module = load("scripts/run_visibility_ablations.py")
    row = {"params_m": 1., "overfit_ratio": 1., "val_fpr": .1, "mean_recall": .5, "train_seconds": 1.,
           "synth": {k: {"recall": .5, "curve": []} for k in module.KINDS},
           "comparison_signature": "legacy-same", "thresholds": {"info_blind": .1}, "scoring": {}}
    changed = {**row, "thresholds": {"info_blind": .9}, "mean_recall": .9}
    module.write_report({"baseline": row, "changed": changed}, tmp_path)
    report = (tmp_path / "ablations.md").read_text(encoding="utf-8")
    assert "不作直接比较" in report and "+40.0pp" not in report


def test_force_only_preserves_unselected_ablation_results(tmp_path, monkeypatch):
    module = load("scripts/run_visibility_ablations.py")
    monkeypatch.setattr(module, "__file__", str(tmp_path / "scripts/run_visibility_ablations.py"))
    (tmp_path / "cfg.yaml").write_text(yaml.safe_dump({"visibility": {
        "model": {"input_size": [16, 16]}, "thresholds": {"use_recon_z": True}}}), encoding="utf-8")
    out = tmp_path / "artifacts/reports/visibility/ablations"
    out.mkdir(parents=True)
    preserved = {"other": {"historical": "keep exactly"}, "baseline": {"stale": True}}
    (out / "results.json").write_text(json.dumps(preserved), encoding="utf-8")
    monkeypatch.setattr(module, "_prepare_inputs", lambda *_: {
        "paths": [], "size": (16, 16), "splits": [np.array([0])] * 4,
        "data_identity": {}, "protocol": {}, "refs": [np.zeros((16, 16, 3), np.uint8)]})

    class Trainer:
        def __init__(self, cfg, **_kwargs):
            self.directory = tmp_path / cfg["train"]["checkpoint_dir"]

        def fit(self, **_kwargs):
            for name in ("last.pt", "best.pt"):
                (self.directory / name).write_bytes(b"test fixture")
            return SimpleNamespace(manual_stop=False, train_loss=[1.], best_epoch=1)

    monkeypatch.setattr(module, "VisibilityTrainer", Trainer)
    monkeypatch.setattr(module.VisibilityScorer, "from_checkpoint", lambda *_a, **_k: SimpleNamespace(
        input_size=(16, 16), scoring_cfg={}))
    monkeypatch.setattr(module, "evaluate_run", lambda *_: {
        "params_m": 1., "overfit_ratio": 1., "val_fpr": .1, "mean_recall": .5, "synth": {}})
    monkeypatch.setattr(module, "write_report", lambda *_: 0)
    monkeypatch.setattr(sys, "argv", ["ablation", "--config", "cfg.yaml", "--force", "--only", "baseline"])
    assert module.main() == 0
    result = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert result["other"] == preserved["other"] and "experiment_signature" in result["baseline"]


def test_attention_comparison_stop_records_terminal_status_under_lock(tmp_path, monkeypatch):
    module = load("scripts/train_weather_attention.py")
    from car_smart_assist.config.weather_attention import load_weather_attention_config
    from car_smart_assist.config.weather_attention import attention_model_config

    cfg = replace(load_weather_attention_config(ROOT / "configs/model/weather_attention.yaml"),
                  checkpoint_root="weights", report_dir="reports", report_file="reports/REPORT.md")
    monkeypatch.setattr(module, "__file__", str(tmp_path / "scripts/train_weather_attention.py"))
    monkeypatch.setattr(module, "load_weather_attention_config", lambda *_: cfg)
    monkeypatch.setattr(module, "prepare_attention_datasets", lambda *_: ({}, {"signature": "same"}))
    folder = tmp_path / "reports"
    folder.mkdir()
    profile = {"model_config": attention_model_config(cfg), "split_signature": "same",
               "backbone_training": {"fine_tune_from": cfg.backbone["fine_tune_from"],
                                     "freeze_epochs": cfg.backbone["freeze_epochs"],
                                     "probe_epoch": cfg.backbone["freeze_epochs"] + 1},
               "batch_size": cfg.train["batch_size"], "channels_last": cfg.train["channels_last"],
               "swad": cfg.train.get("swad"), "l2_sp_coefficient": cfg.train.get("l2_sp", 0)}
    (folder / "preflight.json").write_text(json.dumps(profile), encoding="utf-8")
    monkeypatch.setattr(module, "train_attention", lambda *_a, **_k: (None, {"best_epoch": 1}))

    def stop(*_args, **_kwargs):
        raise SystemExit(75)

    monkeypatch.setattr(module, "compare_attention", stop)
    reports = []
    monkeypatch.setattr(module, "model_report", lambda _c, _r, progress: reports.append(progress))
    assert module.main(["--config", "cfg.yaml"]) == 75
    assert reports[-1]["status"] == "手动停止"
    assert json.loads((folder / "progress.json").read_text(encoding="utf-8"))["termination_reason"] == "manual_stop"


def test_weather_report_does_not_claim_excluded_reference_images(tmp_path):
    from car_smart_assist.config.weather_attention import load_weather_attention_config
    from car_smart_assist.perception.weather_attention_training import model_report

    cfg = load_weather_attention_config(ROOT / "configs/model/weather_attention.yaml")
    cfg = replace(cfg, data={**cfg.data, "include_normal_references": False},
                  report_dir="reports", report_file="REPORT.md")
    model_report(cfg, tmp_path, {"status": "fixture"})
    report = (tmp_path / "REPORT.md").read_text(encoding="utf-8")
    assert "未加入 ACDC 正常参考图" in report
    assert "ACDC 正常参考及原图绑定" not in report


def test_plain_attention_result_preserves_non_swad_consumer_contract(tmp_path, monkeypatch):
    helper = load("tests/unit/test_training_stop_models.py")
    training, cfg, datasets, split, _ = helper._attention_setup(tmp_path, monkeypatch)
    _, trained = training.train_attention(cfg, tmp_path, datasets, split, fresh=True)
    assert "selected_kind" not in trained and "swad" not in trained
    assert trained["best_epoch"] == trained["selected_epoch"] == 1


def test_swad_export_epoch_kept_separate_from_single_point_best(tmp_path, monkeypatch):
    helper = load("tests/unit/test_training_stop_models.py")
    training, cfg, datasets, split, _ = helper._attention_setup(tmp_path, monkeypatch)
    cfg = replace(cfg, train={**cfg.train, "swad": {"enabled": True}, "early_stopping": {"enabled": False}})

    class Tracker:
        def __init__(self, *_args):
            self.valley = SimpleNamespace(summary=lambda: {})

        def update(self, *_args):
            pass

        def end_epoch(self, *_args):
            pass

        def state_dict(self):
            return {}

    def finish(_tracker, model, payload, cfg, root, *_args, **_kwargs):
        selected = {**payload, "epoch": 3, "decision_thresholds": {}, "calibration": {}}
        torch.save(selected, root / cfg.checkpoint_root / "best.pt")
        return model, selected, {"selected_kind": "swad", "swad": {"end_epoch": 3}, "candidates": {}}

    monkeypatch.setattr(training, "SWADTracker", Tracker)
    monkeypatch.setattr(training, "source_balanced_bce", lambda *_: .5)
    monkeypatch.setattr(training, "finish_swad_checkpoints", finish)
    _, trained = training.train_attention(cfg, tmp_path, datasets, split, fresh=True)
    assert trained["best_epoch"] == 1
    assert trained["selected_epoch"] == 3 and trained["selected_kind"] == "swad"


def test_pyramid_cannot_overwrite_deployed_default(tmp_path):
    from car_smart_assist.perception.weather_pyramid_training import _train_variant

    folder = tmp_path / "configs/model"
    folder.mkdir(parents=True)
    (folder / "weather_default.yaml").write_text(yaml.safe_dump({"weather_default": {
        "config": "cfg.yaml", "checkpoint": "weights/asl/best.pt"}}), encoding="utf-8")
    with pytest.raises(ValueError, match="受保护"):
        _train_variant(SimpleNamespace(checkpoint_root="weights"), tmp_path, {}, {}, "asl", "cpu", True)
    assert not (tmp_path / "weights").exists()


def test_runner_unconfirmed_cleanup_aborts_instead_of_continuing():
    runner = load("AI_Skills/Test_Cases/_runner.py")
    calls = []

    def failed_wait(**_kwargs):
        raise OSError("wait failed")

    process = SimpleNamespace(wait=failed_wait, terminate=lambda: calls.append("terminate"),
                              send_signal=lambda _: calls.append("signal"), kill=lambda: calls.append("kill"))
    with pytest.raises(RuntimeError, match="Cannot confirm subprocess exit"):
        runner.wait_for_process(process)
    assert "kill" in calls
