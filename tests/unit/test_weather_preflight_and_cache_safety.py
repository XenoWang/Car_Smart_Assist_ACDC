"""Shared weather cache publication, fresh preflight and threshold boundaries."""

import csv
from dataclasses import dataclass, replace
import hashlib
import json
import multiprocessing
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from _model_test_helpers import ROOT, load_module, synthetic_attention_preflight


def _cache_process(source, cache, role, entered, attempting, release, results):
    from car_smart_assist.data import weather_pyramid as data
    from car_smart_assist.perception.weather_training import WeatherSample

    real_lock, prepare = data.FileLock, data.prepare_pyramid_image
    calls = []

    class Lock:
        def __init__(self, *args, **kwargs):
            self.lock = real_lock(*args, **kwargs)
        def __enter__(self):
            if role == "second":
                attempting.set()
            return self.lock.__enter__()
        def __exit__(self, *args):
            return self.lock.__exit__(*args)

    def prepare_image(image, size):
        calls.append(1)
        if role == "first":
            entered.set()
            if not release.wait(45):
                raise TimeoutError("cache construction release did not arrive")
        return prepare(image, size)

    data.FileLock, data.prepare_pyramid_image = Lock, prepare_image
    try:
        samples = [WeatherSample((1, 0, 0, 0), "group", "fog", path=Path(source))]
        ds = data.PyramidDataset(samples, SimpleNamespace(image_size=(8, 8)), Path(cache))
        pixels = np.asarray(ds.images).copy()
        results.put({"role": role, "calls": len(calls), "shape": pixels.shape,
                     "signature": ds.signature, "sha256": hashlib.sha256(pixels.tobytes()).hexdigest()})
        ds.images._mmap.close()
    except BaseException as exc:
        results.put({"role": role, "error": f"{type(exc).__name__}: {exc}"})
        raise


def test_two_processes_share_one_weather_cache_without_duplicate_builds(tmp_path):
    source, cache = tmp_path / "source.png", tmp_path / "cache"
    Image.new("RGB", (8, 8), (29, 29, 29)).save(source)
    before = source.read_bytes()
    context = multiprocessing.get_context("spawn")
    entered, attempting, release = context.Event(), context.Event(), context.Event()
    queue = context.Queue()
    processes = [context.Process(target=_cache_process,
                 args=(str(source), str(cache), role, entered, attempting, release, queue))
                 for role in ("first", "second")]
    try:
        processes[0].start()
        assert entered.wait(45), "first writer did not reach the locked decode"
        processes[1].start()
        assert attempting.wait(45), "second process did not enter the cache lock"
        release.set()
        for process in processes:
            process.join(60)
            assert not process.is_alive() and process.exitcode == 0
        rows = {row["role"]: row for row in (queue.get(timeout=5), queue.get(timeout=5))}
        assert rows["first"]["calls"] == 1 and rows["second"]["calls"] == 0
        expected = hashlib.sha256(np.full((1, 8, 8, 3), 29, np.uint8).tobytes()).hexdigest()
        assert rows["first"]["sha256"] == expected == rows["second"]["sha256"]
        assert rows["first"]["shape"] == (1, 8, 8, 3)
        assert rows["first"]["signature"] == rows["second"]["signature"]
        assert source.read_bytes() == before
        assert len(list(cache.glob("images_*.npy"))) == 1
        assert not list(cache.glob("*.partial.npy"))
    finally:
        release.set()
        for process in processes:
            if process.pid is not None:
                if process.is_alive():
                    process.terminate()
                process.join(10)
        queue.close()


def _config():
    from car_smart_assist.config.weather_attention import load_weather_attention_config

    cfg = load_weather_attention_config(ROOT / "configs/model/weather_attention.yaml")
    evaluation = {**cfg.evaluation, "attention_references": {},
                  "old_checkpoint": "old.pt", "asl_checkpoint": "asl.pt"}
    evaluation.pop("summary_file", None)
    return replace(cfg, device="cpu", checkpoint_root="weights", report_dir="reports",
                   report_file="reports/REPORT.md", evaluation=evaluation)


def test_cache_publication_denial_preserves_existing_array_and_explains_recovery(tmp_path, monkeypatch):
    from car_smart_assist.data.weather_pyramid import PyramidDataset
    from car_smart_assist.perception.weather_training import WeatherSample

    source, cache = tmp_path / "source.png", tmp_path / "cache"
    Image.new("RGB", (8, 8), (29, 29, 29)).save(source)
    samples = [WeatherSample((1, 0, 0, 0), "group", "fog", path=source)]
    cfg = SimpleNamespace(image_size=(8, 8))
    reader = PyramidDataset(samples, cfg, cache)
    output = next(cache.glob("images_*.npy"))
    manifest = output.with_suffix(".json")
    manifest.write_text('{"signature":"invalid old metadata"}', encoding="utf-8")
    before_array, before_manifest = output.read_bytes(), manifest.read_bytes()
    replace_path = Path.replace

    def deny_array(path, target):
        if Path(target) == output:
            raise PermissionError("controlled mapped-cache publication denial")
        return replace_path(path, target)

    monkeypatch.setattr(Path, "replace", deny_array)
    try:
        with pytest.raises(ValueError, match="data.cache_dir") as raised:
            PyramidDataset(samples, cfg, cache)
        assert isinstance(raised.value.__cause__, PermissionError)
        assert output.read_bytes() == before_array and manifest.read_bytes() == before_manifest
        np.testing.assert_array_equal(np.asarray(reader.images), np.full((1, 8, 8, 3), 29, np.uint8))
        assert not list(cache.glob("*.partial.npy"))
    finally:
        reader.images._mmap.close()


@pytest.mark.parametrize("accepted", [False, True])
def test_fresh_archives_preflight_and_requires_new_profile_before_training(
    tmp_path, monkeypatch, accepted,
):
    from car_smart_assist.perception import weather_attention_training as training

    cfg = _config()
    old = {"preflight.json": b'{"old": true}', "preflight_old.npy": b"old cached preview",
           "preflight_old_metadata.json": b'{"old": 7}', "comparison.json": b"old comparison"}
    report = tmp_path / "reports"
    report.mkdir()
    for name, content in old.items():
        (report / name).write_bytes(content)
    weights = tmp_path / "initializer.pt"
    weights.write_bytes(b"synthetic initializer")
    for name in ("old.pt", "asl.pt"):
        (tmp_path / name).write_bytes(b"preserved baseline")
    monkeypatch.setattr(training, "ensure_pretrained_weights", lambda *_a: weights)
    calls = []

    def preflight(config, root, datasets, info):
        calls.append("preflight")
        assert not (report / "preflight.json").exists()
        assert not (report / "preflight_old.npy").exists()
        torch.rand(9)  # The caller's RNG must survive this new fresh-path work.
        profile = synthetic_attention_preflight(config, root, datasets, info)
        profile["origin"] = "current run"
        if not accepted:
            profile["batch_size"] += 1
        training.write_json(report / "preflight.json", profile)
        return profile

    class TrainingBoundary(Exception):
        pass

    def model(*_args, **_kwargs):
        calls.append("model")
        raise TrainingBoundary

    monkeypatch.setattr(training, "preflight_attention", preflight)
    monkeypatch.setattr(training, "WeatherAttention", model)
    before = torch.get_rng_state().clone()
    with pytest.raises(TrainingBoundary if accepted else ValueError):
        training.train_attention(cfg, tmp_path, {}, {"signature": "controlled-split"}, fresh=True)
    assert calls == (["preflight", "model"] if accepted else ["preflight"])
    assert torch.equal(torch.get_rng_state(), before)
    archives = list((tmp_path / "weights/history").iterdir())
    assert len(archives) == 1
    for name, content in old.items():
        assert (archives[0] / "reports" / name).read_bytes() == content
    assert json.loads((report / "preflight.json").read_text(encoding="utf-8"))["origin"] == "current run"
    assert not (tmp_path / "weights/last.pt").exists()


def test_fresh_cli_does_not_require_an_old_preflight(tmp_path, monkeypatch):
    helper = load_module("tests/unit/test_weather_config_and_report_contracts.py", "fresh_cli_helpers")
    cli, _cfg, _calls = helper._preflight_cli(tmp_path, monkeypatch, {})
    seen = []
    monkeypatch.setattr(cli, "train_attention", lambda *_a, fresh, **_k:
                        (seen.append(fresh), (object(), {"best_epoch": 1}))[1])
    assert cli.main(["--config", "unused.yaml", "--fresh"]) == 0
    assert seen == [True]


@pytest.mark.parametrize("payload", [{}, {"epoch": 1}, {"epoch": 0, "selection": {}},
                                      {"epoch": 1, "selection": []}])
def test_invalid_reference_metadata_is_rejected_before_initializer_or_archival(tmp_path, monkeypatch, payload):
    from car_smart_assist.perception import weather_attention_training as training

    cfg = _config()
    cfg = replace(cfg, evaluation={**cfg.evaluation,
                  "attention_references": {"candidate": {"checkpoint": "reference.pt"}}})
    path = tmp_path / "reference.pt"
    torch.save(payload, path)
    (tmp_path / "weights").mkdir()
    previous = tmp_path / "weights/best.pt"
    previous.write_bytes(b"previous run checkpoint")
    before = path.read_bytes()
    monkeypatch.setattr(training, "ensure_pretrained_weights", lambda *_a: pytest.fail("initializer reached"))
    with pytest.raises(ValueError, match="epoch/selection"):
        training.train_attention(cfg, tmp_path, {}, {"signature": "split"}, fresh=True)
    assert previous.read_bytes() == b"previous run checkpoint"
    assert path.read_bytes() == before and not (tmp_path / "weights/history").exists()


def test_deployment_and_metrics_use_the_same_exact_half_probability_boundary():
    from car_smart_assist.perception.weather_attention import WeatherAttentionPredictor
    from car_smart_assist.perception.weather_pyramid_training import masked_metrics

    cfg = replace(_config(), image_size=(8, 8))
    thresholds = {name: 0.50000001 for name in cfg.attributes}

    class ZeroLogits(torch.nn.Module):
        def forward(self, values):
            return torch.zeros((len(values), 4), device=values.device)

    prediction = WeatherAttentionPredictor(ZeroLogits(), cfg, thresholds, torch.device("cpu")).predict(
        np.zeros((8, 8, 3), np.uint8))
    assert prediction.probabilities["fog"] == 0.5
    assert prediction.decisions["fog"] is False and prediction.attributes == ()
    labels = np.ones((1, 4), np.float32)
    result = masked_metrics(labels, np.ones_like(labels), np.full_like(labels, 0.5), cfg.attributes, thresholds)
    assert result["per_attribute"]["fog"]["tp"] == 0
    assert result["per_attribute"]["fog"]["fn"] == 1


def test_reference_metadata_accepts_a_valid_epoch_and_selection_without_rewriting(tmp_path):
    from car_smart_assist.perception.weather_attention_training import load_attention_reference_metadata

    path = tmp_path / "reference.pt"
    torch.save({"epoch": 3, "selection": {"score": 0.75}}, path)
    before = path.read_bytes()
    assert load_attention_reference_metadata(path) == {"epoch": 3, "selection": {"score": 0.75}}
    assert path.read_bytes() == before


def test_comparison_csv_uses_the_deployed_double_precision_boundary(tmp_path, monkeypatch):
    from car_smart_assist.perception import weather_attention_training as training
    from car_smart_assist.perception.weather_training import WeatherSample

    cfg = _config()
    thresholds = {name: 0.50000001 for name in cfg.attributes}
    sample = WeatherSample((1, 0, 0, 0), "group", "fog", path=tmp_path / "sample.png")
    dataset = SimpleNamespace(samples=[sample], targets=np.array([[1, 0, 0, 0]], np.float32),
                              mask=np.ones((1, 4), np.float32))
    values = np.full((1, 4), 0.5, np.float32)
    monkeypatch.setattr(training, "attention_probabilities", lambda *_a: values.copy())
    monkeypatch.setattr(training, "_old_probabilities", lambda *_a: values.copy())
    monkeypatch.setattr(training, "load_weather_pyramid_config", lambda *_a: SimpleNamespace(image_size=cfg.image_size))
    monkeypatch.setattr(training.WeatherPyramidPredictor, "from_checkpoint", lambda *_a:
                        SimpleNamespace(model=object(), thresholds=thresholds))

    @dataclass
    class OldConfig:
        device: str = "cpu"
        @property
        def decision_thresholds(self):
            return thresholds

    monkeypatch.setattr(training, "load_weather_config", lambda *_a: OldConfig())
    monkeypatch.setattr(training.WeatherPredictor, "from_checkpoint", lambda *_a: object())
    monkeypatch.setattr(training, "WeatherDataset", lambda *_a: dataset)
    monkeypatch.setattr(training, "balanced_thresholds", lambda *_a: (thresholds, {}))
    monkeypatch.setattr(training, "detailed_metrics", lambda *_a: {"controlled": True})
    monkeypatch.setattr(training, "model_report", lambda *_a: None)
    trained = {"thresholds": thresholds, "best_epoch": 1, "selected_epoch": 1,
               "trained_through_epoch": cfg.train["epochs"], "termination_reason": "budget_completed",
               "baseline_hashes": {}}
    split = {"signature": "controlled", "manifest": "synthetic", "splits": {"test": {"count": 1}}}
    training.compare_attention(cfg, tmp_path, {"test": dataset, "calibration": dataset}, split,
                               object(), trained)
    with (tmp_path / "reports/test_predictions.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    selected = [row for row in rows if row["model"] == "attention_asl"]
    assert len(selected) == 4
    assert all(float(row["probability"]) == 0.5 and row["prediction"] == "0" for row in selected)


def test_weather_cli_missing_configuration_is_an_input_error(tmp_path, monkeypatch, capsys):
    cli = load_module("scripts/train_weather_attention.py", "missing_weather_config_cli")
    missing = tmp_path / "missing.yaml"
    assert cli.main(["--config", str(missing)]) == 2
    captured = capsys.readouterr()
    assert str(missing) in captured.err
    assert "Traceback" not in captured.out + captured.err


def test_expected_weather_failure_still_records_interruption_and_preserves_checkpoint(
    tmp_path, monkeypatch, capsys, caplog,
):
    helper = load_module("tests/unit/test_training_cli_and_resume_contracts.py", "weather_failure_cli_helpers")
    cli, _controller, root = helper._weather_main_fixture(
        tmp_path, monkeypatch, compare=lambda *_a: (lambda *_a, **_k: {}))
    checkpoint = root / "checkpoints/best.pt"
    checkpoint.write_bytes(b"previous checkpoint")

    def fail(*_args, **_kwargs):
        raise ValueError("controlled training input failure")

    monkeypatch.setattr(cli, "train_attention", fail)
    assert cli.main(["--config", "unused.yaml"]) == 2
    state = json.loads((root / "reports/progress.json").read_text(encoding="utf-8"))
    assert state["status"] == "中断" and "controlled training input failure" in state["error"]
    assert checkpoint.read_bytes() == b"previous checkpoint"
    captured = capsys.readouterr()
    assert "Traceback" not in captured.out + captured.err + caplog.text
