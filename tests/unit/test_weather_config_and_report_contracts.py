"""Weather routing, configuration, fresh-run report archival and preflight contracts."""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]


def _write_yaml(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def test_weather_report_guard_allows_missing_route_in_independent_synthetic_project(tmp_path):
    from car_smart_assist.perception.weather_attention_training import require_independent_weather_reports

    cfg = SimpleNamespace(report_dir="candidate", report_file="candidate/REPORT.md")
    require_independent_weather_reports(cfg, tmp_path)
    assert not (tmp_path / "candidate").exists()


@pytest.mark.parametrize("route", [[], {}, {"weather_default": None},
                                    {"weather_default": {"checkpoint": "weights/best.pt"}},
                                    {"weather_default": {"config": "missing.yaml",
                                                          "checkpoint": "weights/best.pt"}}])
def test_weather_report_guard_rejects_malformed_or_missing_route_config(tmp_path, route):
    from car_smart_assist.perception.weather_attention_training import require_independent_weather_reports

    config_dir = tmp_path / "configs/model"
    _write_yaml(config_dir / "weather_default.yaml", route)
    cfg = SimpleNamespace(report_dir="candidate", report_file="candidate/REPORT.md")
    with pytest.raises(ValueError, match="默认天气路由或配置无效"):
        require_independent_weather_reports(cfg, tmp_path)
    assert not (tmp_path / "candidate").exists()


def test_weather_report_guard_uses_named_route_when_top_level_order_changes(tmp_path):
    from car_smart_assist.perception.weather_attention_training import require_independent_weather_reports

    config_dir = tmp_path / "configs/model"
    deployed_config = yaml.safe_load(
        (ROOT / "configs/model/weather_attention_a.yaml").read_text(encoding="utf-8")
    )
    deployed_config["weather_attention"]["report_dir"] = "published"
    deployed_config["weather_attention"]["report_file"] = "published/REPORT.md"
    _write_yaml(config_dir / "deployed.yaml", {"unrelated": {"marker": "fixture"}, **deployed_config})
    _write_yaml(config_dir / "weather_default.yaml", {
        "weather_default": {"config": "configs/model/deployed.yaml",
                            "checkpoint": "weights/default.pt"},
    })
    candidate = SimpleNamespace(report_dir="candidate", report_file="candidate/REPORT.md")
    require_independent_weather_reports(candidate, tmp_path)
    assert not (tmp_path / "candidate").exists()

    protected = SimpleNamespace(report_dir="candidate", report_file="published/REPORT.md")
    with pytest.raises(ValueError, match="受保护"):
        require_independent_weather_reports(protected, tmp_path)
    assert not (tmp_path / "published").exists()


@pytest.mark.parametrize("section,key", [("backbone", "fine_tune_frmo"), ("head", "dimention"),
                                           ("data", "sead"), ("train", "early_stoping"),
                                           ("evaluation", "cpu_threadz")])
def test_weather_config_rejects_unknown_keys_in_all_five_sections(tmp_path, section, key):
    from car_smart_assist.config.weather_attention import load_weather_attention_config

    raw = yaml.safe_load((ROOT / "configs/model/weather_attention.yaml").read_text(encoding="utf-8"))
    raw["weather_attention"][section][key] = True
    path = tmp_path / "config.yaml"
    _write_yaml(path, raw)
    with pytest.raises(ValueError, match="未知配置键"):
        load_weather_attention_config(path)


def test_weather_metadata_does_not_change_architecture_identity(tmp_path):
    from car_smart_assist.config.weather_attention import attention_model_config, load_weather_attention_config

    raw = yaml.safe_load((ROOT / "configs/model/weather_attention.yaml").read_text(encoding="utf-8"))
    base_path = tmp_path / "base.yaml"
    _write_yaml(base_path, raw)
    base = attention_model_config(load_weather_attention_config(base_path))
    raw["weather_attention"]["head"]["metadata"] = {"review_tag": "fixture"}
    raw["weather_attention"]["backbone"]["metadata"] = {"source": "fixture"}
    decorated_path = tmp_path / "decorated.yaml"
    _write_yaml(decorated_path, raw)
    decorated = attention_model_config(load_weather_attention_config(decorated_path))
    assert decorated == base


@pytest.mark.parametrize("device", ["", "gpu", 0])
def test_weather_config_rejects_invalid_device_before_device_selection(tmp_path, device):
    from car_smart_assist.config.weather_attention import load_weather_attention_config

    raw = yaml.safe_load((ROOT / "configs/model/weather_attention.yaml").read_text(encoding="utf-8"))
    raw["weather_attention"]["device"] = device
    path = tmp_path / "config.yaml"
    _write_yaml(path, raw)
    with pytest.raises(ValueError, match="device 必须"):
        load_weather_attention_config(path)


@pytest.mark.parametrize("device", ["auto", "cpu", "cuda:0"])
def test_weather_config_accepts_documented_device_forms(tmp_path, device):
    from car_smart_assist.config.weather_attention import load_weather_attention_config

    raw = yaml.safe_load((ROOT / "configs/model/weather_attention.yaml").read_text(encoding="utf-8"))
    raw["weather_attention"]["device"] = device
    path = tmp_path / "config.yaml"
    _write_yaml(path, raw)
    assert load_weather_attention_config(path).device == device


def test_fresh_archives_old_reports_byte_for_byte_before_training_continues(tmp_path, monkeypatch):
    from car_smart_assist.config.weather_attention import load_weather_attention_config
    from car_smart_assist.perception import weather_attention_training as training

    cfg = replace(load_weather_attention_config(ROOT / "configs/model/weather_attention.yaml"),
                  checkpoint_root="weights", report_dir="reports", report_file="reports/REPORT.md")
    old = {"comparison.json": b'{"status":"completed","metric":0.73}\n',
           "test_predictions.csv": b"sample,probability\nframe-a,0.91\n",
           "history.json": b'[{"epoch":1}]\n', "progress.json": b'{"status":"completed"}\n',
           "REPORT.md": b"completed old report\n"}
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    for name, payload in old.items():
        (report_dir / name).write_bytes(payload)
    weights = tmp_path / "synthetic-weights.pt"
    weights.write_bytes(b"synthetic weights")
    for key in ("old_checkpoint", "asl_checkpoint"):
        path = tmp_path / cfg.evaluation[key]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(key.encode())

    class StopController:
        def watch_directory(self, _path):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *_args):
            return False
        def require_running(self):
            pass

    class StopAfterArchive(Exception):
        pass

    monkeypatch.setattr(training, "require_independent_weather_output", lambda *_: None)
    monkeypatch.setattr(training, "require_independent_weather_reports", lambda *_: None)
    monkeypatch.setattr(training, "select_device", lambda *_: SimpleNamespace(type="cpu", index=None))
    monkeypatch.setattr(training, "ensure_pretrained_weights", lambda *_: weights)
    monkeypatch.setattr(training, "preflight_attention", lambda *_a, **_k: (_ for _ in ()).throw(StopAfterArchive()))
    with pytest.raises(StopAfterArchive):
        training.train_attention(cfg, tmp_path, {}, {"signature": "synthetic"}, fresh=True,
                                 stop_controller=StopController())
    archives = list((tmp_path / "weights/history").iterdir())
    assert len(archives) == 1
    archive = archives[0] / "reports"
    for name, payload in old.items():
        assert (archive / name).read_bytes() == payload
        assert not (report_dir / name).exists()


def _preflight_cli(tmp_path, monkeypatch, profile):
    from car_smart_assist.config.weather_attention import load_weather_attention_config

    module = _load_script("scripts/train_weather_attention.py")
    root = tmp_path / "project"
    scripts = root / "scripts"
    scripts.mkdir(parents=True)
    monkeypatch.setattr(module, "__file__", str(scripts / "train_weather_attention.py"))
    cfg = replace(load_weather_attention_config(ROOT / "configs/model/weather_attention.yaml"),
                  checkpoint_root="weights", report_dir="reports", report_file="reports/REPORT.md")
    monkeypatch.setattr(module, "load_weather_attention_config", lambda *_: cfg)
    monkeypatch.setattr(module, "require_independent_weather_reports", lambda *_: None)
    monkeypatch.setattr(module, "require_independent_weather_output", lambda *_: None)
    monkeypatch.setattr(module, "prepare_attention_datasets", lambda *_: ({}, {"signature": "sig"}))
    from car_smart_assist.perception import weather_attention_training as training
    monkeypatch.setattr(training, "attention_model_config", lambda _cfg: {"architecture": "fixture"})
    (root / "reports").mkdir()
    (root / "reports/preflight.json").write_text(json.dumps(profile), encoding="utf-8")

    class Controller:
        requested = False
        def resume(self):
            self.requested = False
        def __enter__(self):
            return self
        def __exit__(self, *_args):
            return False
        def require_running(self):
            pass

    calls = []
    monkeypatch.setattr(module, "train_attention", lambda *_a, **_k: (calls.append("train"),
                                                                          (object(), {"best_epoch": 1}))[1])
    monkeypatch.setattr(module, "compare_attention", lambda *_a, **_k: {"status": "completed"})
    monkeypatch.setattr(module, "FileLock", lambda *_a, **_k: _NullContext())
    monkeypatch.setattr(module, "StopController", lambda *_a, **_k: Controller())
    return module, cfg, calls


class _NullContext:
    def __enter__(self):
        return self
    def __exit__(self, *_args):
        return False


def _load_script(relative):
    import importlib.util

    path = ROOT / relative
    spec = importlib.util.spec_from_file_location("weather_cli_" + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _profile(cfg, **overrides):
    return {"model_config": {"architecture": "fixture"},
            # Synthetic efficiency fields follow the report's public input schema;
            # they describe this fixture, not a measured preflight run.
            "models": {}, "cpu_threads": 1,
            "batch_transfer_ms": {"vector": 1.0, "selected_stack": 1.0},
            "backbone_training": {"fine_tune_from": cfg.backbone["fine_tune_from"],
                                  "freeze_epochs": cfg.backbone["freeze_epochs"],
                                  "probe_epoch": cfg.backbone["freeze_epochs"] + 1},
            "split_signature": "sig", "batch_size": cfg.train["batch_size"],
            "channels_last": cfg.train["channels_last"], "swad": cfg.train.get("swad"),
            "l2_sp_coefficient": cfg.train.get("l2_sp", 0), **overrides}


@pytest.mark.parametrize("profile_change", ["missing", "changed"])
def test_weather_cli_rejects_missing_or_changed_preflight_recipe_before_training(
    tmp_path, monkeypatch, profile_change
):
    from car_smart_assist.config.weather_attention import load_weather_attention_config

    cfg = replace(load_weather_attention_config(ROOT / "configs/model/weather_attention.yaml"),
                  checkpoint_root="weights", report_dir="reports", report_file="reports/REPORT.md")
    profile = _profile(cfg)
    if profile_change == "missing":
        del profile["backbone_training"]
    else:
        profile["backbone_training"]["fine_tune_from"] += 1
    module, _cfg, calls = _preflight_cli(tmp_path, monkeypatch, profile)
    assert module.main(["--config", "unused.yaml"]) == 2
    assert calls == []


def test_weather_cli_accepts_matching_preflight_recipe(tmp_path, monkeypatch):
    from car_smart_assist.config.weather_attention import load_weather_attention_config

    cfg = replace(load_weather_attention_config(ROOT / "configs/model/weather_attention.yaml"),
                  checkpoint_root="weights", report_dir="reports", report_file="reports/REPORT.md")
    module, _cfg, calls = _preflight_cli(tmp_path, monkeypatch, _profile(cfg))
    assert module.main(["--config", "unused.yaml"]) == 0
    assert calls == ["train"]
