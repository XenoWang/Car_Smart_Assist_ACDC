from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from car_smart_assist.perception.visibility.autoencoder import build_autoencoder
from car_smart_assist.perception.visibility.scorer import (
    InformationFeatures,
    VisibilityScorer,
    compute_information_features,
    information_score,
)

ROOT = Path(__file__).resolve().parents[2]


def load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(f"test_script_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_near_black_uint8_and_normalized_float_keep_the_same_feature_scale():
    image = np.zeros((32, 48, 3), dtype=np.uint8)
    image[:, ::2] = 1
    uint8_features = compute_information_features(image)
    float_features = compute_information_features(image.astype(np.float32) / 255.0)
    for name in ("contrast", "entropy", "edge_density", "hf_ratio"):
        assert getattr(uint8_features, name) == pytest.approx(getattr(float_features, name))


@pytest.mark.parametrize("scoring", [
    {"feature_scales": {"contrast": float("nan")}},
    {"aggregation": {"weights": {"contrast": float("nan")}}},
    {"aggregation": {"power": 0.0}},
])
def test_visibility_scorer_rejects_invalid_scoring_overrides(scoring):
    model = build_autoencoder({"input_size": (32, 32), "base_channels": 4, "latent_dim": 8})
    with pytest.raises(ValueError):
        VisibilityScorer(model, scoring_cfg=scoring)


def test_visibility_scoring_accepts_a_valid_override():
    scoring = {
        "feature_scales": {"contrast": 1.0, "entropy": 1.0, "edge_density": 1.0, "hf_ratio": 1.0},
        "aggregation": {"weights": {"contrast": 1.0, "entropy": 1.0,
                                      "edge_density": 1.0, "hf_ratio": 0.0},
                        "power": -1.0, "feature_floor": 0.1},
    }
    scoring["aggregation"]["weights"]["entropy"] = 2.0
    features = InformationFeatures(0.2, 0.4, 0.8, 1.0)
    score = information_score(features, scoring_cfg=scoring)
    # 归一化特征为 0.2/0.4/0.8/1.0，权重为 1/2/1/0：
    # M_-1 = ( (5 + 2*2.5 + 1.25) / 4 )^-1 = 16/45。
    assert score == pytest.approx(16 / 45)


def test_rect_pad_joint_roi_uses_actual_intermediate_image_size(monkeypatch):
    monkeypatch.setenv("YOLO_CONFIG_DIR", str(ROOT / "artifacts/ultralytics_settings"))
    from car_smart_assist.perception import joint_yolo_backend as backend

    image_path = Path("frame.png").resolve()
    dataset = object.__new__(backend.JointDataset)
    dataset.index = {str(image_path): {"roi_mask": True, "supervised_classes": [0], "dataset": "acdc"}}
    dataset.manifest = {"classes": ["obstacle"], "classification_loss_weights": {"acdc": 1.0}}
    dataset.augment = False
    dataset.imgsz = 960
    monkeypatch.setattr(backend, "native_roi", lambda *_: np.ones((1024, 2048), np.uint8))
    monkeypatch.setattr(backend.JointDataset, "get_image_and_label", lambda _s, _i: {
        "im_file": str(image_path), "img": np.zeros((480, 960, 3), np.uint8),
        "rect_shape": (512, 992),
    })
    monkeypatch.setattr(dataset, "transforms", lambda _labels: {
        "img": torch.zeros((3, 512, 992), dtype=torch.uint8),
    }, raising=False)
    roi = dataset[0]["obstacle_roi"].numpy()
    assert roi.shape == (512, 992)
    assert not roi[:16].any() and not roi[-16:].any()
    assert not roi[:, :16].any() and not roi[:, -16:].any()
    assert roi[16:496, 16:976].all()


def test_pipeline_cli_does_not_claim_a_fixed_followup_latency(monkeypatch, capsys):
    module = load_script("run_pipeline.py")
    monkeypatch.setattr(module, "parse_args", lambda: SimpleNamespace(
        log_level="WARNING", config="configs/model/visibility.yaml", checkpoint=None, device=None,
        weather_config=None, weather_loss=None, weather_checkpoint=None,
        image=["sample.png"], synthetic=False, json=False,
    ))
    monkeypatch.setattr(module.yaml, "safe_load", lambda _stream: {"visibility": {}})

    class Result:
        visibility = SimpleNamespace(information=0.5, level=SimpleNamespace(value="BLIND"))
        advisory = SimpleNamespace(should_takeover=True, risk_level=SimpleNamespace(value="HIGH"), text="谨慎停车")
        skipped, weather_warning, timings_ms = {}, None, {"gate": 8.0}

    class Pipeline:
        @classmethod
        def from_config(cls, *_args, **_kwargs): return cls()
        def run_batch(self, images): return [Result() for _ in images]

    monkeypatch.setattr(module, "InferencePipeline", Pipeline)
    assert module.main() == 0
    output = capsys.readouterr().out
    assert "末帧门控内部诊断" in output
    assert "首次含模型预热，后续约 10ms" not in output


def test_visibility_evaluation_uses_checkpoint_size_for_reference_images(tmp_path, monkeypatch):
    module = load_script("evaluate_visibility.py")
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"fixture")
    config = {"visibility": {
        "data": {"acdc_root": "acdc", "project_root": ".", "cleaning_manifest": None},
        "train": {"checkpoint_dir": "checkpoints"}, "model": {"input_size": [32, 32]},
        "scoring": {}, "report": {"sample_per_condition": 1},
    }}
    monkeypatch.setattr(module, "load_visibility_config", lambda _p: config["visibility"])
    monkeypatch.setattr(sys, "argv", ["evaluate_visibility.py", "--checkpoint", str(checkpoint)])
    scorer = SimpleNamespace(input_size=(40, 80), scoring_cfg={}, calibration=None)
    monkeypatch.setattr(module.VisibilityScorer, "from_checkpoint", lambda *_a, **_k: scorer)
    gate = SimpleNamespace(thresholds=SimpleNamespace(use_recon_z=True))
    monkeypatch.setattr(module.VisibilityGate, "from_config", staticmethod(lambda _cfg: gate))
    monkeypatch.setattr(module, "resolve_device", lambda _device: "cpu")
    monkeypatch.setattr(module, "list_ref_images", lambda *_a, **_k: [tmp_path / "ref.png"])
    sizes = []
    monkeypatch.setattr(module, "load_images", lambda _paths, size:
        sizes.append(tuple(size)) or [np.zeros((size[0], size[1], 3), np.uint8)])
    monkeypatch.setattr(module, "eval_fit_quality", lambda *_a: {
        "test_train_ratio": 1.2, "verdict": "fixture",
    })
    monkeypatch.setattr(module, "eval_clean_control", lambda *_a: {"false_positive_rate": 0.1})
    monkeypatch.setattr(module, "eval_real_adverse", lambda *_a, **_k: {})
    monkeypatch.setattr(module, "eval_synthetic", lambda *_a: {"blur": {
        "information_monotonic_decreasing": True, "recall_at_max_severity": 0.5,
    }})
    captured = {}
    monkeypatch.setattr(module, "write_report", lambda report, _out: captured.update(report))
    assert module.main() == 0
    assert sizes == [(40, 80)] and captured["input_size"] == [40, 80]
    assert captured["thresholds"]["use_recon_z"] is True


def test_ablation_reference_preparation_consumes_cleaning_manifest(tmp_path, monkeypatch):
    module = load_script("run_visibility_ablations.py")
    refs = [tmp_path / "ref-a.png", tmp_path / "ref-b.png"]
    for path in refs: path.write_bytes(path.name.encode("utf-8"))
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"invalid": []}', encoding="utf-8")
    calls = []
    monkeypatch.setattr(module, "list_ref_images", lambda acdc, **kwargs:
        calls.append((acdc, kwargs)) or refs)
    monkeypatch.setattr(module, "sequence_of", lambda _p: "sequence")
    monkeypatch.setattr(module, "split_ref_indices", lambda *_a, **_k: (
        np.asarray([0]), np.asarray([], dtype=int), np.asarray([], dtype=int), np.asarray([1]),
    ))
    monkeypatch.setattr(module, "_file_identity", lambda path: {
        "path": Path(path).resolve().as_posix(),
        "sha256": __import__("hashlib").sha256(Path(path).read_bytes()).hexdigest()
        if Path(path).is_file() else "fixture",
    })
    cfg = {"data": {"acdc_root": "acdc", "project_root": ".",
                    "cleaning_manifest": "manifest.json", "split_by_sequence": False},
           "model": {"input_size": [32, 32]}}
    prepared = module._prepare_inputs(cfg, tmp_path)
    assert calls[0][1] == {"cleaning_manifest": "manifest.json", "project_root": tmp_path}
    before = prepared["data_identity"]["cleaning_manifest"]["sha256"]
    manifest.write_text('{"invalid": ["ref-a.png"]}', encoding="utf-8")
    after = module._prepare_inputs(cfg, tmp_path)["data_identity"]["cleaning_manifest"]["sha256"]
    assert before != after


def test_ablation_identity_hash_changes_with_config_or_data():
    module = load_script("run_visibility_ablations.py")
    identity = {"config": {"train": {"epochs": 5}}, "data": {"sha256": "a"}}
    assert module._signature(identity) != module._signature(
        {"config": {"train": {"epochs": 6}}, "data": {"sha256": "a"}})
    assert module._signature(identity) != module._signature(
        {"config": {"train": {"epochs": 5}}, "data": {"sha256": "b"}})
    assert module._signature(identity) != module._signature(
        {**identity, "evaluation_protocol": {"sample_seed": 1}})


def test_ablation_refuses_cached_row_with_stale_experiment_identity(tmp_path, monkeypatch):
    module = load_script("run_visibility_ablations.py")
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    monkeypatch.setattr(module, "__file__", str(scripts / "run_visibility_ablations.py"))
    (tmp_path / "cfg.yaml").write_text("fixture", encoding="utf-8")
    config = {"visibility": {"data": {}, "model": {}, "train": {},
                              "thresholds": {"use_recon_z": True}, "ablations": []}}
    monkeypatch.setattr(module.yaml, "safe_load", lambda _s: config)
    monkeypatch.setattr(sys, "argv", ["run_visibility_ablations.py", "--config", "cfg.yaml",
                                       "--only", "baseline"])
    monkeypatch.setattr(module, "_prepare_inputs", lambda *_a: {
        "paths": [], "size": (32, 32), "splits": tuple(np.asarray([i]) for i in range(4)),
        "data_identity": {"reference_count": 2, "ordered_reference_sha256": "new"},
        "protocol": {"version": "fixture-v2"},
    })
    out = tmp_path / "artifacts/reports/visibility/ablations"
    out.mkdir(parents=True)
    (out / "results.json").write_text(
        '{"baseline": {"val_fpr": 0.1, "synth": {}, "experiment_signature": "old",'
        ' "experiment_identity": {}}}', encoding="utf-8"
    )
    def forbidden_trainer(*_args, **_kwargs):
        pytest.fail("stale cached result must be rejected before training")
    monkeypatch.setattr(module, "VisibilityTrainer", forbidden_trainer)
    assert module.main() == 2


def test_visibility_report_qualifies_data_and_reports_effective_recon_setting(tmp_path):
    module = load_script("evaluate_visibility.py")
    report = {
        "config_path": "custom.yaml", "checkpoint": "best.pt", "input_size": [40, 80],
        "config": {"data": {"seed": 17}}, "scoring": {"aggregation": {"power": -2}},
        "evaluation_protocol": {"sample_per_condition": 3},
        "thresholds": {"use_recon_z": True, "info_blind": 0.2},
        "fit_quality": {
            "split_sizes": {"train": 1, "val": 1, "calib": 1, "test": 1},
            "train_recon_mean": 1.0, "val_recon_mean": 1.1, "calib_recon_mean": 1.2,
            "test_recon_mean": 1.3, "val_train_ratio": 1.1, "test_train_ratio": 1.3,
            "verdict": "fixture", "train_information_median": 0.8,
            "val_information_median": 0.7, "calib_information_median": 0.6,
            "test_information_median": 0.5, "n_each": 3, "test_false_positive_rate": 0.1,
        },
        "clean_control": {"n": 1, "false_positive_rate": 0.0, "degraded_rate": 0.0},
        "real_adverse": {},
        "synthetic": {"blur": {
            "information_monotonic_decreasing": True, "recall_at_max_severity": 0.5,
            "curve": [{"severity": 1.0, "blind_rate": 0.5,
                       "information_mean": 0.2, "recon_z_median": None}],
        }},
    }
    module.write_report(report, tmp_path)
    text = (tmp_path / "evaluation.md").read_text(encoding="utf-8")
    assert "custom.yaml" in text and "use_recon_z=True" in text
    assert "本次评估中仅用于报告" in text
    assert "不能视为真实召回的上界" in text
    assert "本次启用重建判据" in text
    assert "ACDC 里**没有**" not in text


def test_ablation_report_prints_each_rows_effective_identity(tmp_path):
    module = load_script("run_visibility_ablations.py")
    synth = {name: {"recall": 0.2,
                    "curve": [{"blind_rate": 0.2} for _ in module.SEVERITIES]}
             for name in module.KINDS}
    row = {
        "params_m": 0.1, "overfit_ratio": 1.2, "val_fpr": 0.05,
        "mean_recall": 0.2, "synth": synth, "train_seconds": 1,
        "overrides": {"train": {"epochs": 5}},
        "experiment_signature": "sig", "comparison_signature": "cmp",
        "experiment_identity": {"config": {"train": {"epochs": 5}},
                                "data": {"reference_count": 12},
                                "evaluation_protocol": {"sample_seed": 0}},
        "thresholds": {"use_recon_z": True}, "scoring": {"aggregation": {"power": -2}},
    }
    assert module.write_report({"baseline": row}, tmp_path) == 0
    text = (tmp_path / "ablations.md").read_text(encoding="utf-8")
    assert "epochs" in text and "reference_count" in text
    assert "use_recon_z" in text and "power" in text and "sig" in text


def test_weather_attention_report_uses_actual_selection_weights_and_cpu_threads(tmp_path):
    from dataclasses import replace
    from car_smart_assist.config.weather_attention import load_weather_attention_config
    from car_smart_assist.perception.weather_attention_training import model_report

    cfg = load_weather_attention_config(ROOT / "configs/model/weather_attention.yaml")
    evaluation = {**cfg.evaluation, "ap_selection_fraction": 0.7, "cpu_threads": 2,
                  "selection_weights": {name: 1.0 for name in cfg.attributes}}
    cfg = replace(cfg, evaluation=evaluation, train={**cfg.train, "epochs": 13},
                  data={**cfg.data, "seed": 9}, report_file="model_report.md", report_dir="reports")
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    (report_dir / "preflight.json").write_text(
        '{"models": {}, "batch_transfer_ms": {"vector": 1, "selected_stack": 1}, "cpu_threads": 2}',
        encoding="utf-8",
    )
    model_report(cfg, tmp_path, {"status": "fixture", "epoch": 1})
    text = (tmp_path / "model_report.md").read_text(encoding="utf-8")
    assert "AP 占 70%" in text and "当前配置为 2" in text
    assert "weather_attention.yaml" not in text and '"epochs": 13' in text




def test_random_split_training_is_rejected_before_creating_output(tmp_path):
    from car_smart_assist.perception.visibility.trainer import VisibilityTrainer
    cfg = {"data": {"split_mode": "random"}, "device": "cpu",
           "train": {"checkpoint_dir": "checkpoints"}}
    with pytest.raises(ValueError, match="split_mode=fixed"):
        VisibilityTrainer(cfg, project_root=tmp_path, resume="none")
    assert not (tmp_path / "checkpoints").exists()


def test_random_split_fit_quality_is_rejected_before_scoring():
    module = load_script("evaluate_visibility.py")
    with pytest.raises(ValueError, match="split_mode=fixed"):
        module.eval_fit_quality(None, [], [], {"split_mode": "random"}, None, 1)


def test_random_split_evaluation_cli_is_rejected_before_loading_checkpoint(tmp_path, monkeypatch):
    module = load_script("evaluate_visibility.py")
    cfg = tmp_path / "random.yaml"
    cfg.write_text("visibility:\n  data:\n    split_mode: random\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["evaluate_visibility.py", "--config", str(cfg)])
    monkeypatch.setattr(module.VisibilityScorer, "from_checkpoint",
                        lambda *_a, **_k: pytest.fail("random split must fail before model load"))
    assert module.main() == 2


def test_random_split_ablation_preparation_is_rejected_before_reading_images(tmp_path, monkeypatch):
    module = load_script("run_visibility_ablations.py")
    monkeypatch.setattr(module, "list_ref_images",
                        lambda *_a, **_k: pytest.fail("random split must fail before enumeration"))
    with pytest.raises(ValueError, match="split_mode=fixed"):
        module._prepare_inputs({"data": {"split_mode": "random"}}, tmp_path)


def test_random_split_ablation_preset_is_rejected_before_output_or_training(tmp_path, monkeypatch):
    module = load_script("run_visibility_ablations.py")
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    monkeypatch.setattr(module, "__file__", str(scripts / "run_visibility_ablations.py"))
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("visibility:\n  data:\n    split_mode: fixed\n  ablations:\n"
                   "    - tag: random-preset\n      set:\n        data.split_mode: random\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["run_visibility_ablations.py", "--config", str(cfg),
                                     "--only", "random-preset"])
    monkeypatch.setattr(module, "VisibilityTrainer",
                        lambda *_a, **_k: pytest.fail("random split must fail before training"))
    assert module.main() == 2
    assert not (tmp_path / "artifacts").exists()
