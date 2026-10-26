"""集中配置与 CUDA 自动选择的行为测试。"""

import copy
import logging

import numpy as np
import pytest
import torch

from car_smart_assist.config.visibility import MODEL_DEFAULTS, TRAIN_DEFAULTS, training_config
from car_smart_assist.perception.visibility import dataset as visibility_dataset
from car_smart_assist.perception.visibility.autoencoder import build_autoencoder
from car_smart_assist.perception.visibility.dataset import VisibilityImageDataset
from car_smart_assist.perception.visibility.trainer import VisibilityTrainer, resolve_device


def test_training_defaults_do_not_mutate_input_or_other_instances():
    overrides = {"early_stopping": {"patience": 2}, "augmentation": {"contrast_range": [1, 1]}}
    cfg = training_config(overrides)
    cfg["early_stopping"]["enabled"] = False
    cfg["augmentation"]["contrast_range"][0] = 0
    assert overrides["augmentation"]["contrast_range"] == [1, 1]
    assert TRAIN_DEFAULTS["early_stopping"]["enabled"] is True
    assert training_config({})["early_stopping"]["enabled"] is True


def test_model_override_keeps_defaults_intact():
    cfg = {"base_channels": 2, "latent_dim": 4, "input_size": (16, 16)}
    model = build_autoencoder(cfg)
    assert model.to_latent.out_features == 4
    assert MODEL_DEFAULTS["base_channels"] == 32
    assert "in_channels" not in cfg


def test_auto_falls_back_to_cpu_without_cuda(monkeypatch, caplog):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with caplog.at_level(logging.WARNING):
        assert resolve_device("auto").type == "cpu"
    assert "自动使用 CPU" in caplog.text
    with pytest.raises(ValueError, match="无可用 CUDA GPU"):
        resolve_device("cuda")


def test_auto_prefers_cuda_and_explicit_gpu_index_is_checked(monkeypatch, caplog):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index: f"GPU-{index}")
    with caplog.at_level(logging.INFO):
        assert resolve_device("auto").type == "cuda"
        assert resolve_device("cuda:1").index == 1
    assert "GPU-1" in caplog.text
    with pytest.raises(ValueError, match="仅检测到 2 个"):
        resolve_device("cuda:2")


def test_explicit_cpu_does_not_probe_cuda(monkeypatch):
    def unexpected():
        raise AssertionError("显式 CPU 不应查询 CUDA")

    monkeypatch.setattr(torch.cuda, "is_available", unexpected)
    assert resolve_device("cpu").type == "cpu"


def test_configured_augmentation_changes_pixels(tmp_path):
    cache = tmp_path / "images.npy"
    np.save(cache, np.full((1, 16, 16, 3), 100, dtype=np.uint8))
    ds = VisibilityImageDataset(
        [tmp_path / "a.png"], (16, 16), cache_path=cache, augment=True,
        augmentation_cfg={"contrast_range": [1, 1], "brightness_range": [0.1, 0.1]},
    )
    torch.testing.assert_close(ds[0], torch.full((3, 16, 16), 100 / 255 + 0.1))
    with pytest.raises(ValueError, match="下限不大于上限"):
        VisibilityImageDataset([], augmentation_cfg={"contrast_range": [2, 1]})


@pytest.mark.parametrize("cleaning_manifest", [None, "cleaning.json"])
def test_trainer_honors_augmentation_switch(tmp_path, monkeypatch, cleaning_manifest):
    root = tmp_path / "images"
    root.mkdir()
    paths = [root / f"{i}.png" for i in range(20)]
    for path in paths:
        path.touch()
    calls = []

    def list_refs(acdc_root, *, cleaning_manifest=None, project_root="."):
        calls.append((acdc_root, cleaning_manifest, project_root))
        return paths

    monkeypatch.setattr(visibility_dataset, "list_ref_images", list_refs)
    cfg = {
        "device": "cpu",
        "model": {"base_channels": 2, "latent_dim": 4, "input_size": [16, 16]},
        "data": {"acdc_root": "images", "split_by_sequence": False, "cleaning_manifest": cleaning_manifest},
        "train": {"augment": False, "augmentation": {"contrast_range": [1, 1]}},
    }
    trainer = VisibilityTrainer(cfg, project_root=tmp_path, resume="none")
    train, val, calib, _ = trainer._build_datasets()
    assert train.augment is val.augment is calib.augment is False
    assert train.contrast_range == (1.0, 1.0)
    assert calls == [(tmp_path / "images", cleaning_manifest, tmp_path)]


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要可用 CUDA GPU")
@pytest.mark.parametrize("dtype", ["bfloat16", "float16"])
def test_actual_gpu_training_and_resume(tmp_path, monkeypatch, dtype):
    cfg = {
        "device": "auto",
        "model": {"base_channels": 2, "latent_dim": 4, "input_size": [16, 16]},
        "data": {"num_workers": 0},
        "train": {
            "epochs": 3, "batch_size": 2, "dtype": dtype,
            "checkpoint_dir": str(tmp_path),
            "early_stopping": {"enabled": True, "patience": 1, "min_delta": 1e6},
        },
    }
    samples = torch.linspace(0, 1, 4 * 3 * 16 * 16).reshape(4, 3, 16, 16)
    monkeypatch.setattr(
        VisibilityTrainer, "_build_datasets", lambda self: (samples, samples, samples, [])
    )
    trainer = VisibilityTrainer(cfg, resume="none")
    assert next(trainer.model.parameters()).device.type == "cuda"
    history = trainer.fit()
    assert np.isfinite(history.train_loss + history.val_loss).all()
    if dtype == "float16":
        checkpoint = torch.load(tmp_path / "last.pt", map_location="cpu", weights_only=False)
        assert checkpoint["scaler_state"]
        assert checkpoint["optimizer_state"]["param_groups"][0]["lr"] > 0
        resumed = VisibilityTrainer(cfg, resume="auto")
        captured_states = []
        original_step = torch.amp.GradScaler.step
        observe_resume_step = {"enabled": False}

        def capture_first_resume_step(scaler, optimizer, *args, **kwargs):
            if observe_resume_step["enabled"] and not captured_states:
                captured_states.append(copy.deepcopy(scaler.state_dict()))
            return original_step(scaler, optimizer, *args, **kwargs)

        # Public GradScaler API plus the immutable disk checkpoint oracle rejects a pending-only
        # implementation that never loads scaler state into the resumed scaler.
        monkeypatch.setattr(torch.amp.GradScaler, "step", capture_first_resume_step)
        observe_resume_step["enabled"] = True
        resumed.fit()
        assert resumed.scaler.is_enabled()
        assert len(captured_states) == 1
        assert captured_states[0] == checkpoint["scaler_state"]


def test_pipeline_rejects_unsupported_device_before_build_and_preserves_config(tmp_path):
    from car_smart_assist.inference.pipeline import InferencePipeline

    cfg = {"device": "auto", "data": {}, "model": {}, "train": {}}
    original = {key: value.copy() if isinstance(value, dict) else value for key, value in cfg.items()}
    with pytest.raises(ValueError, match="只支持 CPU 或 CUDA"):
        InferencePipeline.from_config(
            cfg, project_root=tmp_path, device="mps", require_visibility=False
        )
    assert cfg == original
