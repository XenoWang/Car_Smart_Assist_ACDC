"""Small isolated loaders and independent filesystem snapshots for model tests."""

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]


def load_module(relative, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    before = list(sys.path)
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = before
    return module


def snapshot(root):
    return {p.relative_to(root): (p.read_bytes(), p.stat().st_mtime_ns)
            for p in root.rglob("*") if p.is_file()}


def synthetic_attention_preflight(cfg, root, _datasets, info):
    """A schema-valid profile for tiny-loop fixtures, never a measured benchmark."""
    from car_smart_assist.perception import weather_attention_training as training

    profile = {
        "model_config": training.attention_model_config(cfg),
        "backbone_training": {"fine_tune_from": cfg.backbone["fine_tune_from"],
                              "freeze_epochs": cfg.backbone["freeze_epochs"],
                              "probe_epoch": cfg.backbone["freeze_epochs"] + 1},
        "split_signature": info["signature"], "batch_size": cfg.train["batch_size"],
        "channels_last": cfg.train["channels_last"], "swad": cfg.train.get("swad"),
        "l2_sp_coefficient": cfg.train.get("l2_sp", 0), "models": {}, "cpu_threads": 1,
        "batch_transfer_ms": {"vector": 1.0, "selected_stack": 1.0},
    }
    training.write_json(root / cfg.report_dir / "preflight.json", profile)
    return profile
