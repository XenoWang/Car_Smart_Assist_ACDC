"""Real tiny joint YOLO CLI save/pause/restore; synthetic data and local architecture."""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import random

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from _model_test_helpers import load_module
from test_yolo_center_loss_training import tiny_architecture


def synthetic_joint_dataset(root):
    classes = [{"name": f"traffic_{i}", "source_id": i} for i in range(8)]
    classes.append({"name": "road_obstacle", "source_id": "lost_and_found:obstacle"})
    rows = []
    for split, side in (("train", .5), ("val", .75)):
        images, labels = root/"images"/split, root/"labels"/split
        images.mkdir(parents=True); labels.mkdir(parents=True)
        for index, category in enumerate((0, 7)):
            image = images/f"{index}.png"
            pixels = np.full((64, 64, 3), 40+30*index, np.uint8)
            pixels[16:48, 16:48] = (180, 90, 30)
            Image.fromarray(pixels).save(image)
            (labels/f"{index}.txt").write_text(f"{category} 0.5 0.5 {side} {side}\n", encoding="utf-8")
            rows.append({"image": image.relative_to(root).as_posix(), "source": str(image),
                         "dataset": "acdc", "roi_mask": None, "supervised_classes": list(range(8)),
                         "boxes": [[category, .5, .5, side, side]]})
    manifest = {"signature": "synthetic-center-runtime-v1", "seed": 42, "joint_supervision": True,
                "classes": classes, "known_class_count": 8, "valid_roi_ids": [1, 2],
                "source_weights": {"acdc": .8, "lost_and_found": .2},
                "classification_loss_weights": {"acdc": 1., "lost_and_found": 1.},
                "images": rows}
    manifest_path = root/"manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    data = {"path": root.as_posix(), "train": "images/train", "val": "images/val",
            "names": {i: c["name"] for i, c in enumerate(classes)},
            "joint_manifest": str(manifest_path), "channels": 3}
    data_path = root/"dataset.yaml"
    data_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return data_path, manifest


def test_real_joint_cli_normalizes_train_only_and_preserves_recipe_across_pause_continue(tmp_path, monkeypatch):
    import sys
    cpu_rng, python_rng, numpy_rng = torch.get_rng_state(), random.getstate(), np.random.get_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    cudnn = torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic
    env_keys = ("CUDA_VISIBLE_DEVICES", "CUBLAS_WORKSPACE_CONFIG", "PYTHONHASHSEED", "YOLO_CONFIG_DIR")
    environment = {key: os.environ.get(key) for key in env_keys}
    torch.set_num_threads(1)
    monkeypatch.setenv("YOLO_CONFIG_DIR", str(tmp_path/"settings"))
    try:
        from ultralytics.data import utils as data_utils
        from ultralytics.utils import downloads
        from ultralytics.engine import trainer as trainer_module, validator as validator_module
        from ultralytics.utils.torch_utils import select_device
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        def select_cpu(device, *args, **kwargs):
            return select_device(torch.device("cpu") if str(device) == "cpu" else device, *args, **kwargs)
        monkeypatch.setattr(trainer_module, "select_device", select_cpu)
        monkeypatch.setattr(validator_module, "select_device", select_cpu)
        monkeypatch.setattr(data_utils, "check_font", lambda *a, **k: None)
        monkeypatch.setattr(downloads, "safe_download", lambda *a, **k: pytest.fail("test attempted download"))
        data_path, manifest = synthetic_joint_dataset(tmp_path/"data")
        model_path = tmp_path/"tiny.yaml"
        model_path.write_text(yaml.safe_dump(tiny_architecture()), encoding="utf-8")
        run = tmp_path/"runs"/"candidate"
        cfg = {"model": str(model_path), "data": str(data_path), "checkpoint": str(run/"weights/best.pt"),
               "project": str(run.parent), "name": run.name, "device": "cpu",
               "inference": {"imgsz": 64, "confidence": .25, "iou": .5, "max_det": 10},
               "train": {"epochs": 2, "seed": 42, "imgsz": 64, "batch": 2, "workers": 0,
                         "patience": 20, "deterministic": False, "amp": False, "pretrained": False,
                         "optimizer": "AdamW", "lr0": .001, "warmup_epochs": 0., "nbs": 2,
                         "close_mosaic": 0, "mosaic": 0., "mixup": 0., "copy_paste": 0.,
                         "cutmix": 0., "degrees": 0., "translate": 0., "scale": 0., "shear": 0.,
                         "perspective": 0., "fliplr": 0., "flipud": 0., "multi_scale": False,
                         "plots": False, "cache": False, "save": True, "val": True},
               "center_loss": {"enabled": True, "weight": .1, "normalization": "train_mean_sqrt_area"}}
        config_path = tmp_path/"config.yaml"
        config_path.write_text(yaml.safe_dump({"yolo_detection": cfg}, sort_keys=False), encoding="utf-8")
        cli = load_module("scripts/train_detection.py", "center_runtime_cli")
        monkeypatch.setattr(cli, "__file__", str(tmp_path/"scripts"/"train_detection.py"))
        # Dataset export integrity is separately covered by existing tests. This
        # seam provides a real local YAML/manifest to the actual trainer/validator.
        @contextmanager
        def scope(*args, **kwargs):
            yield data_path, manifest, {"train": 2, "val": 2}
        monkeypatch.setattr(cli, "detection_data_scope", scope)
        real_model = cli.load_yolo_model
        paused = {"done": False}
        observations = []
        def load_model(root, checkpoint):
            model = real_model(root, checkpoint)
            def training_started(trainer):
                loss = trainer.model.criterion
                observations.append((loss.center_bbox is not None,
                                     trainer._project_center_loss["scale_pixels"]))
            def pause_once(trainer):
                if not paused["done"]:
                    paused["done"] = True
                    marker = run/"weights/STOP"
                    marker.parent.mkdir(parents=True, exist_ok=True)
                    marker.write_bytes(b"synthetic user pause")
            model.add_callback("on_train_start", training_started)
            model.add_callback("on_train_batch_end", pause_once)
            return model
        monkeypatch.setattr(cli, "load_yolo_model", load_model)
        monkeypatch.setattr(sys, "argv", ["train", "--config", str(config_path), "--fresh"])
        assert cli.main() == 75
        state_path = run/"weights/last_training_state.pt"
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        metadata = json.loads((run/"run_metadata.json").read_text(encoding="utf-8"))
        # train boxes32x32; val boxes48x48. C must remain exactly32.
        assert state["center_loss"]["scale_pixels"] == 32.
        assert state["center_loss"]["ground_truth_count"] == 2
        assert state["center_loss"] == metadata["center_loss"]
        assert state["manual_stop"] is True and state["epoch"] == 0
        assert state["training_complete"] is False
        assert "center_loss" not in metadata["train_args"]
        monkeypatch.setattr(sys, "argv", ["train", "--config", str(config_path), "--continue"])
        assert cli.main() == 0
        continued = torch.load(state_path, map_location="cpu", weights_only=True)
        assert continued["epoch"] == 1 and continued["training_complete"] is True
        assert continued["manual_stop"] is False
        assert continued["center_loss"] == state["center_loss"]
        assert not (run/"weights/STOP").exists()
        assert observations == [(True, 32.), (True, 32.)]
        assert continued["checkpoint_sha256"] == hashlib.sha256((run/"weights/last.pt").read_bytes()).hexdigest()
        assert continued["best_checkpoint_sha256"] == hashlib.sha256((run/"weights/best.pt").read_bytes()).hexdigest()

        marker = run/"weights/STOP"
        marker.write_bytes(b"pause must survive rejected recipe")
        protected = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in (run/"weights").iterdir() if p.is_file()}
        for p in (run/"run_metadata.json", run/"validation_summary.json"):
            if p.is_file():
                protected[p] = (p.read_bytes(), p.stat().st_mtime_ns)
        cfg["center_loss"]["weight"] = .2
        config_path.write_text(yaml.safe_dump({"yolo_detection": cfg}, sort_keys=False), encoding="utf-8")
        assert cli.main() == 2
        assert observations == [(True, 32.), (True, 32.)]  #no third trainer started
        assert {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in protected} == protected
    finally:
        monkeypatch.undo()
        for key, value in environment.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        torch.set_rng_state(cpu_rng); random.setstate(python_rng); np.random.set_state(numpy_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        torch.set_num_threads(threads)
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = cudnn
