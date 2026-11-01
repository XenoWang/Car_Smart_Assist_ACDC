"""Validation uses the inference selection contract without changing training args."""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from car_smart_assist.config.detection import yolo_model_selection_contract


def selection_inputs(imgsz, *, joint=False):
    cfg = {
        "inference": {"imgsz": imgsz, "confidence": 0.25, "iou": 0.5, "max_det": 100},
        "joint_foreground_exclusive": False,
    }
    manifest = {
        "joint_supervision": joint,
        "known_class_count": 8,
        "source_weights": {"acdc": 0.75, "laf": 0.25},
        "valid_roi_ids": [1, 2],
    }
    return cfg, manifest, yolo_model_selection_contract(cfg, manifest)


@pytest.mark.parametrize("inference_imgsz,expected_val_imgsz", [(641, 672), (960, 960)])
@pytest.mark.parametrize("joint", [False, True])
def test_both_trainers_separate_validation_args_and_final_eval(
    tmp_path, monkeypatch, inference_imgsz, expected_val_imgsz, joint
):
    from ultralytics.engine import trainer as trainer_module
    from ultralytics.models.yolo.detect import DetectionTrainer
    from ultralytics.utils import checks

    from car_smart_assist.perception import joint_yolo_backend as backend
    from car_smart_assist.perception.yolo_training import detection_trainer_for_resume

    cfg, manifest, contract = selection_inputs(inference_imgsz, joint=joint)
    built = {}
    checked_sizes = []

    def build_yolo_dataset(args, img_path, batch, data, **kwargs):
        built["dataset_args"] = args
        built["imgsz"] = args.imgsz
        return "base-val-dataset"

    def check_imgsz(imgsz, *, max_dim):
        checked_sizes.append((imgsz, max_dim))
        return ((imgsz + 31) // 32) * 32

    monkeypatch.setattr("ultralytics.data.build_yolo_dataset", build_yolo_dataset)
    monkeypatch.setattr(checks, "check_imgsz", check_imgsz)

    train_build = {}

    def base_build_dataset(self, img_path, mode="train", batch=None):
        train_build["args"] = self.args
        return "base-train-dataset"

    monkeypatch.setattr(DetectionTrainer, "build_dataset", base_build_dataset)

    joint_builds = []

    def joint_dataset(**kwargs):
        joint_builds.append(kwargs)
        return "joint-dataset"

    monkeypatch.setattr(backend, "JointDataset", joint_dataset)

    validator_type = backend.JointValidator if joint else backend.DeploymentValidator
    validator_init = {}
    validator_calls = []

    def init_validator(self, dataloader, *, save_dir, args, _callbacks):
        validator_init.update(
            type=type(self), dataloader=dataloader, save_dir=save_dir, args=args,
            callbacks=_callbacks,
        )
        self.args = args

    def call_validator(self, model):
        validator_calls.append(model)
        return {"fitness": 0.81, "mAP50": 0.7}

    monkeypatch.setattr(validator_type, "__init__", init_validator)
    monkeypatch.setattr(validator_type, "__call__", call_validator)

    if joint:
        from car_smart_assist.perception.joint_yolo_backend import joint_trainer

        trainer_type = joint_trainer(None, selection_contract=contract)
    else:
        trainer_type = detection_trainer_for_resume(None, contract)

    trainer = trainer_type.__new__(trainer_type)
    original_args = SimpleNamespace(
        imgsz=640,
        conf=0.25,
        iou=0.7,
        max_det=300,
        task="detect",
        single_cls=False,
        agnostic_nms=False,
        batch=2,
        close_mosaic=0,
        mosaic=0,
        mixup=0,
        copy_paste=0,
        cutmix=0,
        degrees=0,
        translate=0,
        scale=0,
        shear=0,
        perspective=0,
        fliplr=0,
        flipud=0,
        multi_scale=0,
        hsv_h=0.015,
        plots=True,
    )
    trainer.args = original_args
    trainer.model = None
    trainer.data = "prepared-dataset"
    trainer.test_loader = "validation-loader"
    trainer.save_dir = tmp_path
    trainer.callbacks = {"callback": "sentinel"}

    train_dataset = trainer.build_dataset("train", mode="train", batch=2)
    val_dataset = trainer.build_dataset("val", mode="val", batch=2)
    assert train_dataset == ("joint-dataset" if joint else "base-train-dataset")
    if joint:
        assert joint_builds[0]["imgsz"] == 640
        assert joint_builds[0]["augment"] is True
        assert joint_builds[0]["hyp"] is original_args
    else:
        assert train_build["args"] is original_args
    assert val_dataset == ("joint-dataset" if joint else "base-val-dataset")
    if joint:
        assert joint_builds[1]["imgsz"] == expected_val_imgsz
        assert joint_builds[1]["augment"] is False
        assert joint_builds[1]["hyp"].imgsz == expected_val_imgsz
    else:
        assert built["imgsz"] == expected_val_imgsz
        assert built["dataset_args"].imgsz == expected_val_imgsz

    trainer.validator = trainer.get_validator()
    validator = trainer.validator
    assert type(validator) is validator_type
    validation_args = validator_init["args"]
    assert validation_args.imgsz == expected_val_imgsz
    assert validation_args.iou == 0.5
    assert validation_args.max_det == 100
    assert validation_args.conf == 0.001
    assert validation_args is not original_args
    assert checked_sizes == [(inference_imgsz, 1)] * 2
    assert original_args.imgsz == 640
    assert original_args.iou == 0.7
    assert original_args.max_det == 300
    assert original_args.conf == 0.25
    assert original_args.hsv_h == 0.015

    # The project wrapper must verify paired files and retain the base last/best validation path.
    trainer.last = tmp_path / "last.pt"
    trainer.best = tmp_path / "best.pt"
    trainer.last.write_bytes(b"synthetic last")
    trainer.best.write_bytes(b"synthetic best")
    trainer.wdir = tmp_path
    # Supply valid complete history before checking final-evaluation identity;
    # the independent expected validator args and strip side effects stay below.
    fixture_spec = importlib.util.spec_from_file_location(
        "validation_complete_state_fixture", Path(__file__).with_name("test_detection_training_cli.py")
    )
    fixture_module = importlib.util.module_from_spec(fixture_spec)
    fixture_spec.loader.exec_module(fixture_module)
    _, _, fixture_run = fixture_module.setup_run(tmp_path / "full_state_fixture", state_version=5)
    state = torch.load(fixture_run / "weights/last_training_state.pt", weights_only=True)
    state.update({
        "model_selection_contract": contract,
        "checkpoint_sha256": hashlib.sha256(trainer.last.read_bytes()).hexdigest(),
        "best_checkpoint_sha256": hashlib.sha256(trainer.best.read_bytes()).hexdigest(),
    })
    torch.save(state, tmp_path / "last_training_state.pt")
    stripped = []

    def strip_optimizer(path, updates=None):
        stripped.append((path, updates))
        return {"train_results": {"fitness": 0.5}}

    monkeypatch.setattr(trainer_module, "strip_optimizer", strip_optimizer)
    callbacks = []
    trainer.run_callbacks = callbacks.append
    trainer.final_eval()
    assert [path for path, _ in stripped] == [trainer.last, trainer.best]
    assert trainer._project_best_checkpoint_sha256 == hashlib.sha256(trainer.best.read_bytes()).hexdigest()
    trainer.best.write_bytes(b"replaced best")
    with pytest.raises(ValueError, match="Best checkpoint does not match"):
        trainer.final_eval()
    assert len(stripped) == 2  # 拒绝配对错误时不得先strip或评估。
    assert validator_calls == [trainer.best]
    assert trainer.metrics == {"mAP50": 0.7}
    assert callbacks == ["on_fit_epoch_end"]
    assert validator.args.imgsz == expected_val_imgsz


def test_joint_selection_contract_includes_source_policy():
    _, _, contract = selection_inputs(960, joint=True)
    assert contract["joint_selection"] == {
        "known_class_count": 8,
        "source_weights": {"acdc": 0.75, "laf": 0.25},
        "valid_roi_ids": [1, 2],
        "foreground_exclusive": False,
    }


def test_training_main_persists_selection_contract_in_metadata_and_summary(
    tmp_path, monkeypatch
):
    import importlib.util

    script_path = Path(__file__).resolve().parents[2] / "scripts/train_detection.py"
    spec = importlib.util.spec_from_file_location("training_cli_contract_persistence", script_path)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)

    data_dir = tmp_path / "joint-data"
    data_dir.mkdir()
    manifest = {
        "signature": "joint-manifest",
        "seed": 42,
        "classes": [{"name": name, "source_id": index} for index, name in enumerate(
            ["car", "truck", "bus", "motorcycle", "bicycle", "person", "rider", "train"])]
            + [{"name": "road_obstacle", "source_id": "lost_and_found:obstacle"}],
        "joint_supervision": True,
        "known_class_count": 8,
        "source_weights": {"acdc": 0.75, "lost_and_found": 0.25},
        "classification_loss_weights": {"acdc": 1.0, "lost_and_found": 1.0},
        "images": [],
        "valid_roi_ids": [1, 2],
    }
    (data_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    cfg = {
        "model": "initial.pt",
        "checkpoint": "missing.pt",
        "data": str(data_dir / "data.yaml"),
        "project": str(tmp_path / "runs"),
        "name": "contract-test",
        "device": "cpu",
        "train": {
            "epochs": 2, "seed": 42, "imgsz": 640, "batch": 2,
            "close_mosaic": 0, "resume_extra_epochs": None,
        },
        "inference": {"imgsz": 960, "confidence": 0.25, "iou": 0.5, "max_det": 100},
        "joint_foreground_exclusive": False,
    }
    monkeypatch.setattr(cli, "load_yolo_config", lambda _path: cfg)
    validation_calls = []
    def validate_export(_root, selected_data, _mapping=None, *, verify_hashes=False):
        assert verify_hashes is True
        validation_calls.append(Path(selected_data))
        return manifest, {"images": 1}, None

    from car_smart_assist.data import joint_paths
    monkeypatch.setattr(joint_paths, "read_detection_manifest", validate_export)
    monkeypatch.setattr(cli, "resolve_yolo_device", lambda _device: "cpu")
    monkeypatch.setattr(cli, "importlib", SimpleNamespace(metadata=SimpleNamespace(version=lambda _name: "synthetic-version")))
    monkeypatch.setenv("YOLO_CONFIG_DIR", str(tmp_path / "local-settings"))
    monkeypatch.setattr(cli, "attach_training_state_callbacks", lambda *args, **kwargs: None)
    # This entry-point test isolates contract propagation; paired-state rejection is covered separately.
    monkeypatch.setattr(cli, "load_training_state", lambda *_args: {"best_checkpoint_sha256": "fixture-best"})
    captured = {}

    class FakeModel:
        def __init__(self):
            self.callbacks = {}

        def add_callback(self, event, callback):
            self.callbacks[event] = callback

        def train(self, **kwargs):
            captured["train_kwargs"] = kwargs
            self.callbacks["on_model_save"](None)
            self.trainer = SimpleNamespace(best=tmp_path / "best.pt", last=tmp_path / "last.pt", epoch=1)
            return SimpleNamespace(results_dict={"metrics/mAP50(B)": 0.5})

    model = FakeModel()
    monkeypatch.setattr(cli, "load_yolo_model", lambda *_args: model)
    monkeypatch.setattr(cli, "detection_trainer_for_resume", lambda *_args, **_kwargs: object)
    monkeypatch.setattr(
        "car_smart_assist.perception.joint_yolo_backend.joint_trainer",
        lambda *args, **kwargs: captured.update(joint_contract=kwargs["selection_contract"]) or object,
    )
    monkeypatch.setattr(sys, "argv", [str(script_path), "--name", "contract-test"])

    assert cli.main() == 0
    # Literal public selection schema: do not compute this oracle with the factory.
    expected = {
        "schema_version": 1, "postprocess_version": "best_class_threshold_v1",
        "ap_mode": "ap_sweep", "sweep_confidence": 0.001, "imgsz": 960,
        "nms_iou": 0.5, "max_det": 100, "max_nms": 30000, "max_wh": 7680,
        "agnostic_nms": False, "class_selection": "original_best_class_only",
        "threshold_comparison": ">=", "task": "detect", "fitness": "0.1*mAP50+0.9*mAP50_95",
        "joint_selection": {"known_class_count": 8, "source_weights": {"acdc": 0.75, "lost_and_found": 0.25},
                            "valid_roi_ids": [1, 2], "foreground_exclusive": False},
    }
    assert validation_calls == [data_dir / "data.yaml"]
    run = Path(cfg["project"]) / cfg["name"]
    metadata = json.loads((run / "run_metadata.json").read_text(encoding="utf-8"))
    summary = json.loads((run / "validation_summary.json").read_text(encoding="utf-8"))
    assert captured["joint_contract"] == expected
    assert captured["train_kwargs"]["imgsz"] == 640
    assert metadata["model_selection_contract"] == expected
    assert summary["model_selection_contract"] == expected
