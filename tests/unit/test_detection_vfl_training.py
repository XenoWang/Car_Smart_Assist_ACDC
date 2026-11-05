"""Recipe identity and trainer/state propagation, with literal public contracts."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from car_smart_assist.config.detection import (
    load_yolo_config, normalize_classification_loss, yolo_model_selection_contract,
    yolo_selection_validator_args,
)
from car_smart_assist.perception.yolo_training import require_model_selection_contract

ROOT = Path(__file__).resolve().parents[2]
RECIPE = {"mode": "vfl", "alpha": .75, "gamma": 2.0}


def inputs():
    cfg = {"model": "initial.pt", "checkpoint": "run/weights/best.pt",
           "inference": {"imgsz": 960, "confidence": .25, "iou": .5, "max_det": 100},
           "train": {"epochs": 2, "seed": 42, "close_mosaic": 0}}
    manifest = {"joint_supervision": True, "known_class_count": 8,
                "source_weights": {"acdc": .5, "lost_and_found": .5}, "valid_roi_ids": [1, 2],
                "seed": 42, "signature": "fixture"}
    return cfg, manifest


def plain_contract():
    return {"schema_version": 1, "postprocess_version": "best_class_threshold_v1",
            "ap_mode": "ap_sweep", "sweep_confidence": .001, "imgsz": 960, "nms_iou": .5,
            "max_det": 100, "max_nms": 30000, "max_wh": 7680, "agnostic_nms": False,
            "class_selection": "original_best_class_only", "threshold_comparison": ">=",
            "task": "detect", "fitness": "0.1*mAP50+0.9*mAP50_95",
            "joint_selection": {"known_class_count": 8, "source_weights": {"acdc": .5, "lost_and_found": .5},
                                "valid_roi_ids": [1, 2], "foreground_exclusive": True}}


@pytest.mark.parametrize("value", [False, 0, "", [], {"mode": False}, {"mode": "VFL"}, {"typo": 1},
    {"mode": "bce", "alpha": .75}, {"mode": "vfl", "alpha": True}, {"mode": "vfl", "alpha": ".75"},
    {"mode": "vfl", "alpha": -1}, {"mode": "vfl", "gamma": -1}, {"mode": "vfl", "gamma": "2"},
    {"mode": "vfl", "gamma": float("nan")}, {"mode": "vfl", "alpha": float("inf")}])
def test_invalid_recipe_is_explicitly_rejected(value):
    with pytest.raises(ValueError, match="classification_loss"):
        normalize_classification_loss(value)


def test_canonical_defaults_and_parameters_do_not_alias_caller():
    for value in (None, {}, {"mode": "bce"}):
        assert normalize_classification_loss(value) == {"mode": "bce"}
    raw = {"mode": "vfl"}
    actual = normalize_classification_loss(raw)
    assert actual == RECIPE
    actual["alpha"] = 0
    assert raw == {"mode": "vfl"}
    assert normalize_classification_loss({"mode": "vfl", "alpha": 2, "gamma": 0}) == {
        "mode": "vfl", "alpha": 2., "gamma": 0.,
    }


def test_bce_contract_remains_legacy_exact_and_vfl_only_adds_recipe():
    cfg, manifest = inputs()
    old = yolo_model_selection_contract(cfg, manifest)
    assert old == plain_contract()
    cfg["classification_loss"] = {"mode": "bce"}
    assert yolo_model_selection_contract(cfg, manifest) == old
    cfg["classification_loss"] = {"mode": "vfl"}
    new = yolo_model_selection_contract(cfg, manifest)
    assert new == {**plain_contract(), "training_classification_loss": RECIPE}
    assert yolo_selection_validator_args(new) == yolo_selection_validator_args(old)
    with pytest.raises(ValueError, match="joint"):
        yolo_model_selection_contract(cfg, {**manifest, "joint_supervision": False})


@pytest.mark.parametrize("previous,current", [
    ({"mode": "bce"}, RECIPE), (RECIPE, {"mode": "bce"}),
    (RECIPE, {**RECIPE, "alpha": .5}), (RECIPE, {**RECIPE, "gamma": 1.}),
])
def test_loss_changes_refuse_same_run_before_loading_any_checkpoint(tmp_path, monkeypatch, previous, current):
    from _model_test_helpers import load_module
    cli = load_module("scripts/train_detection.py", "vfl_resume_cli_test")
    cfg, manifest = inputs()
    run = tmp_path / "run"
    (run / "weights").mkdir(parents=True)
    for name in ("last.pt", "best.pt"):
        (run / "weights" / name).write_bytes(b"must not deserialize")
    old = plain_contract()
    if previous["mode"] == "vfl":
        old["training_classification_loss"] = previous
    (run / "run_metadata.json").write_text(json.dumps({"model_selection_contract": old}), encoding="utf-8")
    cfg["classification_loss"] = current
    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("Recipe mismatch must reject before checkpoint load"))
    with pytest.raises(ValueError, match="contract"):
        cli.resolve_training_plan(tmp_path, cfg, manifest, run)
    assert (run / "weights/last.pt").read_bytes() == b"must not deserialize"


def test_paired_configs_share_budget_data_initialization_and_keep_default_route():
    a = load_yolo_config(ROOT / "configs/model/yolo_vfl_baseline.yaml")
    b = load_yolo_config(ROOT / "configs/model/yolo_vfl_trial.yaml")
    assert a["classification_loss"] == {"mode": "bce"} and b["classification_loss"] == RECIPE
    assert a["train"]["epochs"] == b["train"]["epochs"] == 80
    assert a["train"]["patience"] == b["train"]["patience"] == 0
    assert a["enabled_for_inference"] is b["enabled_for_inference"] is False
    assert a["device"] == b["device"] == "cuda:0"
    for key in ("name", "checkpoint", "calibration_file"):
        assert a[key] != b[key]
        a.pop(key); b.pop(key)
    a.pop("classification_loss"); b.pop("classification_loss")
    assert a == b
    default = yaml.safe_load((ROOT / "configs/model/yolo_detection.yaml").read_text(encoding="utf-8"))["yolo_detection"]
    assert default.get("classification_loss") is None



@pytest.mark.parametrize("value", [False, 0, "", [], {}, {"alpha": .75}])
def test_invalid_low_level_recipe_is_rejected_before_vendor_criterion(monkeypatch, value):
    import ultralytics.utils.loss as vendor
    from car_smart_assist.perception.joint_yolo import JointDetectionLoss
    monkeypatch.setattr(vendor, "v8DetectionLoss", lambda *a, **k: pytest.fail("Invalid recipe reached vendor construction"))
    with pytest.raises(ValueError, match="classification_loss"):
        JointDetectionLoss(object(), 8, classification_loss=value)
    from car_smart_assist.perception.joint_yolo import MaskedClassificationLoss
    with pytest.raises(ValueError, match="classification_loss"):
        MaskedClassificationLoss(classification_loss=value)


def test_trainer_uses_bound_recipe_and_stores_it_on_model(monkeypatch):
    from car_smart_assist.perception import joint_yolo_backend as backend
    cfg, manifest = inputs()
    cfg["classification_loss"] = RECIPE
    contract = yolo_model_selection_contract(cfg, manifest)
    model = SimpleNamespace()
    class Base:
        def get_model(self, *a, **k):
            return model
        def set_model_attributes(self):
            pass
    monkeypatch.setattr(backend, "detection_trainer_for_resume", lambda *a, **k: Base)
    monkeypatch.setattr(backend, "load_joint_index", lambda data: (
        {**manifest, "classes": [{"name": str(i)} for i in range(9)]}, {}))
    observed = {}
    def criterion(m, obstacle, exclusive, **kwargs):
        observed.update(model=m, obstacle=obstacle, exclusive=exclusive, recipe=kwargs["classification_loss"])
        return "bound-criterion"
    monkeypatch.setattr(backend, "JointDetectionLoss", criterion)
    trainer = backend.joint_trainer(None, selection_contract=contract)()
    trainer.data = {}
    trainer.model = trainer.get_model(weights=None)
    trainer.set_model_attributes()
    assert model.joint_classification_loss == RECIPE
    assert model.criterion == "bound-criterion"
    assert observed["model"] is model and observed["obstacle"] == 8 and observed["exclusive"] is True
    assert observed["recipe"] == RECIPE


def test_real_state_writer_preserves_recipe_and_load_rejects_changed_recipe(tmp_path, monkeypatch):
    from _model_test_helpers import load_module
    fixture = load_module("tests/unit/test_yolo_training_state.py", "vfl_state_fixture")
    from car_smart_assist.perception.yolo_training import attach_training_state_callbacks, load_training_state
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    contract = {**plain_contract(), "training_classification_loss": dict(RECIPE)}
    callbacks = {}
    model = SimpleNamespace(add_callback=lambda name, fn: callbacks.update({name: fn}))
    trainer = fixture.trainer(tmp_path)
    attach_training_state_callbacks(model, "dataset", None, selection_contract=contract,
                                    training_segment=fixture.TRAINING_SEGMENT)
    callbacks["on_model_save"](trainer)
    state = load_training_state(tmp_path / "last_training_state.pt", trainer.last, "dataset", contract)
    assert state["format_version"] == 5
    assert state["model_selection_contract"]["training_classification_loss"] == RECIPE
    wrong = {**contract, "training_classification_loss": {**RECIPE, "gamma": 1.}}
    with pytest.raises(ValueError, match="contract"):
        load_training_state(tmp_path / "last_training_state.pt", trainer.last, "dataset", wrong)
    with pytest.raises(ValueError, match="contract"):
        require_model_selection_contract(state, plain_contract())
