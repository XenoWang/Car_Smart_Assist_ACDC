"""Joint dataset option rejection, nine-class defaults and deployment filtering."""

from copy import deepcopy
import json
import os
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from _model_test_helpers import load_module, snapshot


DEFAULTS = {"fraction": 1.0, "classes": None, "cache": False, "single_cls": False}
NONDEFAULT = [
    ("fraction", 0.5), ("classes", [0]), ("cache", "disk"), ("single_cls", True),
]


@pytest.fixture(autouse=True)
def preserve_rng_and_dependency_environment(monkeypatch, tmp_path):
    python_state, numpy_state = random.getstate(), np.random.get_state()
    torch_state = torch.get_rng_state().clone()
    monkeypatch.setenv("YOLO_CONFIG_DIR", os.environ.get("YOLO_CONFIG_DIR", str(tmp_path / "settings")))
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)


@pytest.fixture
def dataset_inputs(tmp_path):
    from ultralytics.cfg import DEFAULT_CFG

    root = tmp_path / "dataset"
    images, labels = root / "images/train", root / "labels/train"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    classes = [{"name": f"traffic_{i}", "source_id": i} for i in range(8)]
    classes.append({"name": "obstacle", "source_id": "lost_and_found:obstacle"})
    rows = []
    for name, cls in (("a", 0), ("b", 7)):
        path = images / f"{name}.png"
        Image.new("RGB", (16, 16), (20, 30, 40)).save(path)
        (labels / f"{name}.txt").write_text(f"{cls} 0.5 0.5 0.5 0.5\n", encoding="utf-8")
        rows.append({
            "image": f"images/train/{name}.png", "source": str(path),
            "dataset": "acdc", "roi_mask": None,
            "supervised_classes": list(range(8)), "boxes": [[cls, 0.5, 0.5, 0.5, 0.5]],
        })
    manifest = {
        "joint_supervision": True, "classes": classes, "known_class_count": 8,
        "valid_roi_ids": [1, 2], "source_weights": {"acdc": 0.8, "lost_and_found": 0.2},
        "classification_loss_weights": {"acdc": 1.0, "lost_and_found": 1.0},
        "images": rows,
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    hyp = deepcopy(DEFAULT_CFG)
    for key in ("mosaic", "mixup", "copy_paste", "cutmix", "degrees", "translate",
                "scale", "shear", "perspective", "fliplr", "flipud", "multi_scale"):
        setattr(hyp, key, 0)
    hyp.imgsz, hyp.batch = 32, 2
    data = {"joint_manifest": str(manifest_path), "channels": 3,
            "names": {i: row["name"] for i, row in enumerate(classes)}}
    return {
        "img_path": str(images), "imgsz": 32, "batch_size": 2,
        "augment": False, "hyp": hyp, "rect": False, "data": data, "task": "detect",
    }


@pytest.mark.parametrize("explicit", [False, True])
def test_default_options_keep_both_rows_original_labels_and_nine_class_supervision(dataset_inputs, explicit):
    from car_smart_assist.perception.joint_yolo_backend import JointDataset

    options = dict(dataset_inputs)
    if explicit:
        options.update(DEFAULTS)
    protected = {Path(options["data"]["joint_manifest"]): Path(options["data"]["joint_manifest"]).read_bytes()}
    for path in Path(options["img_path"]).parent.parent.rglob("*"):
        if path.is_file():
            protected[path] = path.read_bytes()
    dataset = JointDataset(**options)
    assert len(dataset) == 2
    assert len(dataset.data["names"]) == 9
    assert [float(dataset[i]["cls"].item()) for i in range(2)] == [0.0, 7.0]
    for index in range(2):
        sample = dataset[index]
        assert sample["class_supervision"].tolist() == [True] * 8 + [False]
        assert bool(sample["obstacle_roi"].all())
    assert all(path.read_bytes() == value for path, value in protected.items())


@pytest.mark.parametrize("key,value", NONDEFAULT + [
    ("fraction", 0.0), ("classes", []), ("cache", True), ("cache", "ram"),
])
@pytest.mark.parametrize("location", ["keyword", "hyp"])
def test_direct_dataset_rejects_nondefault_options_before_reading_manifest(
    dataset_inputs, monkeypatch, tmp_path, key, value, location,
):
    from car_smart_assist.perception import joint_yolo_backend as backend

    options = deepcopy(dataset_inputs)
    if location == "keyword":
        options[key] = value
    else:
        setattr(options["hyp"], key, value)
    before = snapshot(tmp_path)

    def unexpected_read(*_a, **_k):
        pytest.fail("Unsupported options reached manifest loading")

    monkeypatch.setattr(backend, "load_joint_index", unexpected_read)
    with pytest.raises(ValueError, match=f"JointDataset requires {key}="):
        backend.JointDataset(**options)
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize("entry", ["validator", "trainer_train", "trainer_val"])
@pytest.mark.parametrize("key,value", NONDEFAULT)
def test_factories_reject_effective_options_before_dataset_or_validation_override(
    dataset_inputs, key, value, entry,
):
    from car_smart_assist.perception import joint_yolo_backend as backend

    args = deepcopy(dataset_inputs["hyp"])
    if entry == "validator":
        instance = object.__new__(backend.JointValidator)
        instance.args, instance.data, instance.stride = args, dataset_inputs["data"], 32
        call = lambda: instance.build_dataset(dataset_inputs["img_path"], batch=2)
    else:
        contract = {"imgsz": 32, "nms_iou": 0.5, "max_det": 100,
                    "sweep_confidence": 0.001, "agnostic_nms": False, "task": "detect"}
        trainer = backend.joint_trainer(None, selection_contract=contract)
        instance = object.__new__(trainer)
        instance.args = args
        instance.data = dataset_inputs["data"]
        call = lambda: instance.build_dataset(
            dataset_inputs["img_path"], mode="val" if entry == "trainer_val" else "train", batch=2,
        )
    # A real default dataset is a positive control for every entry, including
    # the validation-args path that resets single_cls.
    legal = call()
    assert len(legal) == 2 and len(legal.data["names"]) == 9
    setattr(args, key, value)
    with pytest.raises(ValueError, match=f"JointDataset requires {key}="):
        call()


@pytest.mark.parametrize("key,value", NONDEFAULT)
@pytest.mark.parametrize("location", ["positional-option", "positional-hyp"])
def test_positional_constructor_options_cannot_bypass_rejection(
    dataset_inputs, monkeypatch, tmp_path, key, value, location,
):
    from car_smart_assist.perception import joint_yolo_backend as backend

    # Installed BaseDataset's public positional order; independent of the
    # production binding helper. A legal construction checks this prerequisite.
    args = [
        dataset_inputs["img_path"], 32, False, False,
        deepcopy(dataset_inputs["hyp"]), "", False, 2, 32, 0.5, False, None, 1.0,
    ]
    data = dataset_inputs["data"]
    legal = backend.JointDataset(*args, data=data, task="detect")
    assert len(legal) == 2 and len(legal.data["names"]) == 9
    if location == "positional-hyp":
        setattr(args[4], key, value)
    else:
        args[{"cache": 2, "single_cls": 10, "classes": 11, "fraction": 12}[key]] = value
    before = snapshot(tmp_path)

    def unexpected_read(*_a, **_k):
        pytest.fail("Unsupported positional options reached manifest loading")

    monkeypatch.setattr(backend, "load_joint_index", unexpected_read)
    with pytest.raises(ValueError, match=f"JointDataset requires {key}="):
        backend.JointDataset(*args, data=data, task="detect")
    assert snapshot(tmp_path) == before


def _resume_inputs(root):
    helper = load_module("tests/unit/test_detection_training_cli.py", "joint_option_resume_fixture")
    cfg, manifest, run = helper.setup_run(root, epoch=0, state_version=5)
    manifest.update(joint_supervision=True, known_class_count=8,
                    source_weights={"acdc": 0.8, "lost_and_found": 0.2}, valid_roi_ids=[1, 2])
    manifest["classes"] = [{"name": f"traffic_{i}", "source_id": i} for i in range(8)]
    manifest["classes"].append({"name": "obstacle", "source_id": "lost_and_found:obstacle"})
    # This existing identity helper prepares matching fixture metadata; it is
    # not the option-rejection oracle or a test of the selection formula.
    contract = helper.cli.yolo_model_selection_contract(cfg, manifest)
    metadata_path = run / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["classes"] = manifest["classes"]
    metadata["model_selection_contract"] = contract
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    state_path = run / "weights/last_training_state.pt"
    state = torch.load(state_path, weights_only=True)
    state["model_selection_contract"] = contract
    torch.save(state, state_path)
    return helper.cli, cfg, manifest, run


@pytest.mark.parametrize("fresh", [False, True])
def test_default_training_plans_reach_fresh_and_resume_without_changing_weights(tmp_path, fresh):
    cli, cfg, manifest, run = _resume_inputs(tmp_path)
    cfg["train"].update(DEFAULTS)
    weights = snapshot(run / "weights")
    plan = cli.resolve_training_plan(tmp_path, cfg, manifest, run, fresh=fresh)
    assert plan["mode"] == ("fresh" if fresh else "resume")
    assert plan["train"]["epochs"] == 2
    assert snapshot(run / "weights") == weights


@pytest.mark.parametrize("saved", [False, True])
@pytest.mark.parametrize("key,value", NONDEFAULT)
def test_training_plan_rejects_current_or_saved_unsupported_options_without_rewriting_inputs(
    tmp_path, key, value, saved,
):
    cli, cfg, manifest, run = _resume_inputs(tmp_path)
    # A legal positive control proves the fixture reaches the intended plan.
    assert cli.resolve_training_plan(tmp_path, cfg, manifest, run)["mode"] == "resume"
    if saved:
        path = run / "run_metadata.json"
        metadata = json.loads(path.read_text(encoding="utf-8"))
        metadata["train_args"][key] = value
        path.write_text(json.dumps(metadata), encoding="utf-8")
    else:
        cfg["train"][key] = value
    before = snapshot(run)
    with pytest.raises(ValueError, match=f"JointDataset requires {key}="):
        cli.resolve_training_plan(tmp_path, cfg, manifest, run, fresh=not saved)
    assert snapshot(run) == before


@pytest.mark.parametrize("key,value", NONDEFAULT)
def test_non_joint_training_keeps_vendor_options(tmp_path, key, value):
    helper = load_module("tests/unit/test_detection_training_cli.py", "nonjoint_option_plan")
    cfg, manifest, run = helper.setup_run(tmp_path)
    cfg["train"][key] = value
    plan = helper.cli.resolve_training_plan(tmp_path, cfg, manifest, run, fresh=True)
    assert plan["mode"] == "fresh" and plan["train"][key] == value


@pytest.mark.parametrize("competitor", ["unsupported-class", "outside-roi"])
@pytest.mark.parametrize("limit,expected", [(1, []), (2, [0.8])])
def test_deployment_max_det_precedes_source_and_roi_filtering(tmp_path, competitor, limit, expected):
    from car_smart_assist.perception.joint_yolo_backend import JointValidator

    validator = object.__new__(JointValidator)
    validator.args = SimpleNamespace(conf=0.001, iou=0.5, single_cls=False,
                                    agnostic_nms=False, max_det=limit, task="detect")
    validator.names, validator.device, validator.end2end = {0: "traffic", 1: "obstacle"}, torch.device("cpu"), False
    validator.manifest = {"valid_roi_ids": [1, 2]}
    roi = None
    if competitor == "outside-roi":
        mask = np.zeros((16, 16), np.uint8)
        mask[:, :8] = 1
        roi = tmp_path / "roi.png"
        Image.fromarray(mask).save(roi)
    batch = {
        "imgsz": (16, 16), "ori_shape": (16, 16), "ratio_pad": ((1.0, 1.0), (0.0, 0.0)),
        "joint_row": {"supervised_classes": [1], "roi_mask": str(roi) if roi else None, "boxes": []},
    }
    raw = torch.tensor([[
        [12., 4.], [8., 8.], [2., 2.], [2., 2.],
        [0.9 if competitor == "unsupported-class" else 0.0, 0.0],
        [0.0 if competitor == "unsupported-class" else 0.9, 0.8],
    ]])
    post = validator.postprocess(raw)[0]
    assert post["conf"].tolist() == pytest.approx([0.9] if limit == 1 else [0.9, 0.8])
    result = validator._prepare_pred(post, batch)
    assert result["conf"].tolist() == pytest.approx(expected)
    assert result["cls"].tolist() == ([1.0] if expected else [])
    if expected:
        torch.testing.assert_close(result["bboxes"], torch.tensor([[3., 7., 5., 9.]]))
