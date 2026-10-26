"""PRD004 14/25/26: class identity, label ID space and crash-safe JSON publication."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from car_smart_assist.data import preprocessing as pp
from car_smart_assist.perception.joint_yolo import preserve_classification_outputs


CLASS_NAMES = ["person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"]


def classifier(classes, names):
    layers = [torch.nn.Sequential(torch.nn.Conv2d(4, classes, 1)) for _ in range(3)]
    return SimpleNamespace(model=[SimpleNamespace(nc=classes, cv3=layers)], names=names)


@pytest.mark.parametrize("names", [list(reversed(CLASS_NAMES)), None])
def test_rejects_wrong_or_missing_class_identity_without_copying(names):
    # Oracle: the manifest's explicit ordered names, never inferred from source rows.
    source = classifier(8, names)
    target = classifier(9, CLASS_NAMES + ["road_obstacle"])
    before = [layer.state_dict() for layer in target.model[-1].cv3]
    before = [{key: value.clone() for key, value in state.items()} for state in before]
    with pytest.raises(ValueError, match="class names/order"):
        preserve_classification_outputs(source, target, 8, CLASS_NAMES)
    for state, layer in zip(before, target.model[-1].cv3, strict=True):
        for key, value in state.items():
            torch.testing.assert_close(layer.state_dict()[key], value, rtol=0, atol=0)


def test_matching_eight_classes_preserve_rows_and_leave_obstacle_row_untouched():
    source = classifier(8, dict(enumerate(CLASS_NAMES)))
    target = classifier(9, CLASS_NAMES + ["road_obstacle"])
    source_rows = [
        (layer[-1].weight.detach().clone(), layer[-1].bias.detach().clone())
        for layer in source.model[-1].cv3
    ]
    obstacle = [
        (layer[-1].weight[8].detach().clone(), layer[-1].bias[8].detach().clone())
        for layer in target.model[-1].cv3
    ]
    assert preserve_classification_outputs(source, target, 8, CLASS_NAMES) == 3
    for layer, (weights, bias), (last_weight, last_bias) in zip(
        target.model[-1].cv3, source_rows, obstacle, strict=True,
    ):
        torch.testing.assert_close(layer[-1].weight[:8], weights, rtol=0, atol=0)
        torch.testing.assert_close(layer[-1].bias[:8], bias, rtol=0, atol=0)
        torch.testing.assert_close(layer[-1].weight[8], last_weight, rtol=0, atol=0)
        torch.testing.assert_close(layer[-1].bias[8], last_bias, rtol=0, atol=0)


def test_nine_class_initialization_does_not_enter_eight_class_copy():
    source = classifier(9, None)
    target = classifier(9, None)
    before = [layer[-1].weight.detach().clone() for layer in target.model[-1].cv3]
    assert preserve_classification_outputs(source, target, 8, CLASS_NAMES) == 0
    for layer, original in zip(target.model[-1].cv3, before, strict=True):
        torch.testing.assert_close(layer[-1].weight, original, rtol=0, atol=0)


@pytest.mark.parametrize("value, invalid", [(0, True), (7, False), (255, True)])
def test_label_ids_use_zero_void_and_raw_class_bounds(tmp_path, value, invalid):
    # PRD25: labelIds void=0 and valid raw class IDs <=33; 255 is trainIds' ignore.
    image = tmp_path / "rgb_anon/fog/train/scene/frame_rgb_anon.png"
    mask = tmp_path / "gt/fog/train/scene/frame_gt_labelIds.png"
    image.parent.mkdir(parents=True)
    mask.parent.mkdir(parents=True)
    Image.fromarray(np.full((8, 8, 3), 120, np.uint8)).save(image)
    Image.fromarray(np.full((8, 8), value, np.uint8)).save(mask)
    before = mask.read_bytes()
    report = pp.CleaningReport()
    pp.check_acdc(tmp_path, report, {}, {
        "image_integrity": False, "image_statistics": False,
        "acdc_image_mask_pairing": False, "acdc_mask_validity": True,
        "detection_annotations": False,
    }, num_workers=1)
    assert (report.status_of(mask) is pp.SampleStatus.INVALID) is invalid
    assert mask.read_bytes() == before
    if value == 0:
        assert any(issue.check == "acdc/mask_validity" and "ignore" in issue.message for issue in report.issues)


@pytest.mark.parametrize("failure", ["write", "fsync", "replace"])
def test_atomic_json_failure_preserves_original_and_removes_own_temporary(tmp_path, monkeypatch, failure):
    # Independent oracle: literal old bytes survive a partial write or failed publish.
    target = tmp_path / "manifest.json"
    original = b'{"invalid": ["old.png"]}'
    target.write_bytes(original)

    def fail(*_args):
        raise OSError(f"synthetic {failure}")

    if failure == "write":
        def partial(_payload, stream, **_kwargs):
            stream.write('{"invalid":')
            raise OSError("synthetic write")
        monkeypatch.setattr(pp, "json", SimpleNamespace(dump=partial))
    else:
        monkeypatch.setattr(pp, "os", SimpleNamespace(
            fsync=fail if failure == "fsync" else pp.os.fsync,
            replace=fail if failure == "replace" else pp.os.replace,
        ))
    with pytest.raises(OSError, match=f"synthetic {failure}"):
        pp._atomic_write_json(target, {"invalid": ["new.png"]})
    assert target.read_bytes() == original
    assert sorted(path.name for path in tmp_path.iterdir()) == ["manifest.json"]


@pytest.mark.parametrize("kind", ["manifest", "report"])
def test_public_cleaning_writers_preserve_previous_json_on_publication_failure(tmp_path, monkeypatch, kind):
    target = tmp_path / ("acdc.json" if kind == "manifest" else "report.json")
    original = b'{"retained_previous": true}'
    target.write_bytes(original)

    def denied(_source, _target):
        raise PermissionError("publication denied")

    monkeypatch.setattr(pp, "os", SimpleNamespace(fsync=pp.os.fsync, replace=denied))
    with pytest.raises(PermissionError, match="publication denied"):
        if kind == "manifest":
            pp._write_manifest(pp.CleaningReport(), [], tmp_path, "acdc")
        else:
            pp.CleaningReport().write(tmp_path)
    assert target.read_bytes() == original
    assert sorted(path.name for path in tmp_path.iterdir()) == [target.name]


def test_lost_and_found_publishes_complete_manifest_once(tmp_path, monkeypatch):
    dataset = tmp_path / "data/external/lost_and_found/leftImg8bit"
    dataset.mkdir(parents=True)
    image = dataset / "image.png"
    Image.fromarray(np.full((8, 8, 3), 90, np.uint8)).save(image)
    before = image.read_bytes()
    published = []
    actual_write = pp._atomic_write_json

    def capture(path, payload):
        if path.name == "lost_and_found.json":
            published.append(json.loads(json.dumps(payload)))
        actual_write(path, payload)

    monkeypatch.setattr(pp, "_atomic_write_json", capture)
    pp.clean({"datasets": ["lost_and_found"], "num_workers": 1}, project_root=tmp_path)
    assert len(published) == 1
    assert published[0]["policy"] == "corrupt_images_only"
    assert published[0]["retained"] == [str(image)]
    assert published[0]["original_files_modified_or_deleted"] is False
    assert image.read_bytes() == before


def test_corrupt_manifest_error_contains_path_and_reclean_action(tmp_path):
    manifest = tmp_path / "damaged.json"
    manifest.write_text('{"invalid": [', encoding="utf-8")
    with pytest.raises(ValueError, match="clean_data.py") as failure:
        pp.load_invalid_entries(manifest, tmp_path)
    assert str(manifest) in str(failure.value)
