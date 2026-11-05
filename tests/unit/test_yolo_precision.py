"""Precision contracts for native detection, including a hand-computed CUDA DFL box."""
import hashlib
import json
import math
from types import SimpleNamespace

import pytest
import torch

from car_smart_assist.config.detection import inference_half, inference_precision
from car_smart_assist.perception.yolo_postprocess import (
    calibration_identity, calibration_path, read_fp32_reference, validate_calibration,
)
from car_smart_assist.perception.yolo_precision import detection_compute_precision, require_fp32_decoded


def cfg(precision=None):
    args = {"imgsz": 960, "iou": 0.5, "max_det": 100}
    if precision is not None:
        args["precision"] = precision
    return {"checkpoint": "weight.pt", "calibration_file": "reports/calibration.json", "inference": args}


@pytest.mark.parametrize("bad", ["FP16", "", None, False, 16, {}, []])
def test_precision_rejects_implicit_coercion(bad):
    config = cfg()
    config["inference"]["precision"] = bad
    with pytest.raises(ValueError, match="precision"):
        inference_precision(config)


def test_legacy_default_and_explicit_device_contract():
    assert inference_precision(cfg()) == "fp32"
    assert inference_half(cfg(), "cpu") is False
    assert inference_half(cfg("fp16"), "0") is True
    with pytest.raises(ValueError, match="CUDA"):
        inference_half(cfg("fp16"), "cpu")
    with pytest.raises(ValueError, match=r"\.pt"):
        inference_precision({**cfg("fp16"), "checkpoint": "foreign.engine"})


def identities(tmp_path):
    (tmp_path / "weight.pt").write_bytes(b"frozen weight fixture")
    manifest = {"classes": ["car"], "signature": "dataset-fixture"}
    return [calibration_identity(tmp_path, cfg(p), ["car"], manifest=manifest) for p in ("fp32", "fp16")]


def completed(identity):
    return {**identity, "split": "calibration", "passes": ["proposal_sweep", "locked"],
            "confidence_threshold": 0.25, "class_confidence_thresholds": {"car": 0.375},
            "class_threshold_sources": {"car": "calibration_class_f1_proposal"}}


def test_calibration_paths_and_precision_identity_are_separate_in_both_directions(tmp_path):
    a, b = identities(tmp_path)
    assert b == {**a, "inference_precision": "fp16", "decode_precision": "fp32"}
    assert "inference_precision" not in a
    base = tmp_path / "reports/best_class_threshold_v1"
    assert calibration_path(tmp_path, cfg()) == base / "calibration.json"
    assert calibration_path(tmp_path, cfg("fp16")) == base / "fp16/calibration.json"
    assert calibration_path(tmp_path, cfg("fp16"), "new") == tmp_path / "new/best_class_threshold_v1/fp16/calibration.json"
    for payload, identity in ((completed(a), b), (completed(b), a),
                              ({**completed(a), "requested_precision": "fp16"}, a)):
        with pytest.raises(ValueError, match="identity"):
            validate_calibration(payload, identity, ["car"])
    for identity in (a, b):
        assert validate_calibration(completed(identity), identity, ["car"])["class_confidence_thresholds"] == {"car": 0.375}


def test_reference_is_read_only_and_still_binds_weights_data_and_geometry(tmp_path):
    a, b = identities(tmp_path)
    path = tmp_path / "reference.json"
    raw = (json.dumps(completed(a), indent=2) + "\n").encode()
    path.write_bytes(raw)
    reference, source_hash = read_fp32_reference(path, b, ["car"])
    assert reference["class_confidence_thresholds"] == {"car": 0.375}
    assert source_hash == hashlib.sha256(raw).hexdigest()
    for key, different in (("checkpoint_sha256", "other"), ("dataset_signature", "other"),
                           ("imgsz", 640), ("nms_iou", 0.7), ("max_det", 3)):
        with pytest.raises(ValueError, match="identity"):
            read_fp32_reference(path, {**b, key: different}, ["car"])
    assert path.read_bytes() == raw
    path.write_text(json.dumps(completed(b)), encoding="utf-8")
    with pytest.raises(ValueError, match="identity"):
        read_fp32_reference(path, b, ["car"])


@pytest.mark.parametrize("expected,actual", [(True, False), (False, True)])
def test_backend_precision_mismatch_fails_closed(expected, actual):
    with pytest.raises(RuntimeError, match="precision"):
        with detection_compute_precision(SimpleNamespace(fp16=actual), expected):
            pytest.fail("A mismatched backend entered inference")


def test_effective_fp16_rejects_cpu_and_foreign_backend():
    for obj in (SimpleNamespace(fp16=True, pt=True, device=torch.device("cpu")),
                SimpleNamespace(fp16=True, pt=False, device=torch.device("cuda:0"))):
        with pytest.raises(ValueError, match="native PyTorch CUDA"):
            with detection_compute_precision(obj, True):
                pytest.fail("Unsupported backend accepted")


def test_decoded_output_check_rejects_late_upcast_from_half():
    require_fp32_decoded((torch.zeros(1, 13, 1), []))
    with pytest.raises(RuntimeError, match="FP32 decoded"):
        require_fp32_decoded([torch.zeros(1, 13, 1, dtype=torch.float16), []])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for native half precision")
def test_native_half_convolutions_use_float_geometry_and_restore_after_exception():
    from ultralytics.nn.modules.head import Detect

    head = Detect(nc=9, ch=(16,)).cuda().half().eval()
    head.stride = torch.tensor([8.], device="cuda", dtype=torch.float16)
    shell = torch.nn.Module()
    shell.model = torch.nn.Sequential(head)
    backend = SimpleNamespace(fp16=True, pt=True, device=torch.device("cuda:0"), model=shell)
    features = torch.zeros(1, 16, 2, 3, device="cuda", dtype=torch.float16)
    with torch.no_grad(), torch.backends.cudnn.flags(enabled=True, deterministic=True, allow_tf32=False):
        box_conv, class_conv = head.cv2[0][-1], head.cv3[0][-1]
        box_conv.weight.zero_()
        box_conv.bias.zero_()
        box_conv.bias.view(4, 16)[:, 15] = 1.0
        class_conv.weight.zero_()
        class_conv.bias.zero_()
        # Prime the old, half-precision anchor cache at the same shape.
        assert head([features])[0].dtype == torch.float16
        with pytest.raises(LookupError, match="cleanup sentinel"):
            with detection_compute_precision(backend, True):
                output, _ = head([features])
                assert box_conv.weight.dtype == class_conv.weight.dtype == torch.float16
                assert head.dfl.conv.weight.dtype == head.anchors.dtype == head.strides.dtype == torch.float32
                assert output.dtype == torch.float32
                # Fifteen logits of zero and one logit of one, for each of four sides.
                distance = (105 + 15 * math.e) / (15 + math.e)
                expected = torch.tensor([4., 4., 16 * distance, 16 * distance], device="cuda")
                torch.testing.assert_close(output[0, :4, 0], expected, atol=2e-4, rtol=0)
                assert torch.equal(output[0, 4:, 0], torch.full((9,), 0.5, device="cuda"))
                raise LookupError("cleanup sentinel")
        assert head.dfl.conv.weight.dtype == torch.float16
        assert head([features])[0].dtype == torch.float16


def test_benchmark_reference_cannot_replace_a_live_predictor(tmp_path):
    from car_smart_assist.perception.detection_yolo import YoloDetectionPredictor
    adapter = YoloDetectionPredictor.__new__(YoloDetectionPredictor)
    adapter.model = SimpleNamespace(predictor=object())
    with pytest.raises(ValueError, match="before the first prediction"):
        adapter.use_fp32_reference_for_benchmark(tmp_path, tmp_path / "absent.json")


def test_explicit_missing_route_and_conflicting_injection_are_rejected_early(tmp_path):
    from car_smart_assist.inference.pipeline import InferencePipeline
    with pytest.raises(FileNotFoundError, match="Detection configuration"):
        InferencePipeline.from_config({}, tmp_path, detection_config="absent.yaml")
    with pytest.raises(ValueError, match="either"):
        InferencePipeline.from_config({}, tmp_path, detection_config="absent.yaml", predictor=object())
