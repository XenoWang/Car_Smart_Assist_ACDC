from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/summarize_pipeline_evidence.py"


def _module():
    spec = importlib.util.spec_from_file_location("pipeline_evidence_audit", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _strict_json(text):
    def reject_constant(token):
        raise ValueError(f"non-finite JSON output: {token}")

    return json.loads(text, parse_constant=reject_constant)


def _frame(name, *, level="visible", takeover=False, available=True, objects=None,
           triggered=None, reasons=None, **extra):
    return {
        "input_name": name,
        "visibility": None if level is None else {"level": level},
        "perception": None if available is None else {
            "object_detection_available": available,
            **({} if objects is None else {"objects": objects}),
        },
        "advisory": {
            "should_takeover": takeover,
            "policy_details": {
                "risk": {} if triggered is None else {"triggered": triggered},
                "reasons": reasons,
            },
        },
        **extra,
    }


def test_null_missing_and_empty_perception_are_distinguished():
    module = _module()
    payload = [
        _frame("null", available=None),
        {"name": "missing"},
        _frame("empty", objects=[]),
        _frame("blind", level="blind", takeover=True, available=False),
        _frame("degraded", level="degraded", available=False, objects=[]),
    ]
    summary, frames, _ = module.build_report(json.dumps(payload).encode("utf-8"))
    assert summary["frames"]["total"] == 5
    assert summary["frames"]["successful_empty"] == 1
    assert summary["frames"]["objects_unknown"] == 3
    assert summary["frames"]["visibility"] == {
        "visible": 2, "degraded": 1, "blind": 1, "unknown": 1,
    }
    assert summary["frames"]["takeover"] == {"true": 1, "false": 3, "unknown": 1}
    assert [row["frame_index"] for row in frames] == list(range(5))


def test_risk_occurrences_deduplicate_frames_and_preserve_reasons():
    module = _module()
    frames = [
        _frame("a", triggered=["distance_low", "distance_low", "weather_bad"],
               reasons=["原始说明：距离过近", "保留顺序"]),
        _frame("b", triggered=["distance_low"]),
        _frame("c", triggered=[]),
        _frame("d"),
        _frame("e", triggered=["valid", 3]),
    ]
    summary, frame_rows, _ = module.build_report(json.dumps({"frames": frames}, ensure_ascii=False).encode())
    records = {item["code"]: item for item in summary["risk_codes"]}
    assert records["distance_low"]["occurrence_count"] == 3
    assert records["distance_low"]["frame_count"] == 2
    assert records["weather_bad"]["occurrence_count"] == 1
    assert summary["code_availability_frames"] == {"complete": 3, "partial": 1, "unknown": 1}
    assert frame_rows[0]["policy_reasons"] == json.dumps(
        ["原始说明：距离过近", "保留顺序"], ensure_ascii=False
    )
    assert "不能相加当互斥占比" in " ".join(summary["scope"])


@pytest.mark.parametrize("distance", [None, True, "2.0", 0, -1, 10 ** 1000])
def test_invalid_distances_remain_raw_but_are_not_valid(distance):
    module = _module()
    data = json.dumps([_frame("x", objects=[{"distance_m": distance, "direction": "leading"}])],
                      allow_nan=True).encode()
    summary, _, objects = module.build_report(data)
    assert summary["objects"]["total"] == 1
    assert summary["objects"]["missing_distance"] == 1
    assert summary["objects"]["missing_direction"] == 0
    assert objects[0]["valid_distance"] is False


@pytest.mark.parametrize("direction", [None, "", "Leading", "left", 1, True])
def test_invalid_directions_are_retained_as_raw_diagnostics(direction):
    module = _module()
    data = json.dumps([_frame("x", objects=[{"distance_m": 4.0, "direction": direction}])]).encode()
    summary, _, objects = module.build_report(data)
    assert summary["objects"]["missing_direction"] == 1
    assert objects[0]["valid_direction"] is False


@pytest.mark.parametrize("data", [
    b"NaN", b"Infinity", b"-Infinity", b"[{\"x\":1e999}]", b"{\"frames\":[null]}",
    b"[{\"perception\":{\"objects\":[{\"distance_m\":Infinity}]}}]",
    b"{\"frames\":null}", b"{}",
])
def test_rejects_nonfinite_json_and_malformed_frame_shapes(data):
    with pytest.raises((ValueError, TypeError)):
        _module().build_report(data)


def test_duplicate_names_keep_frame_mode_phase_identity_and_empty_outputs(tmp_path):
    module = _module()
    frame = _frame("same.png", objects=[], triggered=[], reasons=[])
    frame.update(input_mode="paths", phase="steady", iteration=0)
    second = dict(frame, input_mode="arrays", phase="steady", iteration=0)
    payload = json.dumps({"frames": [frame, second]}).encode()
    summary, frame_rows, object_rows = module.build_report(payload)
    assert [row["frame_index"] for row in frame_rows] == [0, 1]
    assert [row["input_mode"] for row in frame_rows] == ["paths", "arrays"]
    assert [row["phase"] for row in frame_rows] == ["steady", "steady"]
    assert object_rows == []
    assert summary["objects"]["missing_both_ratio"] is None

    source = tmp_path / "frames.json"
    source.write_bytes(payload)
    output = tmp_path / "evidence"
    assert module.main(["--input", str(source), "--output", str(output)]) == 0
    with (output / "objects_review.csv").open(encoding="utf-8", newline="") as stream:
        assert list(csv.DictReader(stream)) == []
    assert (output / "objects_review.csv").read_text(encoding="utf-8").startswith("frame_index,")
    assert _strict_json((output / "summary.json").read_text(encoding="utf-8"))["frames"]["total"] == 2


def test_cli_refuses_to_overwrite_existing_output(tmp_path):
    module = _module()
    source = tmp_path / "frames.json"
    source.write_text("[]", encoding="utf-8")
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("evidence", encoding="utf-8")
    assert module.main(["--input", str(source), "--output", str(output)]) == 2
    assert marker.read_text(encoding="utf-8") == "evidence"
