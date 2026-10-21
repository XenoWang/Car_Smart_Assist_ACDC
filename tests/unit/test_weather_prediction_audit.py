from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/audit_weather_predictions.py"
FIELDS = (
    "sample", "source", "group", "context", "model", "attribute", "label",
    "known", "probability", "threshold", "prediction",
)


def _module():
    spec = importlib.util.spec_from_file_location("weather_prediction_audit", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _row(sample, *, model="m1", source="s1", group="g1", context="c1",
         attribute="fog", label="0", known="1", probability="0.1",
         threshold="0.5", prediction="0"):
    return [sample, source, group, context, model, attribute, label, known,
            probability, threshold, prediction]


def _csv(rows, header=FIELDS):
    lines = [",".join(header)]
    lines.extend(",".join(row) for row in rows)
    return ("\n".join(lines) + "\n").encode("utf-8")


def test_known_unknown_zero_decimal_confusion_counts_and_model_slices():
    module = _module()
    rows = module.read_input(_csv([
        _row("tp", label="1.0", probability="0.8", prediction="1"),
        _row("tn", label="0.0"),
        _row("fp", label="0.0", probability="0.8", prediction="1"),
        _row("fn", label="1.0", probability="0.2", prediction="0"),
        _row("unknown", label="0.0", known="0", probability="0.8", prediction="1"),
        _row("other-model", model="m2", source="s2", group="g2", context="c2",
             label="1.0", probability="0.8", prediction="1"),
    ]))

    summary, meta = module.build_summary(rows, "digest", "predictions.csv")
    overall = summary["overall"]
    assert {key: overall[key] for key in ("row_count", "known", "unknown", "tp", "tn", "fp", "fn")} == {
        "row_count": 6, "known": 5, "unknown": 1, "tp": 2, "tn": 1, "fp": 1, "fn": 1,
    }
    assert overall["precision"] == pytest.approx(2 / 3)
    assert overall["recall"] == pytest.approx(2 / 3)
    assert overall["f1"] == pytest.approx(2 / 3)
    assert summary["by_model"]["m1"]["fp"] == 1
    assert summary["by_model"]["m1"]["fn"] == 1
    assert summary["by_model"]["m2"]["fp"] == 0
    assert summary["by_model_slices"]["m2"]["source"]["s2"]["fog"]["tp"] == 1
    assert summary["by_group"]["g2"]["tp"] == 1
    assert summary["by_context"]["c2"]["known"] == 1
    assert meta["input_row_count"] == 6


def test_zero_denominators_are_null_and_zero_f1_is_not_null():
    module = _module()
    rows = module.read_input(_csv([
        _row("all-negative", label="0", probability="0.1", prediction="0"),
        _row("false-positive", label="0", probability="0.9", prediction="1"),
    ]))
    counts = module.build_summary(rows, "digest", "x")[0]["overall"]
    assert counts["precision"] == 0.0
    assert counts["recall"] is None
    assert counts["f1"] == 0.0

    unknown = module.read_input(_csv([
        _row("u", known="0", label="1", probability="0.5", threshold="0.5", prediction="1"),
    ]))
    unknown_counts = module.build_summary(unknown, "digest", "x")[0]["overall"]
    assert unknown_counts["tp"] == unknown_counts["fp"] == unknown_counts["fn"] == unknown_counts["tn"] == 0
    assert unknown_counts["precision"] is unknown_counts["recall"] is unknown_counts["f1"] is None


@pytest.mark.parametrize("rows,header", [
    ([_row("same"), _row("same", source="different")], FIELDS),
    ([_row("x")], FIELDS + ("sample",)),
    ([_row("x")], FIELDS[:-1]),
    ([[_row("x")[0]]], FIELDS),
    ([_row("",)], FIELDS),
    ([_row("x", attribute="hail")], FIELDS),
    ([_row("x", known="maybe")], FIELDS),
    ([_row("x", known="1", label="2")], FIELDS),
    ([_row("x", probability="NaN")], FIELDS),
    ([_row("x", probability="Infinity")], FIELDS),
    ([_row("x", probability="1.01")], FIELDS),
    ([_row("x", threshold="-0.1")], FIELDS),
    ([_row("x", prediction="2")], FIELDS),
    ([_row("x", probability="0.5", threshold="0.50001", prediction="1")], FIELDS),
])
def test_rejects_malformed_or_conflicting_inputs(rows, header):
    with pytest.raises(ValueError):
        _module().read_input(_csv(rows, header))


def test_float32_boundary_is_preserved_and_noted_only_when_explainable():
    module = _module()
    rows = module.read_input(_csv([
        _row("edge", probability="0.5", threshold="0.50000001", prediction="1"),
    ]))
    assert rows[0]["precision_note"] == "float32_threshold_boundary"
    summary, meta = module.build_summary(rows, "digest", "x")
    review = module.build_review(rows)
    assert summary["overall"]["tp"] == 0
    assert meta["float32_boundary_rows"] == 1
    assert review[0]["precision_note"] == "float32_threshold_boundary"


def test_threshold_equality_predicts_positive():
    rows = _module().read_input(_csv([
        _row("eq", label="1.0", probability="0.5", threshold="0.5", prediction="1"),
    ]))
    assert rows[0]["_issue"] == "correct"


def test_cli_writes_auditable_outputs_and_refuses_overwrite(tmp_path):
    module = _module()
    source = tmp_path / "predictions.csv"
    source.write_bytes(_csv([
        _row("correct"),
        _row("unknown", known="0", label="1", probability="0.8", prediction="1"),
        _row("fp", label="0", probability="0.8", prediction="1"),
        _row("fn", label="1", probability="0.2", prediction="0"),
    ]))
    out = tmp_path / "new-output"
    assert module.main(["--input", str(source), "--output", str(out)]) == 0
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["overall"]["unknown"] == 1
    with (out / "review.csv").open(encoding="utf-8", newline="") as stream:
        review = list(csv.DictReader(stream))
    assert [row["issue"] for row in review] == ["fn", "fp", "unknown", "correct"]
    assert all(not row[field] for row in review for field in (
        "human_label", "human_rain_severity", "human_cooccurrence", "reviewer", "review_note"
    ))

    marker = out / "summary.json"
    before = marker.read_bytes()
    assert module.main(["--input", str(source), "--output", str(out)]) == 2
    assert marker.read_bytes() == before


def test_zip_member_identity_is_preserved_verbatim():
    sample = "dataset.zip::city/frame 001.png"
    rows = _module().read_input(_csv([_row(sample)]))
    assert _module().build_review(rows)[0]["sample"] == sample
