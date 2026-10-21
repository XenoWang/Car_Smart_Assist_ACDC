#!/usr/bin/env python3
"""整理天气预测的分组分数和待人工复核样本。"""

import argparse
import csv
import hashlib
import io
import json
import math
import sys
import struct
from pathlib import Path

COLUMNS = ["sample", "source", "group", "context", "model", "attribute",
           "label", "known", "probability", "threshold", "prediction"]
REQUIRED = tuple(COLUMNS)
ATTRIBUTES = ("fog", "rain", "snow", "night")
REVIEW_COLUMNS = COLUMNS + ["issue", "precision_note", "human_label", "human_rain_severity",
                            "human_cooccurrence", "reviewer", "review_note"]
GROUP_FIELDS = ("source", "group", "context", "model", "attribute")
IDENTITY_FIELDS = ("sample", "source", "group", "context", "model", "attribute")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="读取已有天气预测CSV进行审计，不重新推理")
    p.add_argument("--input", required=True, help="已有预测CSV路径")
    p.add_argument("--output", required=True, help="新输出目录，不覆盖已有结果")
    return p.parse_args(argv)


def is_finite01(text):
    try:
        v = float(text)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v) or v < 0.0 or v > 1.0:
        return None
    return v


def is_label01(text):
    v = is_finite01(text)
    if v is None or v not in (0.0, 1.0):
        return None
    return v


def read_input(data):
    rows = []
    seen = set()
    text = data.decode("utf-8-sig")
    reader = csv.reader(io.StringIO(text, newline=""))
    try:
        header = next(reader)
    except StopIteration:
        raise ValueError("input CSV is empty")
    if len(header) != len(set(header)):
        raise ValueError("duplicate column names in header")
    missing = [c for c in REQUIRED if c not in header]
    if missing:
        raise ValueError("missing required columns: %r" % (missing,))
    index = {name: i for i, name in enumerate(header)}
    width = len(header)
    for lineno, raw in enumerate(reader, start=2):
        if len(raw) != width:
            raise ValueError("line %d: expected %d columns, got %d"
                             % (lineno, width, len(raw)))
        row = {name: raw[index[name]] for name in REQUIRED}
        if not all(row[c].strip() for c in IDENTITY_FIELDS):
            raise ValueError("line %d: empty identity field" % lineno)
        if row["attribute"] not in ATTRIBUTES:
            raise ValueError("line %d: bad attribute %r"
                             % (lineno, row["attribute"]))
        key = (row["sample"], row["model"], row["attribute"])
        if key in seen:
            raise ValueError("line %d: duplicate %r" % (lineno, key))
        seen.add(key)
        if row["known"] not in ("0", "1"):
            raise ValueError("line %d: known must be 0/1" % lineno)
        known = row["known"] == "1"
        if known and is_label01(row["label"]) is None:
            raise ValueError("line %d: known=1 label must be 0/1" % lineno)
        prob = is_finite01(row["probability"])
        thr = is_finite01(row["threshold"])
        if prob is None or thr is None:
            raise ValueError("line %d: probability/threshold not finite 0..1"
                             % lineno)
        if row["prediction"] not in ("0", "1"):
            raise ValueError("line %d: prediction must be 0/1" % lineno)
        pred = row["prediction"] == "1"
        row["precision_note"] = ""
        if pred != (prob >= thr):
            # 训练导出使用numpy float32标量比较，阈值也会转换成float32。
            f32_prob = struct.unpack("f", struct.pack("f", prob))[0]
            f32_thr = struct.unpack("f", struct.pack("f", thr))[0]
            if prob != f32_prob or pred != (f32_prob >= f32_thr):
                raise ValueError("line %d: prediction inconsistent with "
                                 "probability>=threshold" % lineno)
            row["precision_note"] = "float32_threshold_boundary"
        row["_known"] = known
        row["_label"] = is_label01(row["label"]) == 1.0 if known else None
        row["_pred"] = pred
        if not known:
            row["_issue"] = "unknown"
        elif row["_label"] and not pred:
            row["_issue"] = "fn"
        elif (not row["_label"]) and pred:
            row["_issue"] = "fp"
        else:
            row["_issue"] = "correct"
        rows.append(row)
    if not rows:
        raise ValueError("input CSV has no data rows")
    return rows


def empty_counts():
    return {"row_count": 0, "known": 0, "unknown": 0,
            "tp": 0, "fp": 0, "fn": 0, "tn": 0}


def add_row(counts, row):
    counts["row_count"] += 1
    if not row["_known"]:
        counts["unknown"] += 1
        return
    counts["known"] += 1
    label, pred = row["_label"], row["_pred"]
    if pred and label:
        counts["tp"] += 1
    elif pred and not label:
        counts["fp"] += 1
    elif (not pred) and label:
        counts["fn"] += 1
    else:
        counts["tn"] += 1


def finalize(counts):
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]

    def div(a, b):
        return None if b == 0 else a / b

    counts["precision"] = div(tp, tp + fp)
    counts["recall"] = div(tp, tp + fn)
    denom = 2 * tp + fp + fn
    if denom == 0:
        counts["f1"] = None
    elif tp == 0:
        counts["f1"] = 0.0
    else:
        counts["f1"] = 2 * tp / denom
    return counts


def build_summary(rows, digest, input_path):
    overall = empty_counts()
    groups = {name: {} for name in GROUP_FIELDS}
    for row in rows:
        add_row(overall, row)
        for name in GROUP_FIELDS:
            counts = groups[name].setdefault(row[name], empty_counts())
            add_row(counts, row)
    summary = {
        "overall": finalize(overall),
        "unique_sample_count": len({r["sample"] for r in rows}),
        "sample_model_count": len({(r["sample"], r["model"]) for r in rows}),
    }
    for name in GROUP_FIELDS:
        summary["by_" + name] = {
            k: finalize(v) for k, v in sorted(groups[name].items())
        }
    for axis in ("source", "group", "context", "model"):
        buckets = {}
        for row in rows:
            counts = buckets.setdefault(row[axis], {}).setdefault(row["attribute"], empty_counts())
            add_row(counts, row)
        summary["by_" + axis + "_attribute"] = {
            key: {attr: finalize(count) for attr, count in sorted(values.items())}
            for key, values in sorted(buckets.items())
        }
    per_model_slices = {}
    for row in rows:
        model_slices = per_model_slices.setdefault(row["model"], {})
        for axis in ("source", "group", "context"):
            counts = model_slices.setdefault(axis, {}).setdefault(row[axis], {}).setdefault(
                row["attribute"], empty_counts())
            add_row(counts, row)
    for axes in per_model_slices.values():
        for slices in axes.values():
            for attributes in slices.values():
                for counts in attributes.values():
                    finalize(counts)
    summary["by_model_slices"] = per_model_slices
    meta = {
        "input_path": str(input_path),
        "input_sha256": digest,
        "input_row_count": len(rows),
        "float32_boundary_rows": sum(bool(row["precision_note"]) for row in rows),
        "fields": COLUMNS,
        "attributes": list(ATTRIBUTES),
        "test_definitions": [
            "这是基于已有标签的开发审计；模型输出不充当人工真值。",
            "known=0是未知标签，不当作负样本。",
            "known=1标签只接受0或1，也兼容0.0／1.0文本。",
            "统计单位是属性行，汇总可以包含多个模型；"
            "不是独立图片准确率。",
            "另列独立图片数和图片×模型数，"
            "避免混淆分母。",
        ],
    }
    return summary, meta


def issue_order(issue):
    return {"fn": 0, "fp": 1, "unknown": 2, "correct": 3}[issue]


def build_review(rows):
    ordered = sorted(rows, key=lambda r: (
        issue_order(r["_issue"]), r["sample"], r["model"], r["attribute"]))
    out = []
    for r in ordered:
        rec = {c: r[c] for c in COLUMNS}
        rec["issue"] = r["_issue"]
        rec["precision_note"] = r["precision_note"]
        for c in ("human_label", "human_rain_severity", "human_cooccurrence",
                  "reviewer", "review_note"):
            rec[c] = ""
        out.append(rec)
    return out


def write_outputs(output, summary, meta, review):
    output.mkdir(parents=True, exist_ok=False)
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    (output / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    with open(output / "review.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=REVIEW_COLUMNS)
        w.writeheader()
        w.writerows(review)


def main(argv=None):
    args = parse_args(argv)
    input_path = Path(args.input)
    output = Path(args.output)
    if not input_path.is_file():
        print("error: input not found: %s" % input_path, file=sys.stderr)
        return 2
    if output.exists():
        print("error: output already exists: %s" % output, file=sys.stderr)
        return 2
    try:
        data = input_path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        rows = read_input(data)
        summary, meta = build_summary(rows, digest, input_path)
        review = build_review(rows)
        write_outputs(output, summary, meta, review)
    except (OSError, ValueError, csv.Error) as exc:
        print("error: " + str(exc), file=sys.stderr)
        return 2
    print("wrote %s" % output)
    return 0


if __name__ == "__main__":
    sys.exit(main())