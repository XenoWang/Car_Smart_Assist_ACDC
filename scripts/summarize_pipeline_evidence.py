"""把已有pipeline结果整理为证据统计和空白人工复核表。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

FRAME_FIELDS = (
    "frame_index", "input_name", "input_mode", "phase", "iteration", "input_index",
    "batch_index", "visibility_level", "visibility_raw", "takeover",
    "risk_level", "advisory_source", "detection_status", "object_count", "blocked",
    "skipped", "policy_reasons", "human_visibility_usable", "human_missed_object",
    "reviewer", "review_note",
)
OBJECT_FIELDS = (
    "frame_index", "input_name", "object_index", "category", "direction_raw",
    "distance_m_raw", "distance_uncertainty_m_raw", "bbox", "confidence",
    "valid_distance", "valid_direction", "human_distance_m", "human_direction",
    "reviewer", "review_note",
)


def reject_constant(token):
    raise ValueError(f"JSON不允许非有限常量：{token}")


def check_finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("JSON包含非有限数值")
    if isinstance(value, dict):
        for item in value.values():
            check_finite(item)
    elif isinstance(value, list):
        for item in value:
            check_finite(item)


def raw(value):
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, allow_nan=False)
    return value


def optional_object(value, field):
    if value is not None and not isinstance(value, dict):
        raise ValueError(f"{field}必须是对象或null")
    return value if value is not None else {}


def boolean(value, field):
    if value is None:
        return "unknown"
    if type(value) is not bool:
        raise ValueError(f"{field}必须是bool或null")
    return "true" if value else "false"


def valid_distance(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def build_report(data):
    payload = json.loads(data.decode("utf-8-sig"), parse_constant=reject_constant)
    check_finite(payload)
    frames = payload.get("frames") if isinstance(payload, dict) else payload
    if not isinstance(frames, list):
        raise ValueError('输入必须是帧数组或{"frames": [...]}')
    frame_rows, object_rows, diagnostics = [], [], []
    takeover, detection, visibility = Counter(), Counter(), Counter()
    occurrence, code_frames = Counter(), Counter()
    total_objects = missing_distance = missing_direction = missing_both = 0
    gap_frames = takeover_gap_frames = empty_frames = objects_unknown_frames = 0
    codes_known = codes_partial = codes_unknown = 0
    for index, frame in enumerate(frames):
        if not isinstance(frame, dict):
            raise ValueError(f"帧[{index}]不是对象")
        name = frame.get("input_name", frame.get("name", f"frame_{index}"))
        if not isinstance(name, str):
            raise ValueError(f"帧[{index}]的名称必须是字符串")
        vis = optional_object(frame.get("visibility"), "visibility")
        perception = optional_object(frame.get("perception"), "perception")
        advisory = optional_object(frame.get("advisory"), "advisory")
        skipped = optional_object(frame.get("skipped"), "skipped")
        level = vis.get("level")
        if not isinstance(level, str) or level not in {"visible", "degraded", "blind"}:
            diagnostics.append(f"帧[{index}]能见度档位未知：{level!r}")
            level = "unknown"
        takes = boolean(advisory.get("should_takeover"), "should_takeover")
        available = boolean(perception.get("object_detection_available"), "object_detection_available")
        blocked = boolean(frame.get("blocked"), "blocked")
        takeover[takes] += 1
        detection[available] += 1
        visibility[level] += 1
        objects = perception.get("objects")
        if objects is not None and not isinstance(objects, list):
            raise ValueError(f"帧[{index}]的objects必须是数组或null")
        if objects is None:
            objects_unknown_frames += 1
            diagnostics.append(f"帧[{index}]没有目标列表，不能当成检测成功空框")
        elif available == "true" and not objects:
            empty_frames += 1
        gap = False
        for object_index, obj in enumerate(objects or []):
            if not isinstance(obj, dict):
                raise ValueError(f"帧[{index}]目标[{object_index}]不是对象")
            distance = obj.get("distance_m")
            direction = obj.get("direction")
            has_distance = valid_distance(distance)
            has_direction = isinstance(direction, str) and direction in {"leading", "oncoming"}
            total_objects += 1
            missing_distance += not has_distance
            missing_direction += not has_direction
            missing_both += not has_distance and not has_direction
            gap |= not has_distance or not has_direction
            row = dict.fromkeys(OBJECT_FIELDS, "")
            row.update(frame_index=index, input_name=name, object_index=object_index,
                       category=raw(obj.get("category")), direction_raw=raw(direction),
                       distance_m_raw=raw(distance),
                       distance_uncertainty_m_raw=raw(obj.get("distance_uncertainty_m")),
                       bbox=raw(obj.get("bbox")), confidence=raw(obj.get("confidence")),
                       valid_distance=has_distance, valid_direction=has_direction)
            object_rows.append(row)
        gap_frames += gap
        takeover_gap_frames += gap and takes == "true"
        details = optional_object(advisory.get("policy_details"), "policy_details")
        risk = optional_object(details.get("risk"), "policy_details.risk")
        triggered = risk.get("triggered")
        if triggered is None:
            codes_unknown += 1
        elif not isinstance(triggered, list):
            diagnostics.append(f"帧[{index}]risk.triggered不是数组，未解析代码")
            codes_unknown += 1
        else:
            # 只解析该字段的代码，不把自然语言reasons猜成风险类型。
            codes = [code for code in triggered if isinstance(code, str) and code.strip()]
            if len(codes) != len(triggered):
                diagnostics.append(f"帧[{index}]risk.triggered含非法项，只统计可解析部分")
                codes_partial += 1
            else:
                codes_known += 1
            occurrence.update(codes)
            code_frames.update(set(codes))
        row = dict.fromkeys(FRAME_FIELDS, "")
        row.update(frame_index=index, input_name=name, visibility_level=level,
                   visibility_raw=raw(vis.get("level")), takeover=takes,
                   risk_level=raw(advisory.get("risk_level")),
                   advisory_source=raw(advisory.get("source")), detection_status=available,
                   object_count=len(objects) if objects is not None else None,
                   blocked=blocked, skipped=raw(skipped),
                   policy_reasons=raw(details.get("reasons")))
        for key in ("input_mode", "phase", "iteration", "input_index", "batch_index"):
            row[key] = raw(frame.get(key))
        frame_rows.append(row)
    total = len(frames)
    summary = {
        "input_sha256": hashlib.sha256(data).hexdigest(),
        "frames": {"total": total,
                   "takeover": {key: takeover[key] for key in ("true", "false", "unknown")},
                   "detection_available": {key: detection[key] for key in ("true", "false", "unknown")},
                   "visibility": {key: visibility[key] for key in ("visible", "degraded", "blind", "unknown")},
                   "successful_empty": empty_frames, "objects_unknown": objects_unknown_frames},
        "objects": {"total": total_objects, "missing_distance": missing_distance,
                    "missing_direction": missing_direction, "missing_both": missing_both,
                    "missing_distance_ratio": ratio(missing_distance, total_objects),
                    "missing_direction_ratio": ratio(missing_direction, total_objects),
                    "missing_both_ratio": ratio(missing_both, total_objects)},
        "gap_frames": {"any_gap": gap_frames, "frame_denominator": total,
                       "ratio_of_all_frames": ratio(gap_frames, total),
                       "takeover_with_any_gap": takeover_gap_frames,
                       "takeover_denominator": takeover["true"],
                       "ratio_of_takeover_frames": ratio(takeover_gap_frames, takeover["true"])},
        "risk_codes": [{"code": code, "occurrence_count": occurrence[code],
                        "frame_count": code_frames[code], "frame_denominator": total,
                        "frame_ratio": ratio(code_frames[code], total)} for code in sorted(occurrence)],
        "code_availability_frames": {"complete": codes_known, "partial": codes_partial, "unknown": codes_unknown},
        "diagnostics": diagnostics,
        "scope": ["缺证据与接管共现不代表因果；不同原因及帧可重叠，不能相加当互斥占比。",
                  "目标统计只含已输出目标；无列表不是零目标，不代表没有漏检。",
                  "gap比例仅为已观察缺口帧占全帧；无目标列表的帧仍未知。",
                  "帧数按输入记录计数；同图重复量测仍是多条记录，复核表保留模式／阶段／轮次。",
                  "risk.triggered是代码列表；reasons是原文，不由文本猜代码。",
                  "人工栏留空；本工具不推测米数、来向或道路安全真值，不改变规则。"],
    }
    return summary, frame_rows, object_rows


def write_csv(path, fields, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description="整理已有pipeline输出，不重新推理")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.output.exists():
            raise ValueError(f"输出目录已存在，拒绝覆盖：{args.output}")
        summary, frame_rows, object_rows = build_report(args.input.read_bytes())
        summary["input_path"] = str(args.input.resolve())
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        write_csv(args.output / "frames_review.csv", FRAME_FIELDS, frame_rows)
        write_csv(args.output / "objects_review.csv", OBJECT_FIELDS, object_rows)
    except (OSError, ValueError, csv.Error) as exc:
        print(f"证据统计失败：{exc}", file=sys.stderr)
        return 2
    print(f"结果已保存：{args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
