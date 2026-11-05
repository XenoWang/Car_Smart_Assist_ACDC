"""Small hand-counted examples for frozen detection supervision accounting."""
from collections import Counter, defaultdict
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest
import torch


path = Path(__file__).resolve().parents[2] / "scripts/profile_detection_assignment.py"
spec = spec_from_file_location("detection_assignment_profile", path)
profile = module_from_spec(spec)
spec.loader.exec_module(profile)


def test_counts_exclude_background_and_distinguish_all_assignment_outcomes():
    # Levels have 2, 1, and 2 anchors. GT 0 has no center, GT 1 loses to GT 2.
    inside = torch.tensor([[0, 0, 0, 0, 0], [1, 0, 0, 0, 0], [1, 1, 1, 1, 0],
                           [0, 0, 0, 0, 1], [0, 0, 1, 0, 0]], dtype=torch.bool)
    pre = torch.tensor([[0, 0, 0, 0, 0], [1, 0, 0, 0, 0], [1, 1, 0, 1, 0],
                        [0, 0, 0, 0, 1], [0, 0, 0, 0, 0]], dtype=torch.bool)
    indices = torch.tensor([2, 2, 0, 2, 3])
    fg = torch.tensor([1, 1, 0, 1, 1], dtype=torch.bool)
    rows = profile.assignment_rows(inside, pre, indices, fg, torch.tensor([.25, .5, 0, 0, 0]), [2, 1, 2])
    assert [r["state"] for r in rows] == ["no_center", "lost_at_conflict", "effective",
                                        "assigned_zero_quality", "no_proposal"]
    assert rows[0]["assigned"] == [0, 0, 0]
    assert rows[1]["proposed"] == [1, 0, 0] and rows[1]["assigned"] == [0, 0, 0]
    assert rows[2]["eligible"] == [2, 1, 1] and rows[2]["assigned"] == [2, 0, 1]
    assert rows[2]["effective"] == [2, 0, 0] and rows[2]["quality_sum"] == 0.75
    assert rows[3]["assigned"] == [0, 0, 1] and rows[3]["quality_sum"] == 0


def test_unknown_foreground_target_is_rejected():
    mask = torch.ones(1, 1, dtype=torch.bool)
    with pytest.raises(ValueError, match="unknown GT"):
        profile.assignment_rows(mask, mask, torch.tensor([1]), torch.tensor([True]), torch.ones(1), [1])


def test_filtered_boxes_remain_in_raw_denominator():
    groups = defaultdict(lambda: defaultdict(Counter))
    frame = {"dataset": "acdc", "condition": "night"}
    target = {"size": "tiny_lt16", "class": "person"}
    profile.add_summary(groups, frame, {**target, "state": "transform_dropped"})
    profile.add_summary(groups, frame, {**target, "state": "effective", "eligible": [1, 0, 0],
                                       "assigned": [1, 0, 0], "effective": [1, 0, 0], "quality_sum": .4})
    for key in ("all", "source/acdc", "condition/night", "class/person", "condition_class/night/person"):
        row = groups[key]["tiny_lt16"]
        assert (row["gt"], row["transform_dropped"], row["effective"], row["effective_P3"]) == (2, 1, 1, 1)
        assert row["quality_sum"] == pytest.approx(.4)
