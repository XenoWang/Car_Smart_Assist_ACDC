from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from car_smart_assist.perception.visibility.dataset import sequence_of, split_ref_indices


ROOT = Path(__file__).resolve().parents[2]
GOLDEN_PATH = ROOT / "tests/fixtures/visibility_group_split_legacy_golden.json"
SPLIT_NAMES = ("train", "val", "calibration", "test")
SPLIT_RATIOS = (0.80, 0.05, 0.05, 0.10)

REAL_FAMILY_SIZES = (
    ("gopro:0122", 152),
    ("gopro:0176", 354),
    ("gopro:0351", 149),
    ("gopro:0355", 10),
    ("gopro:0356", 229),
    ("gopro:0364", 151),
    ("gopro:0376", 203),
    ("gopro:0397", 104),
    ("gopro:0400", 299),
    ("gopro:0402", 269),
    ("gopro:0475", 243),
    ("gopro:0476", 142),
    ("gopro:0477", 244),
    ("gopro:0478", 335),
    ("gopro:0479", 29),
    ("gopro:0480", 7),
    ("gopro:0494", 73),
    ("gopro:0496", 59),
    ("gopro:0571", 75),
    ("gopro:0572", 119),
    ("gopro:0573", 238),
    ("gopro:0594", 160),
    ("gopro:0604", 54),
    ("gopro:0605", 3),
    ("gopro:0606", 165),
    ("gopro:0607", 140),
)


def _groups_from_sizes(sizes: tuple[tuple[str, int], ...]) -> list[str]:
    return [name for name, size in sizes for _ in range(size)]


def _split(groups: list[str], seed: int):
    return split_ref_indices(
        len(groups), *SPLIT_RATIOS, seed=seed, mode="fixed", groups=groups
    )


def _assert_partition(parts, groups: list[str], *, seed: int) -> None:
    assert len(parts) == 4, f"seed={seed} must produce four partitions"
    for name, part in zip(SPLIT_NAMES, parts, strict=True):
        assert len(part) > 0, f"seed={seed} produced empty {name}"
        assert np.array_equal(part, np.sort(part)), f"seed={seed} {name} indices are unsorted"

    flattened = np.concatenate(parts)
    assert len(flattened) == len(groups), f"seed={seed} lost or duplicated indices"
    assert set(flattened.tolist()) == set(range(len(groups))), f"seed={seed} is incomplete"
    assert len(set(flattened.tolist())) == len(groups), f"seed={seed} has overlapping indices"

    owners: dict[str, set[int]] = {}
    for split_index, part in enumerate(parts):
        for index in part:
            owners.setdefault(groups[int(index)], set()).add(split_index)
    assert set(owners) == set(groups), f"seed={seed} omitted a group"
    assert all(len(owner) == 1 for owner in owners.values()), f"seed={seed} split a group"


def _group_owners(parts, groups: list[str]) -> dict[str, str]:
    owners: dict[str, str] = {}
    for name, part in zip(SPLIT_NAMES, parts, strict=True):
        for index in part:
            group = groups[int(index)]
            assert group not in owners or owners[group] == name
            owners[group] = name
    return owners


def test_nonempty_legacy_group_split_matches_captured_index_golden():
    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    inputs = golden["inputs"]
    groups = [f"seq{i // inputs['samples_per_group']}" for i in range(inputs["sample_count"])]
    parts = split_ref_indices(
        inputs["sample_count"],
        inputs["train_ratio"],
        inputs["val_ratio"],
        inputs["calib_ratio"],
        inputs["test_ratio"],
        seed=inputs["seed"],
        mode=inputs["mode"],
        groups=groups,
    )

    assert {name: len(part) for name, part in zip(SPLIT_NAMES, parts, strict=True)} == golden[
        "split_counts"
    ]
    for name, part in zip(SPLIT_NAMES, parts, strict=True):
        assert part.tolist() == golden["indices"][name]


def test_original_six_row_four_group_alias_fixture_produces_nonempty_splits():
    groups = [
        sequence_of(Path("GOPR0475") / "a.png"),
        sequence_of(Path("GP010475") / "b.png"),
        sequence_of(Path("GP020475") / "c.png"),
        "sequence:other_a",
        "sequence:other_b",
        "sequence:other_c",
    ]

    parts = split_ref_indices(
        len(groups), 0.5, 0.16, 0.17, 0.17, seed=42, groups=groups
    )

    _assert_partition(parts, groups, seed=42)


@pytest.mark.parametrize("seed", [0, 1, 42, 255])
def test_extremely_imbalanced_four_group_case_remains_complete(seed):
    groups = ["large"] * 997 + ["small-a", "small-b", "small-c"]

    parts = _split(groups, seed=seed)

    _assert_partition(parts, groups, seed=seed)


def test_real_26_family_sizes_fill_calibration_by_moving_one_whole_family():
    groups = _groups_from_sizes(REAL_FAMILY_SIZES)
    parts = _split(groups, seed=42)

    _assert_partition(parts, groups, seed=42)
    assert [len(part) for part in parts] == [3207, 335, 165, 299]
    expected_owners = {
        **{
            name: "train"
            for name, _size in REAL_FAMILY_SIZES
            if name not in {"gopro:0478", "gopro:0400", "gopro:0606"}
        },
        "gopro:0478": "val",
        "gopro:0606": "calibration",
        "gopro:0400": "test",
    }
    assert _group_owners(parts, groups) == expected_owners


@pytest.mark.parametrize("seed", [0, 1, 42, 255])
def test_equal_sse_ties_follow_global_seed_permutation_rank(seed):
    groups = [f"seq{index}" for index in range(4) for _ in range(5)]
    permutation = np.random.default_rng(seed).permutation(4)
    ranked_groups = [f"seq{int(index)}" for index in permutation]

    parts = _split(groups, seed=seed)

    _assert_partition(parts, groups, seed=seed)
    assert _group_owners(parts, groups) == {
        ranked_groups[0]: "val",
        ranked_groups[1]: "calibration",
        ranked_groups[2]: "test",
        ranked_groups[3]: "train",
    }


def test_seed_range_zero_through_255_preserves_partition_properties_and_repeats():
    layouts = (
        tuple((f"four-{index}", size) for index, size in enumerate((2, 3, 4, 5))),
        tuple((f"five-{index}", size) for index, size in enumerate((1, 2, 5, 8, 13))),
        REAL_FAMILY_SIZES,
        tuple((f"many-{index:02d}", 1 + index % 7) for index in range(64)),
    )

    for sizes in layouts:
        groups = _groups_from_sizes(sizes)
        for seed in range(256):
            first = _split(groups, seed)
            repeated = _split(groups, seed)
            _assert_partition(first, groups, seed=seed)
            assert all(
                np.array_equal(left, right)
                for left, right in zip(first, repeated, strict=True)
            ), f"seed={seed} was not reproducible"


def test_split_is_independent_of_group_insertion_order():
    groups = _groups_from_sizes(REAL_FAMILY_SIZES)
    reversed_groups = list(reversed(groups))

    first = _split(groups, seed=42)
    second = _split(reversed_groups, seed=42)

    _assert_partition(first, groups, seed=42)
    _assert_partition(second, reversed_groups, seed=42)
    assert _group_owners(first, groups) == _group_owners(second, reversed_groups)
