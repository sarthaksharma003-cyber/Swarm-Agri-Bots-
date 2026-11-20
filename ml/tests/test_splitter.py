"""Unit tests for reproducible dataset splitting and duplicate leakage prevention."""

import pytest
from pathlib import Path
import sys

# Ensure ml/src is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dataset.splitter import (
    check_existing_splits,
    create_reproducible_splits,
    verify_split_integrity,
)


def test_check_existing_splits_none(tmp_path):
    assert check_existing_splits(tmp_path) is None


def test_reproducible_splits_seed_consistency():
    stems = [f"img_{i:04d}" for i in range(100)]
    classes = {s: {0} if i % 2 == 0 else {1} for i, s in enumerate(stems)}

    split1 = create_reproducible_splits(stems, classes, seed=42)
    split2 = create_reproducible_splits(stems, classes, seed=42)
    split3 = create_reproducible_splits(stems, classes, seed=999)

    assert split1 == split2
    assert split1 != split3


def test_reproducible_splits_disjoint_and_complete():
    stems = [f"img_{i:04d}" for i in range(100)]
    classes = {s: {0} if i % 2 == 0 else {1} for i, s in enumerate(stems)}

    splits = create_reproducible_splits(stems, classes, seed=42)
    train_set = set(splits["train"])
    val_set = set(splits["val"])
    test_set = set(splits["test"])

    # Disjoint
    assert len(train_set & val_set) == 0
    assert len(train_set & test_set) == 0
    assert len(val_set & test_set) == 0

    # Complete
    assert train_set | val_set | test_set == set(stems)

    # Ratios ~ 80/10/10
    assert 75 <= len(train_set) <= 85
    assert 8 <= len(val_set) <= 12
    assert 8 <= len(test_set) <= 12


def test_duplicate_cluster_leakage_prevention():
    # 20 stems, with two duplicate pairs:
    # cluster 1: (img_0001, img_0002)
    # cluster 2: (img_0003, img_0004)
    stems = [f"img_{i:04d}" for i in range(20)]
    classes = {s: {0} for s in stems}
    clusters = [
        ["img_0001", "img_0002"],
        ["img_0003", "img_0004"],
    ]

    splits = create_reproducible_splits(
        stems=stems,
        stem_to_classes=classes,
        clusters=clusters,
        seed=42
    )

    # Verify that img_0001 and img_0002 are in the EXACT SAME split
    for split_name in ["train", "val", "test"]:
        s_set = set(splits[split_name])
        in_0001 = "img_0001" in s_set
        in_0002 = "img_0002" in s_set
        assert in_0001 == in_0002, f"Leakage detected in cluster 1 for {split_name}"

        in_0003 = "img_0003" in s_set
        in_0004 = "img_0004" in s_set
        assert in_0003 == in_0004, f"Leakage detected in cluster 2 for {split_name}"


def test_verify_split_integrity_catches_leakage():
    clusters = [["img_001", "img_002"]]
    classes = {"img_001": {0}, "img_002": {0}}

    # Artificially leak duplicate into two different splits
    leaked_splits = {
        "train": ["img_001"],
        "val": ["img_002"],
        "test": []
    }

    report = verify_split_integrity(leaked_splits, clusters, classes)
    assert report.is_valid is False
    assert report.leakage_detected is True
    assert len(report.leaked_duplicate_groups) == 1


def test_verify_split_integrity_catches_overlap():
    clusters = [["img_001"]]
    classes = {"img_001": {0}}

    # Artificially overlap stems
    overlapping_splits = {
        "train": ["img_001"],
        "val": ["img_001"],
        "test": []
    }

    report = verify_split_integrity(overlapping_splits, clusters, classes)
    assert report.is_valid is False
    assert report.overlap_detected is True
    assert "img_001" in report.overlap_stems
