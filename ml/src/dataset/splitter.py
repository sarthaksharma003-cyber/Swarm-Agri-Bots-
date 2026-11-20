"""Dataset splitting module with duplicate cluster isolation and stratification."""

import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from .config import DEFAULT_SEED, SPLIT_RATIOS


@dataclass
class SplitVerificationReport:
    """Verification results for generated dataset splits."""
    is_valid: bool
    train_count: int
    val_count: int
    test_count: int
    total_count: int
    train_ratio: float
    val_ratio: float
    test_ratio: float
    leakage_detected: bool
    overlap_detected: bool
    leaked_duplicate_groups: List[Dict[str, any]] = field(default_factory=list)
    overlap_stems: List[str] = field(default_factory=list)
    class_distribution_per_split: Dict[str, Dict[int, int]] = field(default_factory=dict)


def check_existing_splits(source_dir: Path) -> Optional[Dict[str, List[Path]]]:
    """Check if the source dataset already contains legitimate predefined splits."""
    if not source_dir.exists():
        return None

    # Check for train / val / test subdirectories
    subdirs = [p.name.lower() for p in source_dir.iterdir() if p.is_dir()]
    if "train" in subdirs and ("val" in subdirs or "validation" in subdirs):
        splits: Dict[str, List[Path]] = defaultdict(list)
        for s in ["train", "val", "validation", "test"]:
            s_dir = source_dir / s
            if s_dir.exists() and s_dir.is_dir():
                key = "val" if s == "validation" else s
                splits[key].extend([p for p in s_dir.rglob("*.jpeg")])
        if len(splits.get("train", [])) > 0:
            return dict(splits)

    # Check for split text files (train.txt, val.txt, test.txt)
    split_files = {
        "train": source_dir / "train.txt",
        "val": source_dir / "val.txt",
        "test": source_dir / "test.txt",
    }
    if split_files["train"].exists() and split_files["val"].exists():
        splits = {}
        for k, p in split_files.items():
            if p.exists():
                lines = [line.strip() for line in p.read_text().splitlines() if line.strip()]
                splits[k] = [Path(l) for l in lines]
        return splits

    return None


def determine_cluster_stratum(cluster: List[str], stem_to_classes: Dict[str, Set[int]]) -> str:
    """Determine the stratification label for a cluster of images."""
    classes = set()
    for s in cluster:
        classes.update(stem_to_classes.get(s, set()))
    if 0 in classes and 1 in classes:
        return "both"
    elif 0 in classes:
        return "crop_only"
    elif 1 in classes:
        return "weed_only"
    else:
        return "no_classes"


def create_reproducible_splits(
    stems: List[str],
    stem_to_classes: Dict[str, Set[int]],
    clusters: Optional[List[List[str]]] = None,
    ratios: Dict[str, float] = SPLIT_RATIOS,
    seed: int = DEFAULT_SEED,
) -> Dict[str, List[str]]:
    """Create reproducible stratified train/validation/test splits.
    
    Guarantees:
      1. Complete reproducibility with fixed random seed.
      2. No data leakage: all images in a cluster (exact duplicates) are strictly
         assigned to the same split.
      3. Proportions closely match target ratios (e.g., 80/10/10).
      4. Stratification across class composition (crop, weed, mixed).
    """
    rng = random.Random(seed)

    # Build clusters if not provided: each unique stem is its own cluster
    if clusters is None:
        clusters = [[s] for s in sorted(stems)]
    else:
        # Ensure all stems are accounted for
        clustered_stems = set(s for c in clusters for s in c)
        missing = set(stems) - clustered_stems
        clusters = list(clusters) + [[s] for s in sorted(list(missing))]

    # Group clusters by stratification stratum
    strata: Dict[str, List[List[str]]] = defaultdict(list)
    for cluster in clusters:
        stratum = determine_cluster_stratum(cluster, stem_to_classes)
        strata[stratum].append(sorted(cluster))

    splits: Dict[str, List[str]] = {
        "train": [],
        "val": [],
        "test": []
    }

    # For each stratum, sort deterministically then shuffle with seed
    for stratum_name in sorted(strata.keys()):
        stratum_clusters = sorted(strata[stratum_name], key=lambda c: c[0])
        rng.shuffle(stratum_clusters)

        # Count total items in this stratum
        total_items_in_stratum = sum(len(c) for c in stratum_clusters)
        target_val = int(round(total_items_in_stratum * ratios.get("val", 0.10)))
        target_test = int(round(total_items_in_stratum * ratios.get("test", 0.10)))

        val_clusters = []
        test_clusters = []
        train_clusters = []

        val_count = 0
        test_count = 0

        for c in stratum_clusters:
            c_size = len(c)
            # Allocate to val first if below target
            if val_count + c_size <= target_val and target_val > 0:
                val_clusters.append(c)
                val_count += c_size
            # Then to test if below target
            elif test_count + c_size <= target_test and target_test > 0:
                test_clusters.append(c)
                test_count += c_size
            else:
                train_clusters.append(c)

        for c in train_clusters:
            splits["train"].extend(c)
        for c in val_clusters:
            splits["val"].extend(c)
        for c in test_clusters:
            splits["test"].extend(c)

    # Sort final lists for deterministic output
    splits["train"] = sorted(splits["train"])
    splits["val"] = sorted(splits["val"])
    splits["test"] = sorted(splits["test"])

    return splits


def verify_split_integrity(
    splits: Dict[str, List[str]],
    clusters: List[List[str]],
    stem_to_classes: Dict[str, Set[int]],
    box_counts: Optional[Dict[str, Dict[int, int]]] = None
) -> SplitVerificationReport:
    """Verify that splits have no overlap, no duplicate leakage, and correct distributions."""
    train_set = set(splits.get("train", []))
    val_set = set(splits.get("val", []))
    test_set = set(splits.get("test", []))

    # Check for overlapping stems
    tv = train_set & val_set
    tt = train_set & test_set
    vt = val_set & test_set
    overlap = sorted(list(tv | tt | vt))

    # Check for duplicate cluster leakage
    leaked_groups = []
    for cluster in clusters:
        if len(cluster) <= 1:
            continue
        c_set = set(cluster)
        splits_present = []
        if c_set & train_set:
            splits_present.append("train")
        if c_set & val_set:
            splits_present.append("val")
        if c_set & test_set:
            splits_present.append("test")

        if len(splits_present) > 1:
            leaked_groups.append({
                "cluster": cluster,
                "splits": splits_present
            })

    total_count = len(train_set) + len(val_set) + len(test_set)
    t_ratio = len(train_set) / total_count if total_count else 0.0
    v_ratio = len(val_set) / total_count if total_count else 0.0
    s_ratio = len(test_set) / total_count if total_count else 0.0

    # Calculate class distribution per split
    class_dist: Dict[str, Dict[int, int]] = {}
    for split_name, split_stems in splits.items():
        dist: Dict[int, int] = defaultdict(int)
        for s in split_stems:
            if box_counts and s in box_counts:
                for cid, count in box_counts[s].items():
                    dist[cid] += count
            else:
                for cid in stem_to_classes.get(s, set()):
                    dist[cid] += 1
        class_dist[split_name] = dict(dist)

    is_valid = len(overlap) == 0 and len(leaked_groups) == 0

    return SplitVerificationReport(
        is_valid=is_valid,
        train_count=len(train_set),
        val_count=len(val_set),
        test_count=len(test_set),
        total_count=total_count,
        train_ratio=t_ratio,
        val_ratio=v_ratio,
        test_ratio=s_ratio,
        leakage_detected=(len(leaked_groups) > 0),
        overlap_detected=(len(overlap) > 0),
        leaked_duplicate_groups=leaked_groups,
        overlap_stems=overlap,
        class_distribution_per_split=class_dist
    )
