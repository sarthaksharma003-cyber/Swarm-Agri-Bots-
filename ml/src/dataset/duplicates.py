"""Duplicate and near-duplicate detection and clustering module for SwarmAgri AI."""

import hashlib
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from PIL import Image
import numpy as np

from .config import DHASH_SIZE, NEAR_DUPLICATE_HAMMING_THRESHOLD, POTENTIAL_NEAR_DUPLICATE_THRESHOLD


@dataclass
class DuplicateReport:
    """Comprehensive duplicate and near-duplicate detection results."""
    total_images_analyzed: int
    unique_file_hashes: int
    unique_pixel_hashes: int
    exact_file_duplicate_groups: Dict[str, List[str]] = field(default_factory=dict)
    exact_pixel_duplicate_groups: Dict[str, List[str]] = field(default_factory=dict)
    identical_dhash_groups: Dict[str, List[str]] = field(default_factory=dict)
    near_duplicate_pairs: List[Dict[str, any]] = field(default_factory=list)
    clusters: List[List[str]] = field(default_factory=list)


class DisjointSetUnion:
    """Union-Find / Disjoint Set Union data structure for clustering items."""
    def __init__(self, elements: List[str]):
        self.parent = {el: el for el in elements}
        self.rank = {el: 0 for el in elements}

    def find(self, i: str) -> str:
        if self.parent[i] != i:
            self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i: str, j: str) -> None:
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            if self.rank[root_i] < self.rank[root_j]:
                self.parent[root_i] = root_j
            elif self.rank[root_i] > self.rank[root_j]:
                self.parent[root_j] = root_i
            else:
                self.parent[root_j] = root_i
                self.rank[root_i] += 1

    def get_clusters(self) -> List[List[str]]:
        clusters = defaultdict(list)
        for el in self.parent:
            root = self.find(el)
            clusters[root].append(el)
        return sorted([sorted(c) for c in clusters.values()], key=lambda c: (len(c), c[0]), reverse=True)


def compute_file_sha256(file_path: Path) -> str:
    """Compute SHA-256 hash of raw file bytes."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_pixel_sha256(image_path: Path) -> str:
    """Compute SHA-256 hash of decoded RGB raster buffer."""
    with Image.open(image_path) as img:
        rgb = img.convert("RGB")
        return hashlib.sha256(rgb.tobytes()).hexdigest()


def compute_dhash_bits(image_path: Path, hash_size: int = DHASH_SIZE) -> np.ndarray:
    """Compute difference hash (dHash) boolean array for perceptual similarity."""
    with Image.open(image_path) as img:
        img_gray = img.convert("L").resize(
            (hash_size + 1, hash_size),
            Image.Resampling.LANCZOS
        )
        arr = np.array(img_gray, dtype=np.int32)
        diff = arr[:, 1:] > arr[:, :-1]
        return diff.flatten()


def dhash_bits_to_hex(bits: np.ndarray) -> str:
    """Convert boolean bits array to hex string representation."""
    val = 0
    for idx, bit in enumerate(bits):
        if bit:
            val += 1 << idx
    return f"{val:016x}"


def detect_duplicates(
    image_paths: List[Path],
    check_pixel_hash: bool = True,
    check_near_duplicates: bool = True,
    near_dup_threshold: int = POTENTIAL_NEAR_DUPLICATE_THRESHOLD,
) -> DuplicateReport:
    """Run comprehensive exact and near duplicate detection across images."""
    file_hashes: Dict[str, List[str]] = defaultdict(list)
    pixel_hashes: Dict[str, List[str]] = defaultdict(list)
    dhash_map: Dict[str, str] = {}
    dhash_bits_list: List[np.ndarray] = []
    stems = [p.stem for p in image_paths]

    for p in image_paths:
        stem = p.stem
        # 1. Raw file hash
        fh = compute_file_sha256(p)
        file_hashes[fh].append(stem)

        # 2. Decoded pixel hash
        if check_pixel_hash:
            ph = compute_pixel_sha256(p)
            pixel_hashes[ph].append(stem)

        # 3. Perceptual dHash
        if check_near_duplicates:
            bits = compute_dhash_bits(p)
            dhash_bits_list.append(bits)
            dhash_hex = dhash_bits_to_hex(bits)
            dhash_map[stem] = dhash_hex

    # Identify exact file duplicates
    exact_file_dups = {h: s for h, s in file_hashes.items() if len(s) > 1}
    # Identify exact pixel duplicates
    exact_pixel_dups = {h: s for h, s in pixel_hashes.items() if len(s) > 1}

    # Group identical dHash
    dhash_groups: Dict[str, List[str]] = defaultdict(list)
    for stem, dhex in dhash_map.items():
        dhash_groups[dhex].append(stem)
    identical_dhash_groups = {h: s for h, s in dhash_groups.items() if len(s) > 1}

    # Check pairwise Hamming distance for near-duplicates
    near_dup_pairs = []
    if check_near_duplicates and dhash_bits_list:
        bits_matrix = np.stack(dhash_bits_list)  # (N, 64)
        N = len(image_paths)
        for i in range(N):
            xor_diff = np.bitwise_xor(bits_matrix[i], bits_matrix[i + 1:])
            dists = np.sum(xor_diff, axis=1)
            for j_offset, dist in enumerate(dists):
                j = i + 1 + j_offset
                d_val = int(dist)
                if d_val <= near_dup_threshold:
                    similarity_pct = round((1.0 - d_val / 64.0) * 100.0, 2)
                    near_dup_pairs.append({
                        "image_a": stems[i],
                        "image_b": stems[j],
                        "hamming_distance": d_val,
                        "similarity_pct": similarity_pct,
                        "identical_dhash": (d_val == 0)
                    })

    # Build DSU clusters for exact duplicates (pixel or file level)
    dsu = DisjointSetUnion(stems)
    for group in exact_file_dups.values():
        for i in range(len(group) - 1):
            dsu.union(group[i], group[i + 1])
    for group in exact_pixel_dups.values():
        for i in range(len(group) - 1):
            dsu.union(group[i], group[i + 1])

    clusters = dsu.get_clusters()

    return DuplicateReport(
        total_images_analyzed=len(image_paths),
        unique_file_hashes=len(file_hashes),
        unique_pixel_hashes=len(pixel_hashes) if check_pixel_hash else 0,
        exact_file_duplicate_groups=exact_file_dups,
        exact_pixel_duplicate_groups=exact_pixel_dups,
        identical_dhash_groups=identical_dhash_groups,
        near_duplicate_pairs=near_dup_pairs,
        clusters=clusters
    )
