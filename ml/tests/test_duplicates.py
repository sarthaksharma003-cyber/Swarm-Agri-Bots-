"""Unit tests for duplicate detection, perceptual hashing, and DSU clustering."""

import pytest
from pathlib import Path
from PIL import Image
import numpy as np
import sys

# Ensure ml/src is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dataset.duplicates import (
    DisjointSetUnion,
    compute_dhash_bits,
    compute_file_sha256,
    compute_pixel_sha256,
    detect_duplicates,
)


def test_compute_file_sha256(tmp_path):
    f = tmp_path / "test.bin"
    f.write_bytes(b"swarmagri_ai_test")
    h = compute_file_sha256(f)
    assert len(h) == 64
    assert isinstance(h, str)


def test_compute_pixel_sha256(tmp_path):
    p1 = tmp_path / "img1.png"
    p2 = tmp_path / "img2.png"

    img = Image.new("RGB", (64, 64), color=(120, 200, 50))
    img.save(p1)
    img.save(p2)

    h1 = compute_pixel_sha256(p1)
    h2 = compute_pixel_sha256(p2)
    assert h1 == h2


def test_dhash_and_hamming(tmp_path):
    p1 = tmp_path / "img1.jpeg"
    p2 = tmp_path / "img2.jpeg"
    p3 = tmp_path / "img3.jpeg"

    # Base gradient image
    arr1 = np.tile(np.linspace(0, 255, 64, dtype=np.uint8), (64, 1))
    Image.fromarray(arr1, mode="L").save(p1)

    # Identical image
    Image.fromarray(arr1, mode="L").save(p2)

    # Very different image (inverted)
    arr3 = 255 - arr1
    Image.fromarray(arr3, mode="L").save(p3)

    bits1 = compute_dhash_bits(p1)
    bits2 = compute_dhash_bits(p2)
    bits3 = compute_dhash_bits(p3)

    # Hamming distance between identical images must be 0
    dist_12 = np.sum(bits1 != bits2)
    assert dist_12 == 0

    # Inverted gradient has opposite comparisons
    dist_13 = np.sum(bits1 != bits3)
    assert dist_13 > 10


def test_disjoint_set_union():
    elements = ["a", "b", "c", "d", "e"]
    dsu = DisjointSetUnion(elements)

    dsu.union("a", "b")
    dsu.union("b", "c")
    dsu.union("d", "e")

    clusters = dsu.get_clusters()
    # Should have two clusters: ['a', 'b', 'c'] and ['d', 'e']
    assert len(clusters) == 2
    assert sorted(clusters[0]) == ["a", "b", "c"]
    assert sorted(clusters[1]) == ["d", "e"]


def test_detect_duplicates(tmp_path):
    p1 = tmp_path / "img_orig.jpeg"
    p2 = tmp_path / "img_exact_copy.jpeg"
    p3 = tmp_path / "img_distinct.jpeg"

    # Base image
    img1 = Image.new("RGB", (64, 64), color=(30, 40, 50))
    img1.save(p1, "JPEG")

    # Exact byte duplicate
    p2.write_bytes(p1.read_bytes())

    # Distinct image
    img3 = Image.new("RGB", (64, 64), color=(200, 150, 100))
    img3.save(p3, "JPEG")

    report = detect_duplicates([p1, p2, p3])
    assert report.total_images_analyzed == 3
    assert report.unique_file_hashes == 2
    assert len(report.exact_file_duplicate_groups) == 1
    # Check cluster grouping
    dup_cluster = [c for c in report.clusters if len(c) > 1]
    assert len(dup_cluster) == 1
    assert set(dup_cluster[0]) == {"img_orig", "img_exact_copy"}
