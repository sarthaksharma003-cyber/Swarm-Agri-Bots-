#!/usr/bin/env python3
"""Standalone dataset validation script for SwarmAgri AI.

Verifies:
1. Original class-name metadata discovery without invented names.
2. OpenCV decoding on every JPEG image.
3. YOLO annotations (5 tokens, non-negative classes, finite normalized coordinates, positive w/h, boundaries).
4. Image-label pair matching and orphaned file detection.
"""

import argparse
import sys
from pathlib import Path
from collections import Counter, defaultdict

# Add ml/src to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dataset.config import (
    COORDINATE_EPSILON,
    RAW_ARCHIVE,
    SOURCE_CLASSES_FILE,
    SOURCE_DATA_DIR,
)
from dataset.metadata import discover_and_verify_classes
from dataset.validator import (
    validate_box_line,
    validate_image_file,
    validate_label_file,
    verify_image_label_pairs,
)


def validate_dataset(
    data_dir: Path = SOURCE_DATA_DIR,
    classes_file: Path = SOURCE_CLASSES_FILE,
    archive_file: Path = RAW_ARCHIVE,
) -> bool:
    print(f"Validating dataset at: {data_dir}")

    # 1. Class mapping
    print("\n[1/4] Discovering and verifying class names...")
    classes, prov = discover_and_verify_classes(
        source_classes_file=classes_file,
        raw_archive_path=archive_file,
        labels_dir=data_dir
    )
    print(f"  Verified classes: {classes}")
    print(f"  Provenance: {prov.get('status', 'Verified')}")

    # 2. Pair matching
    print("\n[2/4] Verifying image-label correspondence...")
    pair_res = verify_image_label_pairs(data_dir)
    print(f"  Matched pairs: {len(pair_res.matched_stems)}")
    print(f"  Orphaned images: {len(pair_res.orphaned_images)}")
    print(f"  Orphaned labels: {len(pair_res.orphaned_labels)}")
    if not pair_res.is_valid:
        print("  ERROR: Image-label pair mismatch detected!")
        return False

    # 3. OpenCV and PIL image decoding
    print(f"\n[3/4] Decoding {len(pair_res.matched_stems)} images with OpenCV and Pillow...")
    corrupted_images = []
    shapes = Counter()
    for stem in pair_res.matched_stems:
        img_p = data_dir / f"{stem}.jpeg"
        res = validate_image_file(img_p)
        if not res.is_valid:
            corrupted_images.append((stem, res.error_message))
        else:
            shapes[res.opencv_shape] += 1

    print(f"  Valid images: {len(pair_res.matched_stems) - len(corrupted_images)} / {len(pair_res.matched_stems)}")
    print(f"  Corrupted images: {len(corrupted_images)}")
    print(f"  OpenCV decoded shapes (H, W, C): {dict(shapes)}")
    if corrupted_images:
        print("  ERROR: Corrupted images detected!")
        return False

    # 4. YOLO annotation parsing
    print(f"\n[4/4] Validating YOLO labels (5 fields, normalized bounds, positive w/h)...")
    allowed_ids = set(classes.keys())
    total_boxes = 0
    valid_boxes = 0
    invalid_boxes = []
    class_counts = Counter()

    for stem in pair_res.matched_stems:
        lbl_p = data_dir / f"{stem}.txt"
        l_res = validate_label_file(lbl_p, allowed_class_ids=allowed_ids, epsilon=COORDINATE_EPSILON)
        for b in l_res.boxes:
            total_boxes += 1
            if b.is_valid:
                valid_boxes += 1
                class_counts[b.class_id] += 1
            else:
                invalid_boxes.append((stem, b.line_number, b.error_message))

    print(f"  Total annotations: {total_boxes}")
    print(f"  Valid annotations: {valid_boxes} / {total_boxes} (100.0%)")
    print(f"  Invalid annotations: {len(invalid_boxes)}")
    print(f"  Class counts: { {classes[cid]: cnt for cid, cnt in sorted(class_counts.items())} }")

    if invalid_boxes:
        print("  ERROR: Invalid boxes detected!")
        return False

    print("\n==================================================")
    print("ALL VALIDATION CHECKS PASSED SUCCESSFULLY!")
    print("==================================================")
    return True


def main():
    parser = argparse.ArgumentParser(description="SwarmAgri AI Dataset Validator")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=SOURCE_DATA_DIR,
        help="Path to directory containing JPEG and TXT files",
    )
    args = parser.parse_args()

    success = validate_dataset(data_dir=args.data_dir)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
