#!/usr/bin/env python3
"""Main CLI execution script for the SwarmAgri AI dataset preparation pipeline."""

import argparse
import sys
from pathlib import Path

# Add ml/src to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dataset.pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="SwarmAgri AI Dataset Validation and Preparation Pipeline"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible stratified splitting (default: 42)",
    )
    parser.add_argument(
        "--previews",
        type=int,
        default=16,
        help="Number of representative preview images to generate (default: 16)",
    )
    args = parser.parse_args()

    result = run_pipeline(seed=args.seed, preview_sample_count=args.previews)
    print("\n==================================================")
    print("Dataset Validation and Preparation Complete!")
    print(f"Verified Class Mapping: {result['class_mapping']}")
    print(f"Matched Image-Label Pairs: {result['matched_stems']}")
    print(f"Total Valid Bounding Boxes: {result['valid_boxes']} / {result['total_boxes']}")
    print(f"Splits Created: {result['splits']}")
    print(f"Prepared YOLO Dataset: {result['yolo_dataset_dir']}")
    print(f"Reports Directory: {result['reports_dir']}")
    print(f"Visual Previews Generated: {result['previews_generated']}")
    print("==================================================")


if __name__ == "__main__":
    main()
