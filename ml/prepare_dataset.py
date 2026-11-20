#!/usr/bin/env python3
"""CLI script to run the SwarmAgri AI dataset preparation pipeline."""

import argparse
import sys
from pathlib import Path

# Add ml/src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

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
    print("\nPipeline finished successfully!")
    print(f"Verified Classes: {result['class_mapping']}")
    print(f"Splits: {result['splits']}")
    print(f"YOLO Dataset: {result['yolo_dataset_dir']}")
    print(f"Reports: {result['reports_dir']}")


if __name__ == "__main__":
    main()
