#!/usr/bin/env python3
"""Metadata inspection and class verification utility for SwarmAgri AI."""

import sys
from pathlib import Path

# Add ml/src to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dataset.config import RAW_ARCHIVE, SOURCE_CLASSES_FILE, SOURCE_DATA_DIR
from dataset.metadata import (
    discover_and_verify_classes,
    read_classes_from_file,
    read_classes_from_zip,
    scan_observed_class_ids,
)


def main():
    print("==================================================")
    print("SwarmAgri AI - Dataset Metadata Inspection")
    print("==================================================")

    # 1. Extracted source classes.txt
    print(f"\n1. Extracted classes file: {SOURCE_CLASSES_FILE}")
    if SOURCE_CLASSES_FILE.exists():
        classes_extracted = read_classes_from_file(SOURCE_CLASSES_FILE)
        print(f"   Classes found: {classes_extracted}")
    else:
        print("   File not found.")

    # 2. Raw archive classes.txt
    print(f"\n2. Raw zip archive: {RAW_ARCHIVE}")
    if RAW_ARCHIVE.exists():
        classes_zip = read_classes_from_zip(RAW_ARCHIVE)
        print(f"   Classes found in archive: {classes_zip}")
    else:
        print("   Archive not found.")

    # 3. Label annotations scan
    print(f"\n3. Scanning observed class IDs in: {SOURCE_DATA_DIR}")
    if SOURCE_DATA_DIR.exists():
        observed_ids = scan_observed_class_ids(SOURCE_DATA_DIR)
        print(f"   Observed class IDs in TXT files: {sorted(list(observed_ids))}")
    else:
        print("   Data directory not found.")

    # 4. Verification result
    print("\n4. Verification Status:")
    try:
        verified, prov = discover_and_verify_classes(
            source_classes_file=SOURCE_CLASSES_FILE,
            raw_archive_path=RAW_ARCHIVE,
            labels_dir=SOURCE_DATA_DIR,
        )
        print(f"   STATUS: VERIFIED")
        print(f"   Verified Class Names: {verified}")
        print(f"   Provenance Details: {prov}")
    except Exception as e:
        print(f"   STATUS: FAILED ({e})")
        sys.exit(1)


if __name__ == "__main__":
    main()
