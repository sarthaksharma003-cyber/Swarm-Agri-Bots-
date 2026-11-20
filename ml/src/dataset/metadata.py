"""Metadata discovery and verification module for SwarmAgri AI dataset."""

import os
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


class MetadataVerificationError(Exception):
    """Raised when original class names or metadata cannot be verified."""
    pass


def extract_classes_from_text(text: str) -> Dict[int, str]:
    """Parse class names from newline-separated text.
    
    YOLO standard convention specifies that line 0 corresponds to class 0,
    line 1 to class 1, etc.
    """
    lines = [line.strip() for line in text.replace("\r", "").split("\n") if line.strip()]
    if not lines:
        raise MetadataVerificationError("Class definition text is empty.")
    return {idx: name for idx, name in enumerate(lines)}


def read_classes_from_file(file_path: Path) -> Dict[int, str]:
    """Read and parse class names from a classes.txt file."""
    if not file_path.exists():
        raise MetadataVerificationError(f"Classes file does not exist at: {file_path}")
    content = file_path.read_text(encoding="utf-8")
    return extract_classes_from_text(content)


def read_classes_from_zip(archive_path: Path) -> Optional[Dict[int, str]]:
    """Read classes definition directly from raw archive.zip if present."""
    if not archive_path.exists():
        return None
    with zipfile.ZipFile(archive_path, "r") as z:
        for name in z.namelist():
            if name.lower() == "classes.txt" or name.lower().endswith("/classes.txt"):
                raw_bytes = z.read(name)
                text = raw_bytes.decode("utf-8")
                return extract_classes_from_text(text)
    return None


def scan_observed_class_ids(labels_dir: Path) -> Set[int]:
    """Scan all label files in a directory to collect all observed class IDs."""
    if not labels_dir.exists():
        raise MetadataVerificationError(f"Labels directory does not exist: {labels_dir}")
    observed: Set[int] = set()
    for entry in labels_dir.iterdir():
        if entry.is_file() and entry.suffix.lower() == ".txt":
            with open(entry, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if parts:
                        try:
                            cid = int(parts[0])
                            observed.add(cid)
                        except ValueError:
                            pass
    return observed


def discover_and_verify_classes(
    source_classes_file: Optional[Path] = None,
    raw_archive_path: Optional[Path] = None,
    labels_dir: Optional[Path] = None,
) -> Tuple[Dict[int, str], Dict[str, str]]:
    """Discover class mapping from metadata and verify against observed label IDs.
    
    Returns:
        Tuple of (verified_class_mapping, verification_provenance_info)
    
    Raises:
        MetadataVerificationError: If classes cannot be verified or mismatch observed labels.
    """
    provenance: Dict[str, str] = {}
    classes_from_file: Optional[Dict[int, str]] = None
    classes_from_zip: Optional[Dict[int, str]] = None

    # Check extracted classes.txt
    if source_classes_file and source_classes_file.exists():
        classes_from_file = read_classes_from_file(source_classes_file)
        provenance["extracted_file"] = str(source_classes_file)

    # Check raw zip classes.txt
    if raw_archive_path and raw_archive_path.exists():
        classes_from_zip = read_classes_from_zip(raw_archive_path)
        if classes_from_zip:
            provenance["raw_archive"] = str(raw_archive_path)

    # Cross-verify if both are present
    if classes_from_file and classes_from_zip:
        if classes_from_file != classes_from_zip:
            raise MetadataVerificationError(
                f"Discrepancy detected between extracted classes ({classes_from_file}) "
                f"and raw archive classes ({classes_from_zip})."
            )
        verified_classes = classes_from_file
        provenance["status"] = "Verified matching between raw archive and extracted source"
    elif classes_from_file:
        verified_classes = classes_from_file
        provenance["status"] = "Verified from extracted source classes.txt"
    elif classes_from_zip:
        verified_classes = classes_from_zip
        provenance["status"] = "Verified from raw archive.zip metadata"
    else:
        raise MetadataVerificationError(
            "Original class names could NOT be found or verified in available metadata! "
            "Pipeline will not invent or guess class names."
        )

    # Verify against observed class IDs in label annotations
    if labels_dir and labels_dir.exists():
        observed_ids = scan_observed_class_ids(labels_dir)
        provenance["observed_ids"] = str(sorted(list(observed_ids)))
        defined_ids = set(verified_classes.keys())
        unknown_ids = observed_ids - defined_ids
        if unknown_ids:
            raise MetadataVerificationError(
                f"Labels contain class IDs {sorted(list(unknown_ids))} which are undefined "
                f"in verified class mapping: {verified_classes}."
            )
        unused_ids = defined_ids - observed_ids
        if unused_ids:
            provenance["warning"] = f"Defined classes {unused_ids} not observed in labels."

    return verified_classes, provenance
