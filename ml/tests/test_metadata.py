"""Unit tests for metadata discovery and class verification."""

import pytest
import zipfile
from pathlib import Path
import sys

# Ensure ml/src is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dataset.metadata import (
    MetadataVerificationError,
    discover_and_verify_classes,
    extract_classes_from_text,
    read_classes_from_file,
    read_classes_from_zip,
    scan_observed_class_ids,
)


def test_extract_classes_from_text_standard():
    raw_text = "crop\nweed\n"
    classes = extract_classes_from_text(raw_text)
    assert classes == {0: "crop", 1: "weed"}


def test_extract_classes_from_text_windows_crlf():
    raw_text = "crop\r\nweed\r\n"
    classes = extract_classes_from_text(raw_text)
    assert classes == {0: "crop", 1: "weed"}


def test_extract_classes_from_text_empty():
    with pytest.raises(MetadataVerificationError, match="empty"):
        extract_classes_from_text("   \n   \r\n  ")


def test_read_classes_from_file(tmp_path):
    class_file = tmp_path / "classes.txt"
    class_file.write_text("crop\nweed\n", encoding="utf-8")
    classes = read_classes_from_file(class_file)
    assert classes == {0: "crop", 1: "weed"}


def test_read_classes_from_missing_file(tmp_path):
    missing_file = tmp_path / "non_existent.txt"
    with pytest.raises(MetadataVerificationError):
        read_classes_from_file(missing_file)


def test_read_classes_from_zip(tmp_path):
    zip_path = tmp_path / "test_archive.zip"
    with zipfile.ZipFile(zip_path, "w") as z:
        z.writestr("classes.txt", "crop\r\nweed\r\n")
        z.writestr("data/sample.txt", "dummy")

    classes = read_classes_from_zip(zip_path)
    assert classes == {0: "crop", 1: "weed"}


def test_scan_observed_class_ids(tmp_path):
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    (labels_dir / "sample1.txt").write_text("0 0.5 0.5 0.2 0.2\n1 0.4 0.4 0.1 0.1\n")
    (labels_dir / "sample2.txt").write_text("0 0.2 0.2 0.1 0.1\n")

    observed = scan_observed_class_ids(labels_dir)
    assert observed == {0, 1}


def test_discover_and_verify_classes_success(tmp_path):
    class_file = tmp_path / "classes.txt"
    class_file.write_text("crop\nweed\n")

    labels_dir = tmp_path / "data"
    labels_dir.mkdir()
    (labels_dir / "sample.txt").write_text("0 0.5 0.5 0.2 0.2\n1 0.3 0.3 0.1 0.1\n")

    mapping, prov = discover_and_verify_classes(
        source_classes_file=class_file,
        labels_dir=labels_dir
    )
    assert mapping == {0: "crop", 1: "weed"}
    assert "extracted_file" in prov


def test_discover_and_verify_classes_unknown_id(tmp_path):
    class_file = tmp_path / "classes.txt"
    class_file.write_text("crop\nweed\n")

    labels_dir = tmp_path / "data"
    labels_dir.mkdir()
    # Label contains class ID 2 which is not in {0, 1}
    (labels_dir / "sample.txt").write_text("2 0.5 0.5 0.2 0.2\n")

    with pytest.raises(MetadataVerificationError, match="undefined"):
        discover_and_verify_classes(
            source_classes_file=class_file,
            labels_dir=labels_dir
        )


def test_discover_and_verify_classes_missing_metadata(tmp_path):
    with pytest.raises(MetadataVerificationError, match="could NOT be found"):
        discover_and_verify_classes(
            source_classes_file=tmp_path / "missing.txt",
            raw_archive_path=tmp_path / "missing.zip"
        )
