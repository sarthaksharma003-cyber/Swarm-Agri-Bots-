"""Unit tests for image decoding (including OpenCV) and YOLO annotation validation."""

import pytest
from pathlib import Path
from PIL import Image
import cv2
import sys

# Ensure ml/src is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dataset.validator import (
    validate_box_line,
    validate_image_file,
    validate_label_file,
    verify_image_label_pairs,
)


@pytest.fixture
def sample_valid_jpeg(tmp_path):
    img_path = tmp_path / "valid_image.jpeg"
    img = Image.new("RGB", (512, 512), color=(100, 150, 200))
    img.save(img_path, "JPEG")
    return img_path


@pytest.fixture
def sample_corrupt_jpeg(tmp_path):
    img_path = tmp_path / "corrupt_image.jpeg"
    img_path.write_bytes(b"\xFF\xD8\xFF\xE0" + b"bad_corrupted_payload_not_jpeg" * 10)
    return img_path


def test_validate_image_file_valid(sample_valid_jpeg):
    res = validate_image_file(sample_valid_jpeg)
    assert res.is_valid is True
    assert res.width == 512
    assert res.height == 512
    assert res.mode == "RGB"
    assert res.format == "JPEG"
    assert res.file_size_bytes > 0
    assert res.opencv_decoded is True
    assert res.opencv_shape == (512, 512, 3)


def test_validate_image_file_opencv_decoding(sample_valid_jpeg):
    # Explicitly test cv2.imread directly on sample image
    cv_img = cv2.imread(str(sample_valid_jpeg))
    assert cv_img is not None
    assert cv_img.shape == (512, 512, 3)


def test_validate_image_file_corrupt(sample_corrupt_jpeg):
    res = validate_image_file(sample_corrupt_jpeg)
    assert res.is_valid is False
    assert res.opencv_decoded is False


def test_validate_image_file_missing(tmp_path):
    res = validate_image_file(tmp_path / "non_existent.jpeg")
    assert res.is_valid is False
    assert "does not exist" in res.error_message


def test_validate_image_file_empty(tmp_path):
    empty_p = tmp_path / "empty.jpeg"
    empty_p.touch()
    res = validate_image_file(empty_p)
    assert res.is_valid is False
    assert "empty" in res.error_message


def test_validate_box_line_valid():
    line = "0 0.500000 0.500000 0.200000 0.200000"
    b = validate_box_line(line, line_number=1, allowed_class_ids={0, 1})
    assert b.is_valid is True
    assert b.class_id == 0
    assert b.x_center == 0.5
    assert b.y_center == 0.5
    assert b.width == 0.2
    assert b.height == 0.2
    assert b.x_min == 0.4
    assert b.x_max == 0.6


def test_validate_box_line_wrong_field_count():
    line = "0 0.5 0.5 0.2"
    b = validate_box_line(line, line_number=1)
    assert b.is_valid is False
    assert "expected exactly 5" in b.error_message


def test_validate_box_line_negative_class():
    line = "-1 0.5 0.5 0.2 0.2"
    b = validate_box_line(line, line_number=1)
    assert b.is_valid is False
    assert "negative" in b.error_message


def test_validate_box_line_disallowed_class():
    line = "5 0.5 0.5 0.2 0.2"
    b = validate_box_line(line, line_number=1, allowed_class_ids={0, 1})
    assert b.is_valid is False
    assert "not in allowed classes" in b.error_message


def test_validate_box_line_non_positive_dim():
    line = "0 0.5 0.5 -0.2 0.2"
    b = validate_box_line(line, line_number=1)
    assert b.is_valid is False
    assert "Non-positive box dimension" in b.error_message


def test_validate_box_line_out_of_bounds():
    # Center at 0.9, width 0.4 -> xmax = 1.1 (spills outside image)
    line = "0 0.9 0.5 0.4 0.2"
    b = validate_box_line(line, line_number=1)
    assert b.is_valid is False
    assert "x_max" in b.error_message


def test_validate_box_line_epsilon_precision():
    # Center + width/2 = 1.0000005 (exceeds 1.0 by 5e-7, within 1e-5 epsilon)
    line = "1 0.513672 0.500977 0.785156 0.998047"
    b = validate_box_line(line, line_number=1, epsilon=1e-5)
    assert b.is_valid is True


def test_validate_label_file(tmp_path):
    lbl_file = tmp_path / "sample.txt"
    lbl_file.write_text(
        "0 0.5 0.5 0.2 0.2\n"
        "1 0.3 0.3 0.1 0.1\n"
    )
    res = validate_label_file(lbl_file, allowed_class_ids={0, 1})
    assert res.is_valid is True
    assert len(res.boxes) == 2
    assert res.class_counts == {0: 1, 1: 1}


def test_verify_image_label_pairs(tmp_path):
    # Matched pair
    (tmp_path / "img_1.jpeg").touch()
    (tmp_path / "img_1.txt").touch()

    # Orphaned image
    (tmp_path / "img_orphan.jpeg").touch()

    # Orphaned label
    (tmp_path / "lbl_orphan.txt").touch()

    res = verify_image_label_pairs(tmp_path)
    assert res.is_valid is False
    assert res.matched_stems == ["img_1"]
    assert len(res.orphaned_images) == 1
    assert res.orphaned_images[0].name == "img_orphan.jpeg"
    assert len(res.orphaned_labels) == 1
    assert res.orphaned_labels[0].name == "lbl_orphan.txt"
