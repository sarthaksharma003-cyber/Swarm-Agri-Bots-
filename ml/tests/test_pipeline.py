"""Integration tests for the end-to-end dataset pipeline."""

import pytest
from pathlib import Path
from PIL import Image
import yaml
import sys

# Ensure ml/src is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dataset.pipeline import run_pipeline


@pytest.fixture
def mini_dataset(tmp_path):
    source_dir = tmp_path / "source"
    data_dir = source_dir / "agri_data" / "data"
    data_dir.mkdir(parents=True)

    # Write classes.txt
    classes_file = source_dir / "classes.txt"
    classes_file.write_text("crop\nweed\n", encoding="utf-8")

    # Generate 15 dummy images and labels
    for i in range(15):
        stem = f"test_img_{i:03d}"
        img_p = data_dir / f"{stem}.jpeg"
        lbl_p = data_dir / f"{stem}.txt"

        # Create 128x128 image
        img = Image.new("RGB", (128, 128), color=(i * 15 % 255, 120, 180))
        img.save(img_p, "JPEG")

        # Class 0 for even, Class 1 for odd, mixed for index 14
        if i == 14:
            lbl_p.write_text("0 0.5 0.5 0.2 0.2\n1 0.2 0.2 0.1 0.1\n")
        elif i % 2 == 0:
            lbl_p.write_text("0 0.5 0.5 0.3 0.3\n")
        else:
            lbl_p.write_text("1 0.4 0.4 0.2 0.2\n")

    return {
        "source_dir": source_dir,
        "data_dir": data_dir,
        "classes_file": classes_file,
        "yolo_target": tmp_path / "output_yolo",
        "reports_dir": tmp_path / "output_reports",
    }


def test_end_to_end_pipeline(mini_dataset):
    res = run_pipeline(
        source_data_dir=mini_dataset["data_dir"],
        source_classes_file=mini_dataset["classes_file"],
        raw_archive_path=Path("/non_existent_archive.zip"),  # will fall back to source_classes_file
        yolo_target_dir=mini_dataset["yolo_target"],
        reports_dir=mini_dataset["reports_dir"],
        seed=42,
        preview_sample_count=4,
    )

    assert res["status"] == "SUCCESS"
    assert res["class_mapping"] == {0: "crop", 1: "weed"}
    assert res["matched_stems"] == 15
    assert res["total_boxes"] == 16

    # Verify YOLO target structure
    yolo_dir = mini_dataset["yolo_target"]
    assert (yolo_dir / "data.yaml").exists()

    for s in ["train", "val", "test"]:
        assert (yolo_dir / "images" / s).is_dir()
        assert (yolo_dir / "labels" / s).is_dir()

    # Verify data.yaml contents
    with open(yolo_dir / "data.yaml") as f:
        data_cfg = yaml.safe_load(f)
    assert data_cfg["nc"] == 2
    assert data_cfg["names"] == {0: "crop", 1: "weed"}

    # Verify reports were created
    rep_dir = mini_dataset["reports_dir"]
    assert (rep_dir / "data_quality_report.json").exists()
    assert (rep_dir / "data_quality_report.md").exists()
    assert (rep_dir / "class_frequency_report.json").exists()
    assert (rep_dir / "image_dimension_report.json").exists()
    assert (rep_dir / "annotation_count_report.json").exists()
    assert (rep_dir / "split_summary.json").exists()

    # Verify previews
    prev_dir = rep_dir / "previews"
    assert prev_dir.is_dir()
    preview_jpegs = list(prev_dir.glob("*.jpg"))
    assert len(preview_jpegs) > 0
    assert (prev_dir / "preview_montage.jpg").exists()
