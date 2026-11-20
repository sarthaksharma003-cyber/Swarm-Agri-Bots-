"""Tests for training configuration, device resolution, and prediction interfaces.

These tests validate configuration parsing, error handling, and class mappings
without downloading weights or initiating model training runs.
"""

import pytest
from pathlib import Path
import sys

# Ensure workspace root and ml/scripts are in sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT / "ml" / "scripts"))
sys.path.insert(0, str(WORKSPACE_ROOT / "ml" / "src"))

from train_yolo import detect_device, parse_args, verify_data_yaml
from evaluate_yolo import get_hardware_info
from predict_image import predict_single_image


def test_training_args_defaults():
    args = parse_args([])
    assert args.imgsz == 512
    assert args.epochs == 50
    assert args.batch == 4
    assert args.patience == 10
    assert args.seed == 42
    assert args.workers == 2
    assert args.weights == "yolo11n.pt"
    assert args.device == "auto"
    assert args.data == "data/processed/yolo_dataset/data.yaml"


def test_training_args_custom_overrides():
    custom_cli = [
        "--epochs", "20",
        "--batch", "8",
        "--imgsz", "640",
        "--patience", "5",
        "--seed", "123",
        "--device", "cpu",
        "--workers", "4",
    ]
    args = parse_args(custom_cli)
    assert args.epochs == 20
    assert args.batch == 8
    assert args.imgsz == 640
    assert args.patience == 5
    assert args.seed == 123
    assert args.device == "cpu"
    assert args.workers == 4


def test_detect_device_cpu():
    dev = detect_device("cpu")
    assert dev == "cpu"


def test_detect_device_auto():
    dev = detect_device("auto")
    assert dev in ["cpu", "cuda:0"]


def test_verify_data_yaml_real_dataset():
    data_yaml_path = WORKSPACE_ROOT / "data" / "processed" / "yolo_dataset" / "data.yaml"
    assert data_yaml_path.exists(), "Prepared data.yaml must exist."

    cfg = verify_data_yaml(data_yaml_path)
    assert cfg["nc"] == 2
    names = cfg["names"]
    if isinstance(names, dict):
        assert names[0] == "crop"
        assert names[1] == "weed"
    else:
        assert names == ["crop", "weed"]


def test_verify_data_yaml_missing(tmp_path):
    missing_yaml = tmp_path / "non_existent.yaml"
    with pytest.raises(FileNotFoundError, match="not found"):
        verify_data_yaml(missing_yaml)


def test_verify_data_yaml_invalid_structure(tmp_path):
    bad_yaml = tmp_path / "bad_data.yaml"
    bad_yaml.write_text("train: images/train\n")  # missing val, nc, names
    with pytest.raises(ValueError, match="Missing required key"):
        verify_data_yaml(bad_yaml)


def test_get_hardware_info():
    info = get_hardware_info()
    assert "os" in info
    assert "platform" in info
    assert "python_version" in info
    assert "device_name" in info


def test_predict_missing_image_graceful(tmp_path):
    missing_img = tmp_path / "no_such_image.jpeg"
    dummy_model = tmp_path / "dummy.pt"
    dummy_model.touch()

    with pytest.raises(FileNotFoundError, match="Input image not found"):
        predict_single_image(image_path=missing_img, model_path=dummy_model)


def test_predict_missing_model_graceful(tmp_path):
    valid_img = tmp_path / "sample.jpeg"
    valid_img.touch()
    missing_model = tmp_path / "missing_model.pt"

    with pytest.raises(FileNotFoundError, match="Trained model checkpoint not found"):
        predict_single_image(image_path=valid_img, model_path=missing_model)
