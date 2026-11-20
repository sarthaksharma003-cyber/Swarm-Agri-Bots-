"""Configuration and constants for SwarmAgri AI dataset preparation pipeline."""

from pathlib import Path
from typing import Dict, Tuple

# Workspace root
WORKSPACE_ROOT = Path("/home/smasher/Desktop/swarmagri-ai").resolve()

# Source dataset paths
RAW_DIR = WORKSPACE_ROOT / "data" / "raw"
RAW_ARCHIVE = RAW_DIR / "archive.zip"
PROCESSED_SOURCE_DIR = WORKSPACE_ROOT / "data" / "processed" / "source"
SOURCE_CLASSES_FILE = PROCESSED_SOURCE_DIR / "classes.txt"
SOURCE_DATA_DIR = PROCESSED_SOURCE_DIR / "agri_data" / "data"

# Prepared YOLO dataset target
YOLO_DATASET_DIR = WORKSPACE_ROOT / "data" / "processed" / "yolo_dataset"
YOLO_DATA_YAML = YOLO_DATASET_DIR / "data.yaml"

# Reports and previews target
ML_DIR = WORKSPACE_ROOT / "ml"
REPORTS_DIR = ML_DIR / "reports"
PREVIEWS_DIR = REPORTS_DIR / "previews"

# Split configuration
DEFAULT_SEED = 42
SPLIT_RATIOS: Dict[str, float] = {
    "train": 0.80,
    "val": 0.10,
    "test": 0.10,
}

# Validation tolerances
COORDINATE_EPSILON = 1e-5  # Float rounding tolerance for YOLO bounds (e.g. 5e-7 on 512px)

# Image hashing parameters
DHASH_SIZE = 8
NEAR_DUPLICATE_HAMMING_THRESHOLD = 0  # Identical perceptual hash
POTENTIAL_NEAR_DUPLICATE_THRESHOLD = 2  # Close perceptual similarity

# Visualizer styling
CLASS_COLORS = {
    0: (46, 204, 113),   # Crop: Emerald Green (#2ECC71)
    1: (231, 76, 60),    # Weed: Coral Red (#E74C3C)
}
CLASS_COLORS_RGBA = {
    0: (46, 204, 113, 60),   # Crop semi-transparent fill
    1: (231, 76, 60, 60),    # Weed semi-transparent fill
}
