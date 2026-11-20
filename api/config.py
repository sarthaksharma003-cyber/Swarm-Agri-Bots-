"""Configuration settings and constants for SwarmAgri AI Inference API."""

import os
from pathlib import Path
from typing import Dict, Set

# Base repository directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Primary verified YOLO checkpoint path (Stage 3 10-epoch best model)
DEFAULT_MODEL_PATH = BASE_DIR / "ml" / "models" / "yolo11n" / "run_10epochs" / "weights" / "best.pt"
FALLBACK_MODEL_PATH = BASE_DIR / "ml" / "models" / "yolo11n" / "best.pt"

# Configurable model path via environment variable
MODEL_PATH = Path(os.getenv("SWARMAGRI_MODEL_PATH", str(DEFAULT_MODEL_PATH)))

# Verified class mapping discovered from original metadata (classes.txt)
VERIFIED_CLASS_MAPPING: Dict[int, str] = {
    0: "crop",
    1: "weed",
}

# Confidence threshold configuration:
# Defaulted to 0.25 based on operational considerations for robotic swarms (prioritizing
# weed recall to mitigate the risk of missed weeds). This value is NOT claimed to be
# universally optimal and should be tuned per crop canopy and field condition.
DEFAULT_CONF_THRESHOLD: float = 0.25
MIN_CONF_THRESHOLD: float = 0.01
MAX_CONF_THRESHOLD: float = 1.00

# Input validation & security constraints
MAX_UPLOAD_SIZE_BYTES: int = 10 * 1024 * 1024  # 10 MB maximum upload limit
ALLOWED_CONTENT_TYPES: Set[str] = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/jpg",
}
ALLOWED_FILE_EXTENSIONS: Set[str] = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
}

# Default image inference size
DEFAULT_IMAGE_SIZE: int = 512

# Swarm simulation & safety constraints (Physical commands disabled)
SIMULATION_MODE: bool = True
HARDWARE_DISPATCH_ENABLED: bool = False

# API Metadata
API_TITLE: str = "SwarmAgri AI - Agricultural Perception & Swarm Task Service"
API_DESCRIPTION: str = (
    "FastAPI service providing real-time crop and weed detection using fine-tuned "
    "YOLO11n, alongside decoupled swarm simulation task allocation for precision "
    "agricultural robots."
)
API_VERSION: str = "1.0.0"
