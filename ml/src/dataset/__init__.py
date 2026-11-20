"""Dataset package for SwarmAgri AI."""

from .config import (
    DEFAULT_SEED,
    ML_DIR,
    PREVIEWS_DIR,
    PROCESSED_SOURCE_DIR,
    RAW_ARCHIVE,
    REPORTS_DIR,
    SOURCE_CLASSES_FILE,
    SOURCE_DATA_DIR,
    SPLIT_RATIOS,
    WORKSPACE_ROOT,
    YOLO_DATA_YAML,
    YOLO_DATASET_DIR,
)
from .duplicates import (
    DisjointSetUnion,
    DuplicateReport,
    compute_file_sha256,
    compute_pixel_sha256,
    detect_duplicates,
)
from .metadata import (
    MetadataVerificationError,
    discover_and_verify_classes,
    read_classes_from_file,
    read_classes_from_zip,
)
from .pipeline import run_pipeline
from .splitter import (
    check_existing_splits,
    create_reproducible_splits,
    verify_split_integrity,
)
from .validator import (
    BoundingBox,
    ImageValidationResult,
    LabelValidationResult,
    PairValidationResult,
    validate_box_line,
    validate_image_file,
    validate_label_file,
    verify_image_label_pairs,
)
from .visualizer import (
    create_annotated_preview,
    create_preview_montage,
    draw_bounding_boxes,
)

__all__ = [
    "DEFAULT_SEED",
    "SPLIT_RATIOS",
    "WORKSPACE_ROOT",
    "RAW_ARCHIVE",
    "SOURCE_DATA_DIR",
    "SOURCE_CLASSES_FILE",
    "YOLO_DATASET_DIR",
    "YOLO_DATA_YAML",
    "REPORTS_DIR",
    "PREVIEWS_DIR",
    "MetadataVerificationError",
    "discover_and_verify_classes",
    "read_classes_from_file",
    "read_classes_from_zip",
    "BoundingBox",
    "ImageValidationResult",
    "LabelValidationResult",
    "PairValidationResult",
    "validate_image_file",
    "validate_box_line",
    "validate_label_file",
    "verify_image_label_pairs",
    "DisjointSetUnion",
    "DuplicateReport",
    "compute_file_sha256",
    "compute_pixel_sha256",
    "detect_duplicates",
    "check_existing_splits",
    "create_reproducible_splits",
    "verify_split_integrity",
    "draw_bounding_boxes",
    "create_annotated_preview",
    "create_preview_montage",
    "run_pipeline",
]
