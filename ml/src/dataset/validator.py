"""Image and YOLO annotation validation module for SwarmAgri AI dataset."""

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import cv2
from PIL import Image

from .config import COORDINATE_EPSILON


@dataclass
class ImageValidationResult:
    """Result of decoding and validating a single image file."""
    path: Path
    is_valid: bool
    width: int = 0
    height: int = 0
    mode: str = ""
    format: str = ""
    file_size_bytes: int = 0
    opencv_decoded: bool = False
    opencv_shape: Optional[Tuple[int, int, int]] = None
    error_message: Optional[str] = None


@dataclass
class BoundingBox:
    """Parsed and validated YOLO bounding box."""
    line_number: int
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    is_valid: bool
    error_message: Optional[str] = None


@dataclass
class LabelValidationResult:
    """Result of parsing and validating a single YOLO label file."""
    path: Path
    is_valid: bool
    boxes: List[BoundingBox] = field(default_factory=list)
    error_messages: List[str] = field(default_factory=list)
    class_counts: Dict[int, int] = field(default_factory=dict)


@dataclass
class PairValidationResult:
    """Result of verifying image and label correspondence."""
    matched_stems: List[str] = field(default_factory=list)
    orphaned_images: List[Path] = field(default_factory=list)
    orphaned_labels: List[Path] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return len(self.orphaned_images) == 0 and len(self.orphaned_labels) == 0


def validate_image_file(image_path: Path) -> ImageValidationResult:
    """Comprehensively validate an image by decoding pixel data.
    
    Checks that the file exists, has non-zero size, is a valid JPEG,
    can be completely decoded without error, and reports dimensions.
    """
    if not image_path.exists():
        return ImageValidationResult(
            path=image_path,
            is_valid=False,
            error_message="File does not exist."
        )

    file_size = image_path.stat().st_size
    if file_size == 0:
        return ImageValidationResult(
            path=image_path,
            is_valid=False,
            file_size_bytes=0,
            error_message="File is empty (0 bytes)."
        )

    try:
        # Step 1: verify file header and structure
        with Image.open(image_path) as img:
            img_format = img.format or ""
            img.verify()

        # Step 2: reopen and load actual pixel raster to catch partial truncation
        with Image.open(image_path) as img:
            img.load()
            w, h = img.size
            mode = img.mode

        if w <= 0 or h <= 0:
            return ImageValidationResult(
                path=image_path,
                is_valid=False,
                width=w,
                height=h,
                mode=mode,
                format=img_format,
                file_size_bytes=file_size,
                error_message=f"Invalid image dimensions: {w}x{h}"
            )

        # Step 3: decode with OpenCV to verify OpenCV compatibility
        cv_img = cv2.imread(str(image_path))
        if cv_img is None:
            return ImageValidationResult(
                path=image_path,
                is_valid=False,
                width=w,
                height=h,
                mode=mode,
                format=img_format,
                file_size_bytes=file_size,
                opencv_decoded=False,
                error_message="OpenCV cv2.imread failed to decode image (returned None)."
            )

        cv_h, cv_w, cv_c = cv_img.shape
        if cv_h != h or cv_w != w:
            return ImageValidationResult(
                path=image_path,
                is_valid=False,
                width=w,
                height=h,
                mode=mode,
                format=img_format,
                file_size_bytes=file_size,
                opencv_decoded=False,
                error_message=f"Dimension mismatch between PIL ({w}x{h}) and OpenCV ({cv_w}x{cv_h})."
            )

        return ImageValidationResult(
            path=image_path,
            is_valid=True,
            width=w,
            height=h,
            mode=mode,
            format=img_format,
            file_size_bytes=file_size,
            opencv_decoded=True,
            opencv_shape=(cv_h, cv_w, cv_c),
        )
    except Exception as e:
        return ImageValidationResult(
            path=image_path,
            is_valid=False,
            file_size_bytes=file_size,
            error_message=f"Decode error: {type(e).__name__}: {str(e)}"
        )


def validate_box_line(
    line: str,
    line_number: int,
    allowed_class_ids: Optional[Set[int]] = None,
    epsilon: float = COORDINATE_EPSILON
) -> BoundingBox:
    """Validate a single YOLO annotation row.
    
    YOLO standard format: <class_id> <x_center> <y_center> <width> <height>
    Constraints:
      - Exactly 5 whitespace-separated fields
      - class_id: nonnegative integer matching allowed_class_ids if provided
      - x_center, y_center, width, height: finite floats in [0, 1]
      - width > 0, height > 0
      - Box boundaries within [0, 1] within floating-point epsilon
    """
    tokens = line.strip().split()
    if len(tokens) != 5:
        return BoundingBox(
            line_number=line_number,
            class_id=-1,
            x_center=0.0,
            y_center=0.0,
            width=0.0,
            height=0.0,
            x_min=0.0,
            y_min=0.0,
            x_max=0.0,
            y_max=0.0,
            is_valid=False,
            error_message=f"Row has {len(tokens)} fields, expected exactly 5."
        )

    # Class ID validation
    try:
        class_id = int(tokens[0])
    except ValueError:
        return BoundingBox(
            line_number=line_number,
            class_id=-1,
            x_center=0.0,
            y_center=0.0,
            width=0.0,
            height=0.0,
            x_min=0.0,
            y_min=0.0,
            x_max=0.0,
            y_max=0.0,
            is_valid=False,
            error_message=f"Class ID '{tokens[0]}' is not an integer."
        )

    if class_id < 0:
        return BoundingBox(
            line_number=line_number,
            class_id=class_id,
            x_center=0.0,
            y_center=0.0,
            width=0.0,
            height=0.0,
            x_min=0.0,
            y_min=0.0,
            x_max=0.0,
            y_max=0.0,
            is_valid=False,
            error_message=f"Class ID {class_id} is negative."
        )

    if allowed_class_ids is not None and class_id not in allowed_class_ids:
        return BoundingBox(
            line_number=line_number,
            class_id=class_id,
            x_center=0.0,
            y_center=0.0,
            width=0.0,
            height=0.0,
            x_min=0.0,
            y_min=0.0,
            x_max=0.0,
            y_max=0.0,
            is_valid=False,
            error_message=f"Class ID {class_id} not in allowed classes {allowed_class_ids}."
        )

    # Coordinate parsing
    try:
        coords = [float(t) for t in tokens[1:]]
    except ValueError as e:
        return BoundingBox(
            line_number=line_number,
            class_id=class_id,
            x_center=0.0,
            y_center=0.0,
            width=0.0,
            height=0.0,
            x_min=0.0,
            y_min=0.0,
            x_max=0.0,
            y_max=0.0,
            is_valid=False,
            error_message=f"Coordinate parse error: {e}"
        )

    xc, yc, w, h = coords

    # Check finite
    if not (math.isfinite(xc) and math.isfinite(yc) and math.isfinite(w) and math.isfinite(h)):
        return BoundingBox(
            line_number=line_number,
            class_id=class_id,
            x_center=xc,
            y_center=yc,
            width=w,
            height=h,
            x_min=0.0,
            y_min=0.0,
            x_max=0.0,
            y_max=0.0,
            is_valid=False,
            error_message="Coordinates contain NaN or infinity."
        )

    # Positive dimensions
    if w <= 0.0 or h <= 0.0:
        return BoundingBox(
            line_number=line_number,
            class_id=class_id,
            x_center=xc,
            y_center=yc,
            width=w,
            height=h,
            x_min=0.0,
            y_min=0.0,
            x_max=0.0,
            y_max=0.0,
            is_valid=False,
            error_message=f"Non-positive box dimension: width={w}, height={h}."
        )

    # Calculate bounding edges
    xmin = xc - w / 2.0
    xmax = xc + w / 2.0
    ymin = yc - h / 2.0
    ymax = yc + h / 2.0

    # Bounds check with epsilon
    errors = []
    if xmin < -epsilon:
        errors.append(f"x_min ({xmin:.6f}) < 0")
    if ymin < -epsilon:
        errors.append(f"y_min ({ymin:.6f}) < 0")
    if xmax > 1.0 + epsilon:
        errors.append(f"x_max ({xmax:.6f}) > 1.0")
    if ymax > 1.0 + epsilon:
        errors.append(f"y_max ({ymax:.6f}) > 1.0")

    if errors:
        return BoundingBox(
            line_number=line_number,
            class_id=class_id,
            x_center=xc,
            y_center=yc,
            width=w,
            height=h,
            x_min=xmin,
            y_min=ymin,
            x_max=xmax,
            y_max=ymax,
            is_valid=False,
            error_message="; ".join(errors)
        )

    return BoundingBox(
        line_number=line_number,
        class_id=class_id,
        x_center=xc,
        y_center=yc,
        width=w,
        height=h,
        x_min=xmin,
        y_min=ymin,
        x_max=xmax,
        y_max=ymax,
        is_valid=True
    )


def validate_label_file(
    label_path: Path,
    allowed_class_ids: Optional[Set[int]] = None,
    allow_empty: bool = False,
    epsilon: float = COORDINATE_EPSILON
) -> LabelValidationResult:
    """Validate a single YOLO annotation text file."""
    if not label_path.exists():
        return LabelValidationResult(
            path=label_path,
            is_valid=False,
            error_messages=["File does not exist."]
        )

    try:
        with open(label_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception as e:
        return LabelValidationResult(
            path=label_path,
            is_valid=False,
            error_messages=[f"Read error: {e}"]
        )

    non_empty_lines = [(i + 1, line) for i, line in enumerate(lines) if line.strip()]

    if not non_empty_lines and not allow_empty:
        return LabelValidationResult(
            path=label_path,
            is_valid=False,
            error_messages=["Label file is completely empty (0 bounding boxes)."]
        )

    boxes: List[BoundingBox] = []
    error_messages: List[str] = []
    class_counts: Dict[int, int] = {}

    for line_no, line in non_empty_lines:
        box = validate_box_line(
            line=line,
            line_number=line_no,
            allowed_class_ids=allowed_class_ids,
            epsilon=epsilon
        )
        boxes.append(box)
        if not box.is_valid:
            error_messages.append(f"Line {line_no}: {box.error_message}")
        else:
            class_counts[box.class_id] = class_counts.get(box.class_id, 0) + 1

    return LabelValidationResult(
        path=label_path,
        is_valid=len(error_messages) == 0,
        boxes=boxes,
        error_messages=error_messages,
        class_counts=class_counts
    )


def verify_image_label_pairs(data_dir: Path) -> PairValidationResult:
    """Verify that every image has a matching label file and vice versa."""
    image_exts = {".jpeg", ".jpg", ".png"}
    images: Dict[str, Path] = {}
    labels: Dict[str, Path] = {}

    for p in data_dir.iterdir():
        if not p.is_file():
            continue
        stem = p.stem
        ext = p.suffix.lower()
        if ext in image_exts:
            images[stem] = p
        elif ext == ".txt" and stem != "classes":
            labels[stem] = p

    img_stems = set(images.keys())
    lbl_stems = set(labels.keys())

    matched = sorted(list(img_stems & lbl_stems))
    orphaned_imgs = [images[s] for s in sorted(list(img_stems - lbl_stems))]
    orphaned_lbls = [labels[s] for s in sorted(list(lbl_stems - img_stems))]

    return PairValidationResult(
        matched_stems=matched,
        orphaned_images=orphaned_imgs,
        orphaned_labels=orphaned_lbls,
    )
