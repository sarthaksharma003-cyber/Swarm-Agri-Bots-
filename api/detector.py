"""Image validation, decoding, and YOLO inference service."""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from fastapi import HTTPException, status

from api.config import (
    ALLOWED_CONTENT_TYPES,
    ALLOWED_FILE_EXTENSIONS,
    DEFAULT_IMAGE_SIZE,
    MAX_UPLOAD_SIZE_BYTES,
    VERIFIED_CLASS_MAPPING,
)
from api.schemas import (
    BoundingBox,
    DetectionItem,
    DetectionResponse,
    ImageDimensions,
)

logger = logging.getLogger("SwarmAgriAPI.Detector")


def validate_and_decode_image(
    file_bytes: bytes,
    filename: Optional[str] = None,
    content_type: Optional[str] = None,
) -> Tuple[np.ndarray, ImageDimensions]:
    """Validate file size, extension, content type and decode into an OpenCV raster image.
    
    Raises HTTPException with informative status codes and error messages
    if validation fails or decoding fails.
    """
    # 1. Validate non-empty payload
    if not file_bytes or len(file_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty (0 bytes). Please upload a valid image file."
        )

    # 2. Enforce upload size limit
    if len(file_bytes) > MAX_UPLOAD_SIZE_BYTES:
        max_mb = MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"Uploaded file ({len(file_bytes)} bytes) exceeds the maximum allowed limit of {max_mb} MB."
        )

    # 3. Validate content type if explicitly provided
    if content_type:
        clean_ct = content_type.lower().split(";")[0].strip()
        # Disallow non-image media types
        if clean_ct and not clean_ct.startswith("image/"):
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=f"Invalid media type '{clean_ct}'. Expected an image (e.g. image/jpeg, image/png)."
            )

    # 4. Validate file extension if filename is provided
    if filename:
        ext = Path(filename).suffix.lower()
        if ext and ext not in ALLOWED_FILE_EXTENSIONS:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=f"Unsupported file extension '{ext}'. Allowed extensions: {sorted(list(ALLOWED_FILE_EXTENSIONS))}"
            )

    # 5. Decode image using OpenCV
    try:
        np_arr = np.frombuffer(file_bytes, np.uint8)
        img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    except Exception as e:
        logger.error(f"Image decode exception: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Corrupted or malformed image data. Could not decode with OpenCV."
        )

    if img_bgr is None or img_bgr.size == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to decode image data. File contents do not represent a valid raster image."
        )

    height, width = img_bgr.shape[:2]
    channels = img_bgr.shape[2] if len(img_bgr.shape) == 3 else 1

    dimensions = ImageDimensions(
        width=int(width),
        height=int(height),
        channels=int(channels)
    )

    return img_bgr, dimensions


def run_yolo_inference(
    model: Any,
    img_bgr: np.ndarray,
    conf_threshold: float,
    class_mapping: Optional[Dict[int, str]] = None,
    imgsz: int = DEFAULT_IMAGE_SIZE,
) -> Tuple[List[DetectionItem], float]:
    """Execute YOLO prediction and map boxes to verified agricultural classes."""
    classes = class_mapping or VERIFIED_CLASS_MAPPING

    t0 = time.perf_counter()
    try:
        results = model.predict(
            source=img_bgr,
            conf=conf_threshold,
            imgsz=imgsz,
            verbose=False,
        )
    except Exception as e:
        logger.error(f"Inference execution failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model inference failed: {str(e)}"
        )
    inference_time_ms = round((time.perf_counter() - t0) * 1000.0, 2)

    detections: List[DetectionItem] = []
    if results and len(results) > 0:
        res = results[0]
        if hasattr(res, "boxes") and res.boxes is not None:
            for box in res.boxes:
                # Handle PyTorch tensors, numpy arrays, or python lists/scalars
                try:
                    # Extract class id
                    raw_cls = box.cls
                    if hasattr(raw_cls, "item"):
                        cls_val = int(raw_cls.item())
                    elif isinstance(raw_cls, (list, tuple)):
                        elem = raw_cls[0]
                        cls_val = int(elem.item() if hasattr(elem, "item") else elem)
                    else:
                        cls_val = int(raw_cls)

                    # Extract confidence
                    raw_conf = box.conf
                    if hasattr(raw_conf, "item"):
                        conf_val = float(raw_conf.item())
                    elif isinstance(raw_conf, (list, tuple)):
                        elem = raw_conf[0]
                        conf_val = float(elem.item() if hasattr(elem, "item") else elem)
                    else:
                        conf_val = float(raw_conf)

                    # Extract coordinates xyxy
                    raw_xyxy = box.xyxy
                    if hasattr(raw_xyxy, "tolist"):
                        coords_list = raw_xyxy.tolist()
                    else:
                        coords_list = list(raw_xyxy)
                    if coords_list and isinstance(coords_list[0], (list, tuple)):
                        coords = coords_list[0]
                    else:
                        coords = coords_list
                except Exception as parse_err:
                    logger.warning(f"Failed parsing box item: {parse_err}")
                    continue

                x1 = round(float(coords[0]), 2)
                y1 = round(float(coords[1]), 2)
                x2 = round(float(coords[2]), 2)
                y2 = round(float(coords[3]), 2)
                w = round(max(0.0, x2 - x1), 2)
                h = round(max(0.0, y2 - y1), 2)
                cx = round(x1 + (w / 2.0), 2)
                cy = round(y1 + (h / 2.0), 2)

                # Map class ID using verified mapping (0: crop, 1: weed)
                cname = classes.get(cls_val, "weed" if cls_val == 1 else ("crop" if cls_val == 0 else f"class_{cls_val}"))

                bbox = BoundingBox(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    width=w,
                    height=h,
                    center_x=cx,
                    center_y=cy,
                )

                detections.append(
                    DetectionItem(
                        class_id=cls_val,
                        class_name=cname,
                        confidence=round(conf_val, 4),
                        bbox=bbox,
                    )
                )

    return detections, inference_time_ms
