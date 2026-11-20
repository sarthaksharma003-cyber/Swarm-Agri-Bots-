"""Visual preview generation module for SwarmAgri AI dataset."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
from PIL import Image, ImageDraw, ImageFont

from .config import CLASS_COLORS, CLASS_COLORS_RGBA
from .validator import BoundingBox, LabelValidationResult


def draw_bounding_boxes(
    image: Image.Image,
    boxes: List[BoundingBox],
    class_names: Dict[int, str],
    line_width: int = 3,
) -> Image.Image:
    """Render bounding boxes and label badges onto a PIL image.
    
    Args:
        image: Source PIL image.
        boxes: List of BoundingBox objects.
        class_names: Mapping of class_id to human-readable class name.
        line_width: Box stroke width.
        
    Returns:
        New PIL image with rendered annotations.
    """
    annotated = image.copy().convert("RGBA")
    draw = ImageDraw.Draw(annotated)
    w_img, h_img = image.size

    # Attempt to load a default font
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for box in boxes:
        cid = box.class_id
        color_rgb = CLASS_COLORS.get(cid, (255, 255, 0))
        color_rgba = CLASS_COLORS_RGBA.get(cid, (color_rgb[0], color_rgb[1], color_rgb[2], 60))

        # Convert normalized coordinates to pixel coordinates
        x1 = max(0, min(w_img - 1, int(round(box.x_min * w_img))))
        y1 = max(0, min(h_img - 1, int(round(box.y_min * h_img))))
        x2 = max(0, min(w_img - 1, int(round(box.x_max * w_img))))
        y2 = max(0, min(h_img - 1, int(round(box.y_max * h_img))))

        # Draw semi-transparent fill
        draw.rectangle([x1, y1, x2, y2], fill=color_rgba)
        # Draw solid bounding rectangle outline
        draw.rectangle([x1, y1, x2, y2], outline=color_rgb, width=line_width)

        # Draw label badge
        label_text = f"{cid}: {class_names.get(cid, f'class_{cid}')}"
        if font:
            bbox = font.getbbox(label_text)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
        else:
            text_w = len(label_text) * 7
            text_h = 12

        pad = 3
        badge_y1 = max(0, y1 - text_h - 2 * pad)
        badge_y2 = badge_y1 + text_h + 2 * pad
        badge_x2 = min(w_img, x1 + text_w + 2 * pad)

        # Draw badge background
        draw.rectangle([x1, badge_y1, badge_x2, badge_y2], fill=color_rgb)
        # Draw badge text in white or dark
        draw.text((x1 + pad, badge_y1 + pad), label_text, fill=(255, 255, 255), font=font)

    return annotated.convert("RGB")


def create_annotated_preview(
    image_path: Path,
    boxes: List[BoundingBox],
    class_names: Dict[int, str],
    output_path: Path,
    header_text: Optional[str] = None
) -> Path:
    """Create and save an annotated preview of an image with its bounding boxes."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(image_path) as img:
        annotated = draw_bounding_boxes(img, boxes, class_names)
        
        if header_text:
            # Add an informative header bar at the top
            w, h = annotated.size
            banner_h = 24
            canvas = Image.new("RGB", (w, h + banner_h), color=(20, 24, 33))
            canvas.paste(annotated, (0, banner_h))
            draw = ImageDraw.Draw(canvas)
            try:
                font = ImageFont.load_default()
            except Exception:
                font = None
            draw.text((8, 6), header_text, fill=(240, 240, 240), font=font)
            annotated = canvas

        annotated.save(output_path, "JPEG", quality=95)
    return output_path


def create_preview_montage(
    preview_paths: List[Path],
    output_path: Path,
    cols: int = 4,
    thumb_size: Tuple[int, int] = (300, 300)
) -> Path:
    """Create a contact-sheet montage of preview images."""
    if not preview_paths:
        return output_path

    rows = (len(preview_paths) + cols - 1) // cols
    w_thumb, h_thumb = thumb_size
    margin = 8
    canvas_w = cols * w_thumb + (cols + 1) * margin
    canvas_h = rows * h_thumb + (rows + 1) * margin

    montage = Image.new("RGB", (canvas_w, canvas_h), color=(15, 20, 25))

    for idx, p in enumerate(preview_paths):
        r = idx // cols
        c = idx % cols
        x = margin + c * (w_thumb + margin)
        y = margin + r * (h_thumb + margin)
        with Image.open(p) as img:
            thumb = img.resize((w_thumb, h_thumb), Image.Resampling.LANCZOS)
            montage.paste(thumb, (x, y))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    montage.save(output_path, "JPEG", quality=92)
    return output_path
