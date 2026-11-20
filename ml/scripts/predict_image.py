#!/usr/bin/env python3
"""Prediction interface for trained YOLO11n on agricultural images."""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("PredictYOLO")


def predict_single_image(
    image_path: Path,
    model_path: Path = Path("ml/models/yolo11n/best.pt"),
    conf_threshold: float = 0.25,
    imgsz: int = 512,
    device: str = "auto",
    output_dir: Path = Path("ml/reports/training/predictions"),
) -> Dict[str, Any]:
    """Run object detection inference on a single image.
    
    Returns structured prediction metadata including class names,
    confidence scores, coordinates, latency, and saved image path.
    """
    image_path = Path(image_path).resolve()
    model_path = Path(model_path).resolve()
    output_dir = Path(output_dir).resolve()

    # Graceful error handling for missing inputs
    if not image_path.exists():
        logger.error(f"Image not found at: {image_path}")
        raise FileNotFoundError(f"Input image not found: {image_path}")

    if not model_path.exists():
        logger.error(f"Model checkpoint not found at: {model_path}")
        raise FileNotFoundError(
            f"Trained model checkpoint not found at: {model_path}. "
            "Please run training first (e.g., via ml/scripts/train_yolo.py)."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO

    model = YOLO(str(model_path))

    # Resolve device
    dev = device
    if dev == "auto":
        import torch
        dev = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Measure inference timing
    t0 = time.perf_counter()
    results = model.predict(
        source=str(image_path),
        conf=conf_threshold,
        imgsz=imgsz,
        device=dev,
        save=False,
        verbose=False,
    )
    inference_time_ms = round((time.perf_counter() - t0) * 1000.0, 2)

    res = results[0]
    class_names = res.names

    detections: List[Dict[str, Any]] = []
    if res.boxes is not None:
        for box in res.boxes:
            cid = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            xyxy = [round(float(c), 2) for c in box.xyxy[0].tolist()]
            cname = class_names.get(cid, f"class_{cid}")

            detections.append({
                "class_id": cid,
                "class_name": cname,
                "confidence": round(conf, 4),
                "bbox_xyxy": xyxy,
            })

    # Save annotated preview image
    annotated_bgr = res.plot()
    out_filename = f"pred_{image_path.stem}.jpg"
    out_path = output_dir / out_filename
    cv2.imwrite(str(out_path), annotated_bgr)

    return {
        "image_path": str(image_path),
        "model_path": str(model_path),
        "confidence_threshold": conf_threshold,
        "inference_time_ms": inference_time_ms,
        "boxes_detected": len(detections),
        "detections": detections,
        "annotated_image_path": str(out_path),
    }


def main():
    parser = argparse.ArgumentParser(description="Run YOLO11n object detection on an image.")
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to target image file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ml/models/yolo11n/best.pt",
        help="Path to trained YOLO11n model weights (default: ml/models/yolo11n/best.pt)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold (default: 0.25)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=512,
        help="Image size in pixels (default: 512)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Compute device (default: auto)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="ml/reports/training/predictions",
        help="Directory to save annotated preview (default: ml/reports/training/predictions)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print pure JSON output to stdout",
    )
    args = parser.parse_args()

    try:
        output = predict_single_image(
            image_path=Path(args.image),
            model_path=Path(args.model),
            conf_threshold=args.conf,
            imgsz=args.imgsz,
            device=args.device,
            output_dir=Path(args.output_dir),
        )
        if args.json:
            print(json.dumps(output, indent=2))
        else:
            print("\n==================================================")
            print("SwarmAgri AI — YOLO11n Detection Output")
            print("==================================================")
            print(f"Source Image: {output['image_path']}")
            print(f"Model Checkpoint: {output['model_path']}")
            print(f"Inference Latency: {output['inference_time_ms']} ms")
            print(f"Total Objects Detected: {output['boxes_detected']}")
            for idx, d in enumerate(output["detections"], 1):
                print(f"  [{idx}] {d['class_name']} (ID {d['class_id']}): "
                      f"conf={d['confidence']:.2f}, bbox={d['bbox_xyxy']}")
            print(f"Annotated Image Saved: {output['annotated_image_path']}")
            print("==================================================")
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
