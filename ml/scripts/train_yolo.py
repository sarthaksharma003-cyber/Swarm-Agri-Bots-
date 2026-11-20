#!/usr/bin/env python3
"""Reproducible YOLO11n training pipeline for SwarmAgri AI."""

import argparse
import json
import logging
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TrainYOLO")


def detect_device(requested_device: Optional[str] = "auto") -> str:
    """Detect available compute hardware; fall back safely to CPU if CUDA is unavailable."""
    if requested_device and requested_device.lower() != "auto":
        req = requested_device.lower()
        if "cuda" in req:
            try:
                import torch
                if not torch.cuda.is_available():
                    logger.warning("CUDA requested but not available. Falling back to CPU.")
                    return "cpu"
                return req
            except ImportError:
                return "cpu"
        return req

    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"CUDA-capable GPU detected: {gpu_name}. Using device='cuda:0'.")
            return "cuda:0"
    except ImportError:
        pass

    logger.info("No CUDA GPU available. Using CPU compute (device='cpu').")
    return "cpu"


def verify_data_yaml(data_yaml_path: Path) -> Dict[str, Any]:
    """Verify that data.yaml exists and contains required dataset specifications."""
    if not data_yaml_path.exists():
        raise FileNotFoundError(f"Dataset configuration not found at: {data_yaml_path}")

    with open(data_yaml_path, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    required_keys = ["train", "val", "nc", "names"]
    for k in required_keys:
        if k not in data_cfg:
            raise ValueError(f"Missing required key '{k}' in {data_yaml_path}")

    names = data_cfg["names"]
    if isinstance(names, dict):
        if names.get(0) != "crop" or names.get(1) != "weed":
            logger.warning(f"Unexpected class names in data.yaml: {names}")
    elif isinstance(names, list):
        if len(names) < 2 or names[0] != "crop" or names[1] != "weed":
            logger.warning(f"Unexpected class list in data.yaml: {names}")

    return data_cfg


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Train YOLO11n on SwarmAgri AI dataset.")
    parser.add_argument(
        "--data",
        type=str,
        default="data/processed/yolo_dataset/data.yaml",
        help="Path to data.yaml dataset definition",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="yolo11n.pt",
        help="Pretrained model weights (default: yolo11n.pt)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Maximum training epochs (default: 50)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=4,
        help="Batch size (default: 4)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=512,
        help="Image size in pixels (default: 512)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience (default: 10)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Number of DataLoader worker processes (default: 2)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Compute device: 'auto', 'cpu', 'cuda', etc. (default: auto)",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="ml/models/yolo11n",
        help="Directory to save model runs (default: ml/models/yolo11n)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Run name (default: timestamped train_<YYYYMMDD_HHMMSS>)",
    )
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="Whether to overwrite existing run directory",
    )
    return parser.parse_args(args)


def train_yolo(
    data_yaml: Path,
    weights: str = "yolo11n.pt",
    epochs: int = 50,
    batch_size: int = 4,
    imgsz: int = 512,
    patience: int = 10,
    seed: int = 42,
    workers: int = 2,
    device: str = "auto",
    project_dir: Path = Path("ml/models/yolo11n"),
    run_name: Optional[str] = None,
    exist_ok: bool = False,
) -> Dict[str, Any]:
    """Execute YOLO11n fine-tuning on the agricultural dataset."""
    project_dir = Path(project_dir).resolve()
    data_yaml = Path(data_yaml).resolve()

    # Verify data config
    data_cfg = verify_data_yaml(data_yaml)

    # Determine compute device
    chosen_device = detect_device(device)

    # Assign deterministic run name if not provided
    if not run_name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"run_{timestamp}"

    run_dir = project_dir / run_name
    if run_dir.exists() and not exist_ok:
        raise FileExistsError(
            f"Run directory already exists at {run_dir}. Use --exist-ok or specify a unique --name."
        )

    logger.info("==================================================")
    logger.info("Starting YOLO11n Training Pipeline")
    logger.info("==================================================")
    logger.info(f"Model Pretrained Weights: {weights}")
    logger.info(f"Dataset Config: {data_yaml}")
    logger.info(f"Class Names: {data_cfg.get('names')}")
    logger.info(f"Target Epochs: {epochs} (Early Stopping Patience: {patience})")
    logger.info(f"Batch Size: {batch_size} | Image Size: {imgsz}x{imgsz}")
    logger.info(f"Device: {chosen_device} | Seed: {seed} | Workers: {workers}")
    logger.info(f"Run Output Directory: {run_dir}")

    start_time = time.time()

    # Import Ultralytics YOLO
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("Ultralytics is not installed. Please install ultralytics in the virtual environment.")
        raise

    model = YOLO(weights)

    # Attach memory safety monitor callback
    try:
        import psutil
        def memory_safety_monitor(trainer):
            mem = psutil.virtual_memory()
            avail_mb = mem.available / (1024 * 1024)
            if avail_mb < 250.0:
                logger.error(f"CRITICAL: Available RAM is only {avail_mb:.1f} MB (< 250 MB). Aborting training safely.")
                raise MemoryError(f"System memory critical ({avail_mb:.1f} MB remaining). Aborted to prevent crash.")
            logger.info(f"[Memory Monitor] Epoch {trainer.epoch + 1}: RAM used={mem.percent}%, available={avail_mb:.0f} MB")
        model.add_callback("on_train_epoch_end", memory_safety_monitor)
    except ImportError:
        logger.warning("psutil not available; running without runtime RAM monitor.")

    # Train model
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        patience=patience,
        seed=seed,
        workers=workers,
        device=chosen_device,
        project=str(project_dir),
        name=run_name,
        plots=True,
        save=True,
        deterministic=True,
        verbose=True,
    )

    elapsed_time_sec = round(time.time() - start_time, 2)
    logger.info(f"Training completed in {elapsed_time_sec} seconds.")

    # Locate generated checkpoints
    actual_run_dir = Path(results.save_dir) if hasattr(results, "save_dir") else run_dir
    best_pt = actual_run_dir / "weights" / "best.pt"
    last_pt = actual_run_dir / "weights" / "last.pt"

    # Safely link or copy primary checkpoints to project_dir
    primary_best = project_dir / "best.pt"
    primary_last = project_dir / "last.pt"

    if best_pt.exists():
        shutil.copy2(best_pt, primary_best)
        logger.info(f"Saved primary best checkpoint to: {primary_best}")
    if last_pt.exists():
        shutil.copy2(last_pt, primary_last)
        logger.info(f"Saved primary last checkpoint to: {primary_last}")

    # Parse per-epoch metrics from results.csv
    import csv
    epoch_metrics = []
    results_csv = actual_run_dir / "results.csv"
    if results_csv.exists():
        with open(results_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                clean_row = {k.strip(): v.strip() for k, v in row.items()}
                try:
                    ep = int(clean_row.get("epoch", 0))
                    epoch_metrics.append({
                        "epoch": ep,
                        "train_box_loss": round(float(clean_row.get("train/box_loss", 0)), 4),
                        "train_cls_loss": round(float(clean_row.get("train/cls_loss", 0)), 4),
                        "train_dfl_loss": round(float(clean_row.get("train/dfl_loss", 0)), 4),
                        "val_box_loss": round(float(clean_row.get("val/box_loss", 0)), 4),
                        "val_cls_loss": round(float(clean_row.get("val/cls_loss", 0)), 4),
                        "val_dfl_loss": round(float(clean_row.get("val/dfl_loss", 0)), 4),
                        "precision": round(float(clean_row.get("metrics/precision(B)", 0)), 4),
                        "recall": round(float(clean_row.get("metrics/recall(B)", 0)), 4),
                        "map50": round(float(clean_row.get("metrics/mAP50(B)", 0)), 4),
                        "map50_95": round(float(clean_row.get("metrics/mAP50-95(B)", 0)), 4),
                    })
                except (ValueError, TypeError):
                    continue

    # Compile training summary metrics
    training_summary = {
        "model_architecture": "yolo11n",
        "pretrained_weights": weights,
        "dataset_yaml": str(data_yaml),
        "classes": data_cfg.get("names"),
        "training_parameters": {
            "epochs_requested": epochs,
            "epochs_completed": len(epoch_metrics),
            "batch_size": batch_size,
            "image_size": imgsz,
            "patience": patience,
            "seed": seed,
            "device": chosen_device,
            "workers": workers,
        },
        "timing": {
            "elapsed_seconds": elapsed_time_sec,
            "elapsed_minutes": round(elapsed_time_sec / 60.0, 2),
        },
        "artifacts": {
            "run_dir": str(actual_run_dir),
            "best_checkpoint": str(best_pt) if best_pt.exists() else None,
            "last_checkpoint": str(last_pt) if last_pt.exists() else None,
            "primary_best": str(primary_best) if primary_best.exists() else None,
            "results_csv": str(results_csv) if results_csv.exists() else None,
        },
        "epoch_metrics": epoch_metrics,
    }

    # Save summary report
    reports_dir = Path("ml/reports/training").resolve()
    reports_dir.mkdir(parents=True, exist_ok=True)

    summary_json = reports_dir / "training_summary.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(training_summary, f, indent=2)

    # Format epoch rows for markdown
    epoch_table_rows = ""
    for em in epoch_metrics:
        epoch_table_rows += (
            f"| {em['epoch']} "
            f"| {em['train_box_loss']:.4f} / {em['train_cls_loss']:.4f} "
            f"| {em['val_box_loss']:.4f} / {em['val_cls_loss']:.4f} "
            f"| {em['precision']:.4f} "
            f"| {em['recall']:.4f} "
            f"| **{em['map50']:.4f}** "
            f"| **{em['map50_95']:.4f}** |\n"
        )

    summary_md = reports_dir / "training_summary.md"
    md_content = f"""# SwarmAgri AI — YOLO11n Training Summary

## Training Configuration
- **Model**: YOLO11n (Pretrained: `{weights}`)
- **Dataset**: `{data_yaml}`
- **Classes**: `{data_cfg.get('names')}`
- **Epochs Requested**: {epochs} (Completed: {len(epoch_metrics)}, Patience: {patience})
- **Batch Size**: {batch_size}
- **Image Size**: {imgsz} x {imgsz}
- **Compute Device**: `{chosen_device}`
- **Random Seed**: `{seed}`
- **Duration**: {elapsed_time_sec} s ({round(elapsed_time_sec / 60.0, 2)} min)

## Model Checkpoints
- **Run Directory**: `{actual_run_dir}`
- **Best Weights**: `{best_pt}`
- **Last Weights**: `{last_pt}`
- **Primary Checkpoint**: `{primary_best}`

## Per-Epoch Metrics Progression
| Epoch | Train (Box/Cls) | Val (Box/Cls) | Precision | Recall | mAP@50 | mAP@50-95 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
{epoch_table_rows}
"""
    with open(summary_md, "w", encoding="utf-8") as f:
        f.write(md_content)

    logger.info(f"Training report saved to {summary_json} and {summary_md}")
    return training_summary


def main():
    args = parse_args()
    try:
        train_yolo(
            data_yaml=Path(args.data),
            weights=args.weights,
            epochs=args.epochs,
            batch_size=args.batch,
            imgsz=args.imgsz,
            patience=args.patience,
            seed=args.seed,
            workers=args.workers,
            device=args.device,
            project_dir=Path(args.project),
            run_name=args.name,
            exist_ok=args.exist_ok,
        )
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
