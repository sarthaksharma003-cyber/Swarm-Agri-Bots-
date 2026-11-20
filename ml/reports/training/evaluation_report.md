# SwarmAgri AI — YOLO11n Test Split Evaluation Report

## Evaluation Overview
- **Model Checkpoint**: `/home/smasher/Desktop/swarmagri-ai/ml/models/yolo11n/run_10epochs/weights/best.pt`
- **Evaluated Split**: `test` (Held-out Test Split)
- **Dataset**: `/home/smasher/Desktop/swarmagri-ai/data/processed/yolo_dataset/data.yaml`
- **Image Size**: 512 x 512
- **Compute Hardware**: CPU (Intel / x86_64) ()

## Overall Detection Performance
| Metric | Score |
|:---|:---:|
| **Precision (P)** | **0.7682** |
| **Recall (R)** | **0.7631** |
| **mAP @ 0.50** | **0.8155** |
| **mAP @ 0.50:0.95** | **0.4972** |

## Per-Class Performance
| Class | Precision | Recall | mAP@0.50 | mAP@0.50:0.95 |
|:---|:---:|:---:|:---:|:---:|
| **crop** | 0.7070 | 0.7562 | 0.7748 | 0.4676 |
| **weed** | 0.8295 | 0.7701 | 0.8563 | 0.5267 |


## Inference Speed & Latency Benchmark
- **Preprocess**: 1.88 ms
- **Inference**: 47.69 ms
- **Postprocess**: 0.79 ms
- **Total Latency**: **50.37 ms / image**
- **Throughput**: **19.9 FPS**
- **Hardware & Measurement**: CPU (Intel / x86_64) (Synchronized batch-1 forward pass under PyTorch 2.14.1+cpu)

## Failure Mode Analysis & Critical Limitations
1. **Sparse Mixed-Class Co-occurrence**:
   The underlying dataset contains only 2 images with simultaneous crop and weed presence (`agri_0_1852`, `agri_0_3820`). While test performance is strong on isolated crop and weed images, **this model must NOT be assumed fully reliable in dense mixed fields** without targeted field-level validation and synthetic co-occurrence augmentation (Mosaic/Copy-Paste).
2. **Weed Recall & Missed Weeds**:
   In robotic swarm weeding, false negatives (missed weeds) allow weeds to compete with crops and produce seeds. Confidence thresholds on swarm rovers should be tuned for high weed recall (e.g., `conf=0.20-0.25`).
