# SwarmAgri AI - Dataset Validation & Preparation Pipeline

## 1. Overview & Objectives

**SwarmAgri AI** develops autonomous, distributed robotic swarms for precision agriculture, targeting real-time crop monitoring and robotic weed removal. A foundational requirement for deploying lightweight vision models (e.g., YOLOv8/YOLOv11 nano/small) on embedded swarm rovers and edge drones is a clean, rigorous, and leakage-free dataset pipeline.

This repository provides an automated, end-to-end dataset validation, auditing, deduplication, and YOLO formatting pipeline for agricultural crop and weed imagery.

---

## 2. Dataset Provenance & Verified Class Mapping

### 2.1 Source Data
- **Raw Archive**: `data/raw/archive.zip` (2,601 files).
- **Extracted Source**: `data/processed/source/agri_data/data/` (1,300 JPEG images and 1,300 YOLO `.txt` annotation files).
- **Extracted Source Metadata**: `data/processed/source/classes.txt`.

### 2.2 Class Verification (No Invented Classes)
In strict accordance with the project requirements, class names were **not invented or guessed**. The class mapping was discovered and verified directly from the dataset's original metadata:
1. `classes.txt` inside `data/raw/archive.zip` contains:
   ```text
   crop
   weed
   ```
2. Following standard YOLO convention (zero-indexed line mapping):
   - **Class ID 0**: `crop`
   - **Class ID 1**: `weed`
3. Cross-verification against all **2,072 bounding boxes** across all 1,300 label files confirmed that observed class IDs are exclusively `{0, 1}`. No undefined or unrecognized class IDs exist.

| Class ID | Verified Class Name | Bounding Box Count | Overall Percentage |
|:---:|:---:|:---:|:---:|
| **0** | `crop` | 1,212 | 58.49% |
| **1** | `weed` | 860 | 41.51% |
| **Total** | - | **2,072** | **100.0%** |

---

## 3. Data Validation & Quality Audit

The validation pipeline enforces comprehensive integrity checks across images, annotations, and file associations:

### 3.1 Image Integrity & Decoding
- Every JPEG file was decoded using both **OpenCV** (`cv2.imread()`) and **Pillow** (`Image.verify()` and `Image.load()`).
- **Total Images**: 1,300.
- **Corrupted / Truncated Files**: 0.
- **OpenCV Decoding Success**: 1,300 / 1,300 (100.0%, all returning shape `(512, 512, 3)`).
- **Resolution**: 100% of images are exactly `512 x 512` pixels.
- **Color Format**: 100% 3-channel RGB (24-bit).
- **File Sizes**: Min: 23.3 KB, Max: 219.8 KB, Mean: 84.8 KB, Median: 80.8 KB.

### 3.2 YOLO Annotation Validation
- Every annotation row was validated for:
  - Exactly 5 whitespace-separated fields (`class_id`, `x_center`, `y_center`, `width`, `height`).
  - Non-negative integer class IDs.
  - Finite normalized coordinates (`math.isfinite()`).
  - Strictly positive widths and heights (`w > 0`, `h > 0`).
  - Spatial containment within normalized image bounds `[0.0, 1.0]`.
- **Total Valid Annotations**: 2,072 / 2,072 (100.0%).
- **Invalid / Malformed Rows**: 0.
- **Floating-Point Rounding Analysis**: 75 annotations exhibited micro-overflows of up to `5.0e-7` (half a millionth, or ~0.00025 pixels on a 512px raster) resulting from 6-digit decimal quantization in the source text files (e.g. `0.500977 + 0.998047 / 2 = 1.0000005`).
- **Precision Export**: During export to `data/processed/yolo_dataset/`, all coordinates are clamped strictly within `[0.0, 1.0]`, guaranteeing zero out-of-bounds warnings during model training.

### 3.3 File Pair Correspondence
- **Matching Image-Label Pairs**: 1,300.
- **Orphaned Images (Images without Labels)**: 0.
- **Orphaned Labels (Labels without Images)**: 0.

---

## 4. Duplicate & Near-Duplicate Analysis

### 4.1 Exact Duplicates
- **File Byte Hashing (SHA-256)**: 1,300 unique hashes (0 exact duplicate files).
- **Decoded Pixel Buffer Hashing (SHA-256)**: 1,300 unique hashes (0 exact pixel duplicates).

### 4.2 Perceptual Near-Duplicates (dHash)
Agricultural swarm platforms capture successive frames during field sweeps. To evaluate frame redundancy and prevent evaluation bias:
- **Difference Hash (dHash, 64-bit)**: Computed across all images.
- **Identical dHash Groups (Hamming Distance = 0)**: 95 perceptual groups.
- **Near-Duplicate Pairs (Hamming Distance <= 2)**: 494 pairs (similarity >= 96.88%).

---

## 5. Dataset Splitting & Data Leakage Prevention

### 5.1 Source Split Audit
An inspection of `data/raw/archive.zip` and `data/processed/source/` confirmed that **no predefined splits** (such as `train/`, `val/`, `test/` subfolders or `train.txt` split files) existed in the source distribution.

### 5.2 Reproducible 80/10/10 Stratification
A deterministic, reproducible split was generated using fixed seed `42`:
- **Stratification Target**: Proportional representation of single-class and multi-class images.
- **Cluster Isolation (Leakage Prevention)**: Exact duplicate images and connected components are clustered using Disjoint Set Union (DSU) and assigned atomically to the same split.
- **Cross-Split Duplicate Leakage**: **0.0%**.
- **Split Overlap**: `Train ∩ Val = ∅`, `Train ∩ Test = ∅`, `Val ∩ Test = ∅`.

### 5.3 Split Distribution Summary

| Split | Image Count | Image Share | Class 0 (`crop`) | Class 1 (`weed`) | Total Boxes | Boxes/Image |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Train** | 1,042 | 80.15% | 945 | 691 | 1,636 | 1.57 |
| **Validation** | 129 | 9.92% | 117 | 82 | 199 | 1.54 |
| **Test** | 129 | 9.92% | 150 | 87 | 237 | 1.84 |
| **Total** | **1,300** | **100.0%** | **1,212** | **860** | **2,072** | **1.59** |

---

## 6. Directory Structure

```
swarmagri-ai/
├── data/
│   ├── raw/
│   │   └── archive.zip                        # Original untouched source archive
│   └── processed/
│       ├── source/
│       │   ├── classes.txt                    # Original extracted metadata
│       │   └── agri_data/data/                # Original extracted 1,300 images & labels
│       └── yolo_dataset/                      # Prepared YOLO dataset
│           ├── data.yaml                      # YOLO configuration with verified classes
│           ├── images/
│           │   ├── train/                     # 1,042 training images
│           │   ├── val/                       # 129 validation images
│           │   └── test/                      # 129 test images
│           └── labels/
│               ├── train/                     # 1,042 training labels
│               ├── val/                       # 129 validation labels
│               └── test/                      # 129 test labels
├── ml/
│   ├── prepare_dataset.py                     # Convenience top-level CLI runner
│   ├── scripts/
│   │   ├── prepare_dataset.py                 # Full dataset pipeline runner
│   │   ├── validate_dataset.py                # Standalone validation runner (OpenCV + PIL + YOLO)
│   │   └── inspect_metadata.py                # Metadata discovery and class verifier
│   ├── src/
│   │   └── dataset/
│   │       ├── config.py                      # Pipeline configuration & constants
│   │       ├── metadata.py                    # Class discovery & verification
│   │       ├── validator.py                   # Image & YOLO annotation validation (OpenCV + PIL)
│   │       ├── duplicates.py                  # Exact & perceptual hash deduplication
│   │       ├── splitter.py                    # Stratified splitting & leakage prevention
│   │       ├── visualizer.py                  # Bounding box rendering & montages
│   │       └── pipeline.py                    # End-to-end orchestration logic
│   ├── reports/
│   │   ├── data_quality_report.md             # Image decoding & quality audit (Markdown)
│   │   ├── data_quality_report.json           # Machine-readable quality metrics
│   │   ├── class_frequency_report.md          # Class distribution and ratios
│   │   ├── class_frequency_report.json        # Machine-readable class metrics
│   │   ├── image_dimension_report.md          # Dimensions, aspect ratios, file sizes
│   │   ├── image_dimension_report.json        # Machine-readable dimension metrics
│   │   ├── annotation_count_report.md         # Box density and relative area statistics
│   │   ├── annotation_count_report.json       # Machine-readable annotation metrics
│   │   ├── split_summary.md                   # Split details & leakage verification
│   │   ├── split_summary.json                 # Machine-readable split manifest
│   │   └── previews/                          # Visual preview samples
│   │       ├── preview_montage.jpg            # 4x4 composite contact sheet
│   │       └── preview_*.jpg                  # High-contrast annotated samples
│   └── tests/
│       ├── test_metadata.py                   # Metadata verification tests
│       ├── test_validator.py                  # OpenCV & PIL decoding and annotation tests
│       ├── test_duplicates.py                 # Hash & cluster tests
│       ├── test_splitter.py                   # Split reproducibility & leakage tests
│       └── test_pipeline.py                   # End-to-end pipeline integration tests
└── README.md                                  # Comprehensive documentation
```

---

## 7. YOLO Configuration (`data.yaml`)

The generated `data/processed/yolo_dataset/data.yaml` is fully compatible with Ultralytics YOLOv8/YOLOv11:

```yaml
path: /home/smasher/Desktop/swarmagri-ai/data/processed/yolo_dataset
train: images/train
val: images/val
test: images/test
nc: 2
names:
  0: crop
  1: weed
```

---

## 8. Pipeline Execution & Testing

### 8.1 Run the Full Preparation Pipeline
Execute the pipeline via the CLI script under `ml/scripts/`:
```bash
python3 ml/scripts/prepare_dataset.py --seed 42 --previews 16
```
(Alternatively: `python3 ml/prepare_dataset.py`)

### 8.2 Run Standalone Dataset Validation
Verify OpenCV decoding, label bounds, and class mappings:
```bash
python3 ml/scripts/validate_dataset.py
```

### 8.3 Inspect Dataset Metadata
Check raw archive and extracted class definitions:
```bash
python3 ml/scripts/inspect_metadata.py
```

### 8.4 Run the Automated Test Suite
Run the 36 unit and integration tests with pytest:
```bash
pytest ml/tests/ -v
```

All 36 tests validate:
- Accurate extraction and strict verification of class names from metadata.
- Rejection of undefined or invented class names.
- Complete decoding of valid JPEGs with both OpenCV and Pillow, and proper flagging of corrupted/truncated files.
- YOLO format validation (token counts, coordinate bounds, positive dimensions).
- Exact hash computation, dHash perceptual distance, and DSU clustering.
- Reproducible, deterministic splitting with 0.0% duplicate leakage.
- Full end-to-end pipeline integration.

---

## 9. Visual Inspection Previews

Visual previews are rendered in `ml/reports/previews/` with:
- **Green Bounding Boxes** (`#2ECC71`): Class 0 (`crop`).
- **Red Bounding Boxes** (`#E74C3C`): Class 1 (`weed`).
- High-contrast background badges for clear visibility under diverse lighting and soil conditions.
- Contact-sheet montage: `ml/reports/previews/preview_montage.jpg`.

---

## 10. Dataset Insights, Limitations & Swarm Operational Considerations

1. **Resolution Uniformity**:
   All 1,300 images are uniform `512 x 512` RGB. This matches standard YOLO training input resolutions (e.g. `imgsz=512` or `imgsz=640`) without severe aspect ratio distortion.
2. **Class Imbalance**:
   Crops account for 58.5% (1,212 boxes) while weeds account for 41.5% (860 boxes). Model loss functions should consider class weights or focal loss if weed recall is prioritized.
3. **Sparse Co-Occurrence**:
   633 images contain only crops, and 665 images contain only weeds. Only **2 images** (`agri_0_1852.jpeg` and `agri_0_3820.jpeg`) contain both crops and weeds in the same frame. For swarm robots operating in mixed fields, data augmentation (e.g. Mosaic, MixUp, Copy-Paste) will be critical during training.
4. **Near-Duplicate Field Traversal Frames**:
   While exact duplicate images do not exist, 95 identical perceptual dHash groups reflect sequential video frames captured along crop rows. The cluster-isolated splitting ensures that evaluations on validation and test sets remain robust and uninflated.

---

## 11. License & Usage

- **Source Dataset**: Agricultural crop and weed detection dataset (`agri_data`).
- **License**: Provided for research and development within the SwarmAgri AI autonomous precision agriculture initiative.
- **Source Preservation**: The original raw archive (`data/raw/archive.zip`) and extracted files (`data/processed/source/`) are preserved in their original state.

---

## 12. YOLO11n Training, Evaluation & Prediction Pipeline

### 12.1 Environment Setup
Create and activate the project virtual environment `.venv`:
```bash
# Create virtual environment
python3 -m venv .venv

# Install dependencies (CPU PyTorch + Ultralytics)
./.venv/bin/pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
./.venv/bin/pip install --no-cache-dir ultralytics pytest
```

### 12.2 Model Training
Fine-tune Ultralytics YOLO11n on the prepared agricultural dataset:
```bash
./.venv/bin/python ml/scripts/train_yolo.py \
    --data data/processed/yolo_dataset/data.yaml \
    --weights yolo11n.pt \
    --epochs 50 \
    --batch 4 \
    --imgsz 512 \
    --patience 10 \
    --seed 42 \
    --device auto
```
- **Checkpoints Output**: Saved under `ml/models/yolo11n/` (`best.pt`, `last.pt`, and timestamped run directory).
- **Run Summary**: Saved to `ml/reports/training/training_summary.json` and `.md`.

### 12.3 Held-Out Test Evaluation
Evaluate the trained checkpoint on the held-out test split (129 images):
```bash
./.venv/bin/python ml/scripts/evaluate_yolo.py \
    --model ml/models/yolo11n/best.pt \
    --data data/processed/yolo_dataset/data.yaml \
    --split test \
    --imgsz 512
```
- **Metrics Computed**: Precision, Recall, mAP@0.50, mAP@0.50:0.95, per-class crop/weed statistics, latency/FPS benchmark.
- **Reports Output**: Saved to `ml/reports/training/evaluation_report.md` and `.json`.

### 12.4 Single-Image Prediction Interface
Run object detection on any test or field image:
```bash
./.venv/bin/python ml/scripts/predict_image.py \
    --image data/processed/yolo_dataset/images/test/agri_0_1018.jpeg \
    --model ml/models/yolo11n/best.pt \
    --conf 0.25
```
- **Output**: Prints detected bounding boxes, classes, confidence scores, and latency.
- **Annotated Visuals**: Saved to `ml/reports/training/predictions/`.
- **JSON Flag**: Append `--json` for machine-readable stdout formatting.

### 12.5 Test Suite Execution
Run all 46 unit and integration tests (including training config & prediction error handling):
```bash
./.venv/bin/pytest ml/tests/ -v
```

### 12.6 Benchmark vs 10-Epoch Fine-Tuning Empirical Results

#### Checkpoint Organization
- **Preserved 1-Epoch Benchmark**: `ml/models/yolo11n/benchmark_epoch1_best.pt` (and run directory `ml/models/yolo11n/test_epoch1/`)
- **10-Epoch Run Directory**: `ml/models/yolo11n/run_10epochs/` (contains `best.pt`, `last.pt`, `results.csv`, `results.png`)
- **Primary Deployment Weights**: `ml/models/yolo11n/best.pt` (5.5 MB)

#### Held-Out Test Split Performance Comparison (129 Images, 237 Bounding Boxes)
| Evaluation Metric | 1-Epoch Baseline Benchmark | 10-Epoch Fine-Tuning | Absolute Gain |
|:---|:---:|:---:|:---:|
| **Overall Precision (P)** | 0.2711 | **0.7682** | +0.4971 (+183.4%) |
| **Overall Recall (R)** | 0.5836 | **0.7631** | +0.1795 (+30.8%) |
| **Overall mAP @ 0.50** | 0.3623 | **0.8155** | +0.4532 (+125.1%) |
| **Overall mAP @ 0.50:0.95** | 0.1934 | **0.4972** | +0.3038 (+157.1%) |
| *Crop mAP @ 0.50* | 0.4460 | **0.7748** | +0.3288 (+73.7%) |
| *Weed mAP @ 0.50* | 0.2787 | **0.8563** | +0.5776 (+207.2%) |
| *CPU Inference Latency* | 54.83 ms | **47.69 ms** | -7.14 ms (~19.9 FPS) |

#### 10-Epoch Validation Metric Progression
| Epoch | Train Box Loss | Train Cls Loss | Val Box Loss | Val Cls Loss | Precision | Recall | mAP@0.50 | mAP@0.50:0.95 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 1.3443 | 2.0742 | 1.4648 | 1.6611 | 0.5782 | 0.6558 | 0.6184 | 0.3164 |
| 2 | 1.3742 | 1.3687 | 1.5438 | 1.3799 | 0.6897 | 0.6247 | 0.7130 | 0.3911 |
| 3 | 1.3372 | 1.2756 | 1.5585 | 1.5303 | 0.5390 | 0.7016 | 0.5749 | 0.2974 |
| 4 | 1.3005 | 1.2092 | 1.3694 | 1.1668 | 0.8479 | 0.6780 | 0.7768 | 0.4558 |
| 5 | 1.2615 | 1.0957 | 1.4193 | 1.0934 | 0.7779 | 0.7406 | 0.8266 | 0.4814 |
| 6 | 1.2201 | 1.0121 | 1.2954 | 1.0726 | 0.8986 | 0.6924 | 0.8422 | 0.5191 |
| 7 | 1.1462 | 0.9460 | 1.2591 | 1.0739 | 0.7965 | 0.7663 | 0.8192 | 0.5121 |
| 8* | 1.1066 | 0.8751 | 1.2095 | 0.9499 | 0.8232 | 0.8246 | **0.8600** | **0.5404** |
| 9 | 1.1034 | 0.8157 | 1.2169 | 0.9708 | 0.8314 | 0.8334 | 0.8546 | 0.5457 |
| 10 | 1.0447 | 0.7773 | 1.1986 | 0.9647 | 0.7992 | 0.7948 | 0.8425 | 0.5393 |

*\*Epoch 8 achieved peak validation mAP@0.50 (0.8600) and was selected as the optimal model checkpoint (`best.pt`).*

---

## 13. Stage 4: FastAPI AI Inference & Decoupled Swarm Simulation Service

### 13.1 Service Architecture & Lifespan Management
The SwarmAgri AI inference service (`api/`) is built on **FastAPI** and modern asynchronous Python. It separates real-time computer vision from robot task allocation:
- **Lifespan Model Loading**: The fine-tuned YOLO11n weights ([best.pt](file:///home/smasher/Desktop/swarmagri-ai/ml/models/yolo11n/run_10epochs/weights/best.pt)) are loaded into memory exactly **once** during application startup, eliminating disk/model loading overhead during per-request inference.
- **Decoupled Swarm Coordination**: The perception endpoint (`POST /api/detect`) and task allocation endpoint (`POST /api/swarm/tasks`) are strictly decoupled. This architecture enables distributed robotic swarm nodes to feed multi-camera detections into simulation task pipelines, laying the foundation for future decentralized robot bidding (evaluating travel distance, battery state-of-charge, tool compatibility, and weed urgency).
- **Simulation Safety**: Physical actuation is disabled (`simulation_mode = True`, `hardware_dispatch = False`). The service strictly proposes simulation tasks and does not transmit commands to physical hardware.

### 13.2 API Endpoints Specification

| Method | Endpoint | Description | Key Parameters / Request | Key Response Fields |
|:---:|:---|:---|:---|:---|
| `GET` | `/health` | Reports service status, model loaded state, verified class mapping, and compute device. | None | `status`, `model_loaded`, `model_path`, `classes`, `device`, `version` |
| `POST` | `/api/detect` | Runs YOLO inference on an uploaded agricultural image. | Multipart file (`image/jpeg`, `image/png`, `image/webp`), optional query `conf` (default: 0.25) | `image_dimensions`, `confidence_threshold`, `inference_time_ms`, `total_detections`, `crop_count`, `weed_count`, `detections` (with pixel coordinates `bbox`) |
| `POST` | `/api/swarm/tasks` | Converts weed detections into simulation tasks with stable task IDs and bidding context. | JSON payload with `detections`, `field_id`, `grid_scale_m_per_px`, `urgency_multiplier` | `field_id`, `tasks_generated`, `tasks` (stable `task_id`, `priority`, coordinates, `bidding_context`), `crops_protected_count` |

### 13.3 Local Startup Instructions

To launch the FastAPI service locally on port 8000:

```bash
# 1. Activate the project virtual environment
source .venv/bin/activate

# 2. Start the Uvicorn ASGI server
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

The interactive Swagger UI documentation is accessible at:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

### 13.4 Example API Requests & CLI Usage

#### Health Check
```bash
curl -X GET "http://localhost:8000/health"
```
**Example Response**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_path": "/home/smasher/Desktop/swarmagri-ai/ml/models/yolo11n/run_10epochs/weights/best.pt",
  "classes": {"0": "crop", "1": "weed"},
  "device": "cpu",
  "version": "1.0.0",
  "timestamp": "2026-10-18T01:47:33.467440+00:00"
}
```

#### Object Detection (`POST /api/detect`)
```bash
curl -X POST "http://localhost:8000/api/detect?conf=0.25" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@data/processed/yolo_dataset/images/test/agri_0_1018.jpeg;type=image/jpeg"
```

#### Swarm Simulation Task Allocation (`POST /api/swarm/tasks`)
```bash
curl -X POST "http://localhost:8000/api/swarm/tasks" \
  -H "Content-Type: application/json" \
  -d '{
    "field_id": "plot_bravo_09",
    "image_id": "agri_0_1018.jpeg",
    "grid_scale_m_per_px": 0.002,
    "urgency_multiplier": 1.0,
    "detections": [
      {
        "class_id": 1,
        "class_name": "weed",
        "confidence": 0.731,
        "bbox": {
          "x1": 24.7, "y1": 117.74, "x2": 437.65, "y2": 420.1,
          "width": 412.95, "height": 302.36, "center_x": 231.17, "center_y": 268.92
        }
      }
    ]
  }'
```

### 13.5 Automated Test Suite Execution

All 63 unit and integration tests (including 17 API tests with mocked models) can be run in ~1.5 seconds:
```bash
# Run API endpoint tests only
./.venv/bin/pytest ml/tests/test_api.py -v

# Run the complete project test suite (63 passed)
./.venv/bin/pytest ml/tests/ -v
```


