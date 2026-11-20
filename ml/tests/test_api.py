"""Unit and integration tests for FastAPI inference and swarm task service.

All tests utilize mocked models to ensure zero network/weight download overhead
and sub-second test execution.
"""

import io
import sys
from pathlib import Path
from typing import List, Optional

# Ensure workspace root is in sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

import cv2
import numpy as np
import pytest
from starlette.testclient import TestClient

from api.main import app
from api.model_loader import model_manager
from api.schemas import (
    BoundingBox,
    DetectionItem,
    SwarmTaskRequest,
)


class MockBox:
    """Mock bounding box mimicking Ultralytics box representation."""
    def __init__(self, cls_id: int, conf: float, xyxy: List[float]):
        self.cls = [cls_id]
        self.conf = [conf]
        self.xyxy = [xyxy]


class MockResult:
    """Mock YOLO Result object."""
    def __init__(self, boxes: Optional[List[MockBox]] = None):
        self.boxes = boxes or []


class MockYOLOModel:
    """Mock YOLO model for testing without weight files."""
    def __init__(self, boxes: Optional[List[MockBox]] = None):
        self.boxes = boxes or []
        self.names = {0: "crop", 1: "weed"}
        self.last_conf = None
        self.last_imgsz = None

    def predict(self, source, conf=0.25, imgsz=512, verbose=False):
        self.last_conf = conf
        self.last_imgsz = imgsz
        return [MockResult(boxes=self.boxes)]


def create_test_image_bytes(width: int = 512, height: int = 512, format: str = ".jpg") -> bytes:
    """Generate an in-memory encoded image raster for upload tests."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    # Draw simple gradient/shape so it's a valid non-empty JPEG
    cv2.circle(img, (width // 2, height // 2), 50, (0, 255, 0), -1)
    success, buffer = cv2.imencode(format, img)
    assert success, "Failed to encode test image"
    return buffer.tobytes()


@pytest.fixture
def client():
    """TestClient fixture with mocked model pre-loaded."""
    mock_boxes = [
        MockBox(0, 0.88, [40.0, 50.0, 180.0, 220.0]),
        MockBox(1, 0.76, [260.0, 280.0, 440.0, 460.0]),
    ]
    mock_model = MockYOLOModel(boxes=mock_boxes)
    model_manager.set_mock_model(mock_model, model_path="mock_yolo11n.pt")
    
    with TestClient(app) as test_client:
        yield test_client
        
    model_manager.unload()


# ---------------------------------------------------------------------------
# 1. Health Endpoint Tests
# ---------------------------------------------------------------------------

def test_health_endpoint_healthy(client):
    """GET /health reports healthy status when model is loaded."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["classes"] == {"0": "crop", "1": "weed"}
    assert data["model_path"] == "mock_yolo11n.pt"
    assert "version" in data
    assert "timestamp" in data


def test_health_endpoint_degraded(monkeypatch):
    """GET /health reports degraded status when model is not loaded."""
    monkeypatch.setattr(model_manager, "load_model", lambda target_path=None: False)
    model_manager.unload()
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "degraded"
        assert data["model_loaded"] is False


def test_root_endpoint(client):
    """GET / returns service overview and route links."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "endpoints" in data
    assert data["endpoints"]["detect"] == "POST /api/detect"
    assert data["endpoints"]["swarm_tasks"] == "POST /api/swarm/tasks"


# ---------------------------------------------------------------------------
# 2. Upload Validation & Error Handling Tests
# ---------------------------------------------------------------------------

def test_detect_model_not_loaded_returns_503(monkeypatch):
    """POST /api/detect returns 503 when model is not loaded."""
    monkeypatch.setattr(model_manager, "load_model", lambda target_path=None: False)
    model_manager.unload()
    with TestClient(app) as client:
        img_bytes = create_test_image_bytes()
        files = {"file": ("test.jpg", img_bytes, "image/jpeg")}
        response = client.post("/api/detect", files=files)
        assert response.status_code == 503
        assert "not currently loaded" in response.json()["detail"]


def test_detect_empty_file_upload(client):
    """POST /api/detect rejects 0-byte uploaded file with 400 Bad Request."""
    files = {"file": ("empty.jpg", b"", "image/jpeg")}
    response = client.post("/api/detect", files=files)
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_detect_unsupported_file_extension(client):
    """POST /api/detect rejects unsupported file extensions with 415."""
    files = {"file": ("unsupported_raster.bmp", b"BM dummy bmp data", "image/bmp")}
    response = client.post("/api/detect", files=files)
    assert response.status_code == 415
    assert "unsupported file extension" in response.json()["detail"].lower()


def test_detect_invalid_media_type(client):
    """POST /api/detect rejects non-image MIME types with 415."""
    files = {"file": ("document.pdf", b"%PDF-1.4", "application/pdf")}
    response = client.post("/api/detect", files=files)
    assert response.status_code == 415
    assert "invalid media type" in response.json()["detail"].lower()


def test_detect_corrupted_image_data(client):
    """POST /api/detect rejects random byte payload disguised as JPEG with 400."""
    fake_bytes = b"NOT_A_VALID_JPEG_HEADER_1234567890"
    files = {"file": ("corrupt.jpg", fake_bytes, "image/jpeg")}
    response = client.post("/api/detect", files=files)
    assert response.status_code == 400
    assert "decode" in response.json()["detail"].lower()


def test_detect_payload_too_large(client, monkeypatch):
    """POST /api/detect enforces maximum upload size limits (413)."""
    # Temporarily set upload limit to 1 KB to test 413 without allocating 10MB
    import api.detector as detector_module
    monkeypatch.setattr(detector_module, "MAX_UPLOAD_SIZE_BYTES", 1024)

    large_payload = b"X" * 2048
    files = {"file": ("oversized.jpg", large_payload, "image/jpeg")}
    response = client.post("/api/detect", files=files)
    assert response.status_code == 413
    assert "exceeds the maximum allowed limit" in response.json()["detail"].lower()


# ---------------------------------------------------------------------------
# 3. Detection Endpoint Logic & Class Mapping Tests
# ---------------------------------------------------------------------------

def test_detect_success(client):
    """POST /api/detect returns 200 with verified class mappings and pixel boxes."""
    img_bytes = create_test_image_bytes(512, 512, ".jpg")
    files = {"file": ("field_sample.jpg", img_bytes, "image/jpeg")}
    response = client.post("/api/detect", files=files)
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "success"
    assert data["image_dimensions"] == {"width": 512, "height": 512, "channels": 3}
    assert data["total_detections"] == 2
    assert data["crop_count"] == 1
    assert data["weed_count"] == 1
    assert data["confidence_threshold"] == 0.25

    # Check verified class 0: crop
    crop_item = next(d for d in data["detections"] if d["class_id"] == 0)
    assert crop_item["class_name"] == "crop"
    assert crop_item["confidence"] == 0.88
    assert crop_item["bbox"]["x1"] == 40.0
    assert crop_item["bbox"]["y1"] == 50.0
    assert crop_item["bbox"]["width"] == 140.0
    assert crop_item["bbox"]["height"] == 170.0
    assert crop_item["bbox"]["center_x"] == 110.0
    assert crop_item["bbox"]["center_y"] == 135.0

    # Check verified class 1: weed
    weed_item = next(d for d in data["detections"] if d["class_id"] == 1)
    assert weed_item["class_name"] == "weed"
    assert weed_item["confidence"] == 0.76
    assert weed_item["bbox"]["x1"] == 260.0
    assert weed_item["bbox"]["y1"] == 280.0
    assert weed_item["bbox"]["width"] == 180.0
    assert weed_item["bbox"]["height"] == 180.0
    assert weed_item["bbox"]["center_x"] == 350.0
    assert weed_item["bbox"]["center_y"] == 370.0


def test_detect_empty_detections(client):
    """POST /api/detect handles images with zero detected objects cleanly."""
    mock_model = MockYOLOModel(boxes=[])
    model_manager.set_mock_model(mock_model)

    img_bytes = create_test_image_bytes(512, 512, ".jpg")
    files = {"file": ("soil_only.jpg", img_bytes, "image/jpeg")}
    response = client.post("/api/detect", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["total_detections"] == 0
    assert data["crop_count"] == 0
    assert data["weed_count"] == 0
    assert data["detections"] == []


def test_detect_configurable_confidence(client):
    """POST /api/detect respects custom confidence threshold query parameter."""
    img_bytes = create_test_image_bytes(512, 512, ".jpg")
    files = {"file": ("field.jpg", img_bytes, "image/jpeg")}
    response = client.post("/api/detect?conf=0.45", files=files)
    assert response.status_code == 200
    assert response.json()["confidence_threshold"] == 0.45
    assert model_manager.model.last_conf == 0.45


def test_detect_invalid_confidence_ranges(client):
    """POST /api/detect validates confidence threshold boundaries (422)."""
    img_bytes = create_test_image_bytes(512, 512, ".jpg")
    files = {"file": ("field.jpg", img_bytes, "image/jpeg")}
    
    # Test negative
    res_neg = client.post("/api/detect?conf=-0.1", files=files)
    assert res_neg.status_code == 422

    # Test > 1.0
    res_high = client.post("/api/detect?conf=1.5", files=files)
    assert res_high.status_code == 422


# ---------------------------------------------------------------------------
# 4. Swarm Task Allocation & Bidding Hook Tests
# ---------------------------------------------------------------------------

def test_swarm_tasks_weed_conversion_and_crop_protection(client):
    """POST /api/swarm/tasks converts weeds into tasks while protecting crops."""
    payload = {
        "field_id": "field_section_04",
        "image_id": "frame_0012.jpg",
        "grid_scale_m_per_px": 0.002,
        "urgency_multiplier": 1.2,
        "detections": [
            {
                "class_id": 0,
                "class_name": "crop",
                "confidence": 0.92,
                "bbox": {
                    "x1": 50.0, "y1": 60.0, "x2": 150.0, "y2": 160.0,
                    "width": 100.0, "height": 100.0, "center_x": 100.0, "center_y": 110.0
                }
            },
            {
                "class_id": 1,
                "class_name": "weed",
                "confidence": 0.84,
                "bbox": {
                    "x1": 120.0, "y1": 130.0, "x2": 220.0, "y2": 230.0,
                    "width": 100.0, "height": 100.0, "center_x": 170.0, "center_y": 180.0
                }
            }
        ]
    }
    response = client.post("/api/swarm/tasks", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "success"
    assert data["field_id"] == "field_section_04"
    assert data["tasks_generated"] == 1
    assert data["crops_protected_count"] == 1
    assert data["simulation_mode"] is True
    assert data["hardware_dispatch"] is False

    task = data["tasks"][0]
    assert task["task_type"] == "WEED_REMOVAL"
    assert task["status"] == "PROPOSED"
    assert task["task_id"].startswith("task_weed_")
    assert task["weed_confidence"] == 0.84
    assert task["simulation_mode"] is True
    assert task["hardware_dispatch"] is False

    # Verify spatial coordinates
    assert task["coordinates"]["pixel_center_x"] == 170.0
    assert task["coordinates"]["pixel_center_y"] == 180.0
    # 170.0 * 0.002 = 0.34m, 180.0 * 0.002 = 0.36m
    assert task["coordinates"]["sim_x_m"] == 0.34
    assert task["coordinates"]["sim_y_m"] == 0.36

    # Verify decoupled bidding context
    bidding = task["bidding_context"]
    assert bidding["auction_status"] == "open_for_bidding"
    assert bidding["assigned_robot_id"] is None
    assert bidding["target_biomass_score"] > 0
    assert bidding["estimated_energy_cost_wh"] > 0
    assert len(bidding["candidate_tools"]) > 0


def test_swarm_tasks_stable_deterministic_task_id(client):
    """POST /api/swarm/tasks produces stable, identical task IDs across multiple calls."""
    payload = {
        "field_id": "field_beta",
        "image_id": "camera_01.jpg",
        "detections": [
            {
                "class_id": 1,
                "class_name": "weed",
                "confidence": 0.775,
                "bbox": {
                    "x1": 100.0, "y1": 100.0, "x2": 250.0, "y2": 250.0,
                    "width": 150.0, "height": 150.0, "center_x": 175.0, "center_y": 175.0
                }
            }
        ]
    }

    # Call 1
    resp1 = client.post("/api/swarm/tasks", json=payload)
    task_id_1 = resp1.json()["tasks"][0]["task_id"]

    # Call 2 (identical payload)
    resp2 = client.post("/api/swarm/tasks", json=payload)
    task_id_2 = resp2.json()["tasks"][0]["task_id"]

    assert task_id_1 == task_id_2
    assert task_id_1.startswith("task_weed_")


def test_swarm_tasks_only_crops_generates_zero_tasks(client):
    """POST /api/swarm/tasks ignores crop-only detections for weed removal."""
    payload = {
        "field_id": "field_gamma",
        "detections": [
            {
                "class_id": 0,
                "class_name": "crop",
                "confidence": 0.95,
                "bbox": {
                    "x1": 50.0, "y1": 50.0, "x2": 150.0, "y2": 150.0,
                    "width": 100.0, "height": 100.0, "center_x": 100.0, "center_y": 100.0
                }
            }
        ]
    }
    response = client.post("/api/swarm/tasks", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["tasks_generated"] == 0
    assert data["tasks"] == []
    assert data["crops_protected_count"] == 1


def test_swarm_tasks_empty_detections_payload(client):
    """POST /api/swarm/tasks handles empty detection list gracefully."""
    payload = {
        "field_id": "field_empty",
        "detections": []
    }
    response = client.post("/api/swarm/tasks", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["tasks_generated"] == 0
    assert data["tasks"] == []
    assert data["crops_protected_count"] == 0
