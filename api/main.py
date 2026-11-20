"""FastAPI AI inference and swarm task allocation service for SwarmAgri AI."""

import logging
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI, File, HTTPException, Query, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.config import (
    API_DESCRIPTION,
    API_TITLE,
    API_VERSION,
    DEFAULT_CONF_THRESHOLD,
    MAX_CONF_THRESHOLD,
    MIN_CONF_THRESHOLD,
)
from api.detector import run_yolo_inference, validate_and_decode_image
from api.model_loader import model_manager
from api.schemas import (
    DetectionResponse,
    HealthResponse,
    ServiceStatus,
    SwarmTaskRequest,
    SwarmTaskResponse,
)
from api.swarm_tasks import convert_detections_to_swarm_tasks

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("SwarmAgriAPI")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager to load YOLO model once during startup."""
    logger.info("Initializing SwarmAgri AI inference service...")
    # Load model once on startup if not already loaded (e.g. by test mocks)
    if not model_manager.is_loaded:
        success = model_manager.load_model()
        if success:
            logger.info("Model loaded and verified successfully.")
        else:
            logger.warning(
                f"Model initialization encountered an issue: {model_manager.load_error}. "
                "Service will start in degraded mode. See /health."
            )
    yield
    # Cleanup on shutdown
    logger.info("Shutting down SwarmAgri AI inference service...")
    model_manager.unload()


app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS for local simulation dashboards
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["General"])
async def root():
    """Service overview and quick-start links."""
    return {
        "service": API_TITLE,
        "version": API_VERSION,
        "status": "online",
        "documentation": "/docs",
        "health": "/health",
        "endpoints": {
            "health": "GET /health",
            "detect": "POST /api/detect",
            "swarm_tasks": "POST /api/swarm/tasks",
        },
    }


@app.get("/health", response_model=HealthResponse, tags=["System Health"])
async def health_check():
    """Report service health, model status, device, and verified classes."""
    svc_status = ServiceStatus.HEALTHY if model_manager.is_loaded else ServiceStatus.DEGRADED
    return HealthResponse(
        status=svc_status,
        model_loaded=model_manager.is_loaded,
        model_path=model_manager.model_path or "unloaded",
        classes=model_manager.classes,
        device=model_manager.device,
        version=API_VERSION,
    )


@app.post(
    "/api/detect",
    response_model=DetectionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Perception & Vision"],
)
async def detect_objects(
    file: UploadFile = File(..., description="Uploaded agricultural image (JPEG, PNG, or WebP)"),
    conf: float = Query(
        default=DEFAULT_CONF_THRESHOLD,
        ge=MIN_CONF_THRESHOLD,
        le=MAX_CONF_THRESHOLD,
        description=(
            "Configurable confidence threshold defaulting to 0.25 to prioritize weed recall; "
            "this threshold is not claimed to be universally optimal."
        ),
    ),
):
    """Detect crop and weed objects from an uploaded image raster.
    
    Accepts JPEG, PNG, or WebP files, decodes them safely with OpenCV, runs
    fine-tuned YOLO11n inference, and returns exact pixel bounding boxes,
    confidence scores, and verified class mappings.
    """
    if not model_manager.is_loaded or model_manager.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Inference model is not currently loaded: {model_manager.load_error or 'Unknown error'}",
        )

    # Read binary content
    try:
        file_bytes = await file.read()
    except Exception as e:
        logger.error(f"Failed reading uploaded file: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read uploaded file payload: {str(e)}",
        )

    # Validate image payload and decode with OpenCV
    img_bgr, dimensions = validate_and_decode_image(
        file_bytes=file_bytes,
        filename=file.filename,
        content_type=file.content_type,
    )

    # Execute YOLO inference using the pre-loaded model
    detections, latency_ms = run_yolo_inference(
        model=model_manager.model,
        img_bgr=img_bgr,
        conf_threshold=conf,
        class_mapping=model_manager.classes,
    )

    crop_count = sum(1 for d in detections if d.class_id == 0 or d.class_name.lower() == "crop")
    weed_count = sum(1 for d in detections if d.class_id == 1 or d.class_name.lower() == "weed")

    return DetectionResponse(
        status="success",
        image_dimensions=dimensions,
        confidence_threshold=conf,
        inference_time_ms=latency_ms,
        total_detections=len(detections),
        crop_count=crop_count,
        weed_count=weed_count,
        detections=detections,
    )


@app.post(
    "/api/swarm/tasks",
    response_model=SwarmTaskResponse,
    status_code=status.HTTP_200_OK,
    tags=["Swarm Coordination & Simulation"],
)
async def generate_swarm_tasks(request: SwarmTaskRequest):
    """Convert weed detections into proposed simulation tasks with decoupled bidding metadata.
    
    Accepts vision detection outputs, extracts weed targets while preserving
    crops, computes stable deterministic task IDs, calculates urgency/priority
    scores, and generates simulation task payloads ready for decentralized
    robot auction and bidding.
    
    NOTE: Physical hardware dispatch is strictly disabled. This endpoint operates
    in simulation mode only.
    """
    try:
        response = convert_detections_to_swarm_tasks(request)
        return response
    except Exception as e:
        logger.error(f"Task generation failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed generating swarm simulation tasks: {str(e)}",
        )
