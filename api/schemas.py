"""Pydantic schemas for SwarmAgri AI Inference and Task Allocation API."""

from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class ServiceStatus(str, Enum):
    """Health check status states."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    INITIALIZING = "initializing"


class TaskPriority(str, Enum):
    """Priority levels for agricultural swarm actions."""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class TaskStatus(str, Enum):
    """Status states for swarm simulation tasks."""
    PROPOSED = "PROPOSED"
    PENDING_AUCTION = "PENDING_AUCTION"
    ASSIGNED = "ASSIGNED"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"


class HealthResponse(BaseModel):
    """Health check response schema."""
    status: ServiceStatus
    model_loaded: bool
    model_path: str
    classes: Dict[int, str]
    device: str
    version: str
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class BoundingBox(BaseModel):
    """Pixel coordinate bounding box with derived spatial metrics."""
    x1: float = Field(..., description="Top-left X in pixel coordinates")
    y1: float = Field(..., description="Top-left Y in pixel coordinates")
    x2: float = Field(..., description="Bottom-right X in pixel coordinates")
    y2: float = Field(..., description="Bottom-right Y in pixel coordinates")
    width: float = Field(..., description="Width in pixels")
    height: float = Field(..., description="Height in pixels")
    center_x: float = Field(..., description="Center X in pixel coordinates")
    center_y: float = Field(..., description="Center Y in pixel coordinates")


class ImageDimensions(BaseModel):
    """Raster image dimensions."""
    width: int = Field(..., description="Image width in pixels")
    height: int = Field(..., description="Image height in pixels")
    channels: int = Field(default=3, description="Color channels (e.g. 3 for RGB)")


class DetectionItem(BaseModel):
    """Individual object detection item."""
    class_id: int = Field(..., description="Verified Class ID (0: crop, 1: weed)")
    class_name: str = Field(..., description="Verified Class Name ('crop' or 'weed')")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence score")
    bbox: BoundingBox = Field(..., description="Pixel-space bounding box")


class DetectionResponse(BaseModel):
    """Structured detection response from YOLO inference."""
    status: str = "success"
    image_dimensions: ImageDimensions
    confidence_threshold: float
    inference_time_ms: float
    total_detections: int
    crop_count: int
    weed_count: int
    detections: List[DetectionItem]


class BiddingContext(BaseModel):
    """Decoupled bidding metadata for future decentralized robot allocation."""
    candidate_tools: List[str] = Field(
        default_factory=lambda: ["mechanical_cutter", "precision_micro_sprayer", "laser_weeder"],
        description="Compatible effector tools for this target"
    )
    target_biomass_score: float = Field(
        ...,
        description="Estimated weed biomass index based on bounding area and confidence"
    )
    estimated_energy_cost_wh: float = Field(
        ...,
        description="Simulated baseline energy cost for physical effector interaction in watt-hours"
    )
    urgency_score: float = Field(
        ...,
        ge=0.0,
        description="Composite urgency score (0.0 to 10.0 scale) based on crop proximity and size"
    )
    proximity_crop_exclusion_zones: int = Field(
        default=0,
        description="Number of neighboring crops within protective perimeter"
    )
    auction_status: str = Field(
        default="open_for_bidding",
        description="Current auction lifecycle state for decentralized consensus"
    )
    assigned_robot_id: Optional[str] = Field(
        default=None,
        description="ID of winning robot rover after auction resolution (null in proposed state)"
    )


class SwarmCoordinates(BaseModel):
    """Spatial coordinates in both image pixel space and simulated metric field space."""
    pixel_center_x: float
    pixel_center_y: float
    sim_x_m: float = Field(..., description="Simulated field X position in meters")
    sim_y_m: float = Field(..., description="Simulated field Y position in meters")


class SwarmWeedTask(BaseModel):
    """Proposed simulation task for weed removal by autonomous rovers."""
    task_id: str = Field(..., description="Stable deterministic task identifier")
    task_type: str = Field(default="WEED_REMOVAL", description="Simulation task category")
    status: TaskStatus = Field(default=TaskStatus.PROPOSED, description="Lifecycle status")
    priority: TaskPriority = Field(..., description="Categorical priority level")
    priority_score: float = Field(..., description="Continuous priority ranking score")
    weed_confidence: float = Field(..., description="Original vision detection confidence")
    coordinates: SwarmCoordinates = Field(..., description="Target spatial coordinates")
    bbox: BoundingBox = Field(..., description="Pixel bounding box of target")
    bidding_context: BiddingContext = Field(..., description="Context payload for robot bidding")
    simulation_mode: bool = Field(default=True, description="Strictly simulation flag")
    hardware_dispatch: bool = Field(default=False, description="Physical hardware dispatch disabled")
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class SwarmTaskRequest(BaseModel):
    """Request payload for converting weed detections into simulation tasks."""
    detections: List[DetectionItem] = Field(..., description="List of detected crops and weeds")
    image_id: Optional[str] = Field(default=None, description="Source frame or image identifier")
    field_id: str = Field(default="field_alpha", description="Target agricultural plot identifier")
    grid_scale_m_per_px: float = Field(
        default=0.002,
        ge=0.0001,
        description="Spatial resolution mapping pixel dimension to simulated meters (e.g. 512px = ~1.024m)"
    )
    urgency_multiplier: float = Field(
        default=1.0,
        ge=0.1,
        le=5.0,
        description="Field-level multiplier scaling task priority"
    )


class SwarmTaskResponse(BaseModel):
    """Response containing proposed simulation tasks."""
    status: str = "success"
    field_id: str
    tasks_generated: int
    tasks: List[SwarmWeedTask]
    crops_protected_count: int
    simulation_mode: bool = True
    hardware_dispatch: bool = False
    message: str
