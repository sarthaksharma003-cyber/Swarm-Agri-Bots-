"""Swarm simulation task generator decoupled from perception for decentralized robot bidding."""

import hashlib
import logging
import math
from typing import List, Tuple

from api.config import (
    HARDWARE_DISPATCH_ENABLED,
    SIMULATION_MODE,
)
from api.schemas import (
    BiddingContext,
    DetectionItem,
    SwarmCoordinates,
    SwarmTaskRequest,
    SwarmTaskResponse,
    SwarmWeedTask,
    TaskPriority,
    TaskStatus,
)

logger = logging.getLogger("SwarmAgriAPI.SwarmTasks")


def calculate_crop_proximity(weed_center: Tuple[float, float], crop_centers: List[Tuple[float, float]], threshold_px: float = 80.0) -> int:
    """Count number of detected crops within proximity threshold (in pixels) of the weed."""
    wx, wy = weed_center
    count = 0
    for cx, cy in crop_centers:
        dist = math.hypot(wx - cx, wy - cy)
        if dist <= threshold_px:
            count += 1
    return count


def generate_stable_task_id(
    field_id: str,
    image_id: str,
    index: int,
    center_x: float,
    center_y: float,
    width: float,
    height: float,
    confidence: float,
) -> str:
    """Generate a stable, deterministic 12-character hex task ID.
    
    Guarantees that identical detection inputs produce identical task IDs,
    enabling idempotency across distributed swarm nodes.
    """
    key_str = (
        f"{field_id}:{image_id}:{index}:"
        f"{center_x:.1f}:{center_y:.1f}:{width:.1f}:{height:.1f}:{confidence:.3f}"
    )
    digest = hashlib.sha256(key_str.encode("utf-8")).hexdigest()[:12]
    return f"task_weed_{digest}"


def determine_priority_and_score(
    confidence: float,
    biomass_score: float,
    proximity_crops: int,
    urgency_multiplier: float,
) -> Tuple[TaskPriority, float]:
    """Calculate numerical priority score and categorical priority tier.
    
    Larger weeds, weeds with higher detection certainty, and weeds in close
    proximity to crops receive higher urgency.
    """
    # 40% confidence certainty, 40% biomass size index, 20% crop proximity risk
    base_score = (confidence * 40.0) + (min(10.0, biomass_score) * 4.0) + (min(3, proximity_crops) * 6.67)
    final_score = round(min(100.0, base_score * urgency_multiplier), 2)

    if final_score >= 80.0:
        priority = TaskPriority.CRITICAL
    elif final_score >= 60.0:
        priority = TaskPriority.HIGH
    elif final_score >= 35.0:
        priority = TaskPriority.MEDIUM
    else:
        priority = TaskPriority.LOW

    return priority, final_score


def select_candidate_tools(biomass_score: float, proximity_crops: int) -> List[str]:
    """Select compatible effector tools for robotic weed termination."""
    if proximity_crops > 0 and biomass_score < 4.0:
        # Near sensitive crop roots: prioritize high-precision micro-intervention
        return ["precision_micro_sprayer", "laser_weeder"]
    elif biomass_score >= 6.0:
        # Large weed biomass: requires mechanical cutting or high-power thermal ablation
        return ["mechanical_cutter", "laser_weeder"]
    else:
        return ["mechanical_cutter", "precision_micro_sprayer", "laser_weeder"]


def convert_detections_to_swarm_tasks(request: SwarmTaskRequest) -> SwarmTaskResponse:
    """Convert weed detections into proposed simulation tasks with decoupled bidding metadata.
    
    Does NOT transmit physical commands to robotic hardware. All generated
    tasks are flagged with simulation_mode=True and hardware_dispatch=False.
    """
    image_id_label = request.image_id or "unnamed_frame"
    
    # 1. Partition detections into weeds and crops
    weeds: List[DetectionItem] = []
    crops: List[DetectionItem] = []
    for item in request.detections:
        # Use verified class mapping (ID 1 = weed, ID 0 = crop)
        if item.class_id == 1 or item.class_name.lower() == "weed":
            weeds.append(item)
        elif item.class_id == 0 or item.class_name.lower() == "crop":
            crops.append(item)

    crop_centers = [(c.bbox.center_x, c.bbox.center_y) for c in crops]
    tasks: List[SwarmWeedTask] = []

    # 2. Generate simulation task for each weed
    for idx, weed in enumerate(weeds, start=1):
        # Calculate spatial area and simulated field positions
        area_px = weed.bbox.width * weed.bbox.height
        # Normalize biomass to 0-10 scale based on standard 512x512 raster
        biomass_score = round(min(10.0, (area_px / (512.0 * 512.0)) * 50.0), 2)
        
        # Check nearby crops within 80px radius (exclusion/caution zone)
        proximity_crops = calculate_crop_proximity(
            (weed.bbox.center_x, weed.bbox.center_y),
            crop_centers,
            threshold_px=80.0
        )

        priority, priority_score = determine_priority_and_score(
            confidence=weed.confidence,
            biomass_score=biomass_score,
            proximity_crops=proximity_crops,
            urgency_multiplier=request.urgency_multiplier,
        )

        # Stable, reproducible task ID
        task_id = generate_stable_task_id(
            field_id=request.field_id,
            image_id=image_id_label,
            index=idx,
            center_x=weed.bbox.center_x,
            center_y=weed.bbox.center_y,
            width=weed.bbox.width,
            height=weed.bbox.height,
            confidence=weed.confidence,
        )

        # Coordinate transformation (pixels to simulated field meters)
        sim_x = round(weed.bbox.center_x * request.grid_scale_m_per_px, 4)
        sim_y = round(weed.bbox.center_y * request.grid_scale_m_per_px, 4)

        candidate_tools = select_candidate_tools(biomass_score, proximity_crops)
        estimated_energy_wh = round(2.0 + (biomass_score * 1.5), 2)

        # Context for future decentralized robot bidding (distance, battery, tool, urgency)
        bidding_context = BiddingContext(
            candidate_tools=candidate_tools,
            target_biomass_score=biomass_score,
            estimated_energy_cost_wh=estimated_energy_wh,
            urgency_score=round(priority_score / 10.0, 2),
            proximity_crop_exclusion_zones=proximity_crops,
            auction_status="open_for_bidding",
            assigned_robot_id=None,
        )

        task = SwarmWeedTask(
            task_id=task_id,
            task_type="WEED_REMOVAL",
            status=TaskStatus.PROPOSED,
            priority=priority,
            priority_score=priority_score,
            weed_confidence=weed.confidence,
            coordinates=SwarmCoordinates(
                pixel_center_x=weed.bbox.center_x,
                pixel_center_y=weed.bbox.center_y,
                sim_x_m=sim_x,
                sim_y_m=sim_y,
            ),
            bbox=weed.bbox,
            bidding_context=bidding_context,
            simulation_mode=SIMULATION_MODE,
            hardware_dispatch=HARDWARE_DISPATCH_ENABLED,
        )
        tasks.append(task)

    # Sort tasks by descending priority score so highest urgency tasks are presented first
    tasks.sort(key=lambda t: t.priority_score, reverse=True)

    msg = (
        f"Generated {len(tasks)} proposed simulation tasks from {len(weeds)} weed detections. "
        f"{len(crops)} crops identified and protected in exclusion zones."
    )
    logger.info(msg)

    return SwarmTaskResponse(
        status="success",
        field_id=request.field_id,
        tasks_generated=len(tasks),
        tasks=tasks,
        crops_protected_count=len(crops),
        simulation_mode=SIMULATION_MODE,
        hardware_dispatch=HARDWARE_DISPATCH_ENABLED,
        message=msg,
    )
