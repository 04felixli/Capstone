"""Convert raw detections into navigation-friendly frame results."""

from typing import Iterable, List, Set

from config import (
    CENTER_REGION,
    COMMAND_FORWARD,
    COMMAND_GO_LEFT,
    COMMAND_GO_RIGHT,
    COMMAND_STOP,
    LEFT_REGION,
    OBSTACLE_CLASSES,
    RIGHT_REGION,
)
from haptos_types import Detection


def map_bbox_to_region(bbox, frame_width: int) -> str:
    """Map a detection center point to LEFT, CENTER, or RIGHT image thirds."""

    x1, _, x2, _ = bbox
    center_x = (x1 + x2) / 2.0
    one_third = frame_width / 3.0
    two_thirds = 2.0 * frame_width / 3.0

    if center_x < one_third:
        return LEFT_REGION
    if center_x < two_thirds:
        return CENTER_REGION
    return RIGHT_REGION


def filter_and_enrich_detections(
    raw_detections: Iterable[Detection],
    frame_width: int,
    confidence_threshold: float,
) -> List[Detection]:
    """Apply confidence filtering and add region/obstacle metadata."""

    enriched: List[Detection] = []
    for detection in raw_detections:
        if detection.confidence < confidence_threshold:
            continue

        region = map_bbox_to_region(detection.bbox, frame_width)
        is_obstacle = detection.class_name.lower() in OBSTACLE_CLASSES
        enriched.append(
            Detection(
                class_name=detection.class_name,
                confidence=detection.confidence,
                bbox=detection.bbox,
                region=region,
                is_obstacle=is_obstacle,
            )
        )

    return enriched


def generate_navigation_hint(detections: Iterable[Detection]) -> str:
    """Produce a simple command for later sensor fusion/haptic logic."""

    obstacle_regions: Set[str] = {
        detection.region
        for detection in detections
        if detection.is_obstacle and detection.region is not None
    }

    if not obstacle_regions:
        return COMMAND_FORWARD
    if CENTER_REGION in obstacle_regions:
        return COMMAND_STOP
    if len(obstacle_regions) > 1:
        return COMMAND_STOP
    if LEFT_REGION in obstacle_regions:
        return COMMAND_GO_RIGHT
    if RIGHT_REGION in obstacle_regions:
        return COMMAND_GO_LEFT
    return COMMAND_FORWARD
