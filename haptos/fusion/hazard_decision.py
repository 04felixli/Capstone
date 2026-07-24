"""Fuse CV depth and LiDAR range into a conservative navigation command."""

from typing import Iterable, Optional

from haptos.config import COMMAND_STOP, LIDAR_FAULT_NONE
from haptos.cv.postprocess import generate_navigation_hint
from haptos.types import Detection, LidarFrameSummary


def generate_fused_navigation_hint(
    detections: Iterable[Detection],
    lidar_summary: Optional[LidarFrameSummary] = None,
    *,
    max_obstacle_distance_m: float = 2.5,
    emergency_stop_distance_m: float = 0.8,
    min_depth_valid_fraction: float = 0.15,
    max_relative_depth_uncertainty: float = 0.35,
) -> str:
    """Use trusted ranges to ignore distant objects and stop for near hazards."""

    if max_obstacle_distance_m <= 0:
        raise ValueError("max_obstacle_distance_m must be positive")
    if emergency_stop_distance_m <= 0:
        raise ValueError("emergency_stop_distance_m must be positive")
    if emergency_stop_distance_m > max_obstacle_distance_m:
        raise ValueError("emergency_stop_distance_m cannot exceed max_obstacle_distance_m")

    detections = list(detections)
    for detection in detections:
        if (
            detection.is_obstacle
            and _has_trusted_depth(
                detection,
                min_depth_valid_fraction=min_depth_valid_fraction,
                max_relative_depth_uncertainty=max_relative_depth_uncertainty,
            )
            and detection.median_depth_m <= emergency_stop_distance_m
        ):
            return COMMAND_STOP

    if (
        lidar_summary is not None
        and lidar_summary.fault_state == LIDAR_FAULT_NONE
        and lidar_summary.nearest_distance_m is not None
        and lidar_summary.nearest_distance_m <= emergency_stop_distance_m
    ):
        return COMMAND_STOP

    actionable = [
        detection
        for detection in detections
        if not _is_trusted_distant_detection(
            detection,
            max_obstacle_distance_m=max_obstacle_distance_m,
            min_depth_valid_fraction=min_depth_valid_fraction,
            max_relative_depth_uncertainty=max_relative_depth_uncertainty,
        )
    ]
    return generate_navigation_hint(actionable)


def _is_trusted_distant_detection(
    detection: Detection,
    *,
    max_obstacle_distance_m: float,
    min_depth_valid_fraction: float,
    max_relative_depth_uncertainty: float,
) -> bool:
    return (
        detection.is_obstacle
        and _has_trusted_depth(
            detection,
            min_depth_valid_fraction=min_depth_valid_fraction,
            max_relative_depth_uncertainty=max_relative_depth_uncertainty,
        )
        and detection.median_depth_m > max_obstacle_distance_m
    )


def _has_trusted_depth(
    detection: Detection,
    *,
    min_depth_valid_fraction: float,
    max_relative_depth_uncertainty: float,
) -> bool:
    if detection.median_depth_m is None or detection.median_depth_m <= 0:
        return False
    if detection.depth_fault_state != "none":
        return False
    if detection.depth_valid_fraction < min_depth_valid_fraction:
        return False
    if detection.depth_uncertainty_m is None:
        return False
    relative_uncertainty = detection.depth_uncertainty_m / detection.median_depth_m
    return relative_uncertainty <= max_relative_depth_uncertainty
