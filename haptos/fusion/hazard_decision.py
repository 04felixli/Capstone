"""Fuse camera region detections with LiDAR range into a navigation command."""

from typing import Iterable, Optional

from haptos.config import COMMAND_STOP, LIDAR_FAULT_NONE
from haptos.cv.postprocess import generate_navigation_hint
from haptos.types import Detection, LidarFrameSummary


def generate_fused_navigation_hint(
    detections: Iterable[Detection],
    lidar_summary: Optional[LidarFrameSummary] = None,
    *,
    emergency_stop_distance_m: float = 0.8,
) -> str:
    """Force STOP for a trusted near LiDAR return, otherwise use camera region logic."""

    if emergency_stop_distance_m <= 0:
        raise ValueError("emergency_stop_distance_m must be positive")

    if (
        lidar_summary is not None
        and lidar_summary.fault_state == LIDAR_FAULT_NONE
        and lidar_summary.nearest_distance_m is not None
        and lidar_summary.nearest_distance_m <= emergency_stop_distance_m
    ):
        return COMMAND_STOP

    return generate_navigation_hint(detections)
