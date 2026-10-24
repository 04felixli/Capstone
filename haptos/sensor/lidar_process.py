"""Stage-two LiDAR filtering and point cloud formation."""

import numpy as np

from haptos.config import (
    LIDAR_FAULT_NO_VALID_POINTS,
    LIDAR_FAULT_NONE,
    LIDAR_MAX_DISTANCE_M,
    LIDAR_MIN_DISTANCE_M,
    LIDAR_MIN_QUALITY,
)
from haptos.types import ProcessedLidarFrame, RawLidarScan


# Drop out-of-range/low-quality points, convert the survivors to XYZ, and flag a fault if none remain.
def process_lidar_scan(
    scan: RawLidarScan,
    min_distance_m: float = LIDAR_MIN_DISTANCE_M,
    max_distance_m: float = LIDAR_MAX_DISTANCE_M,
    min_quality: int = LIDAR_MIN_QUALITY,
) -> ProcessedLidarFrame:
    """Turn a raw 2D LiDAR scan into a cleaned, XYZ-projected frame.

    Filters out-of-range/low-quality points, converts the rest into XYZ points,
    and reports a fault state when no points survive filtering.

    The coordinate convention is:
    x = left/right, y = vertical, z = forward.
    """

    angles = np.asarray(scan.angles_rad, dtype=np.float32)
    distances = np.asarray(scan.distances_m, dtype=np.float32)
    _validate_scan_shapes(angles, distances, scan.qualities)

    valid = np.isfinite(angles) & np.isfinite(distances)
    valid &= distances >= min_distance_m
    valid &= distances <= max_distance_m

    if scan.qualities is not None:
        qualities = np.asarray(scan.qualities)
        valid &= qualities >= min_quality

    filtered_angles = angles[valid]
    filtered_distances = distances[valid]

    if filtered_distances.size == 0:
        return _empty_processed_frame(scan.timestamp_ms, LIDAR_FAULT_NO_VALID_POINTS)

    points_xyz = polar_scan_to_xyz(filtered_angles, filtered_distances)

    return ProcessedLidarFrame(
        timestamp_ms=scan.timestamp_ms,
        points_xyz=points_xyz,
        distances_m=filtered_distances.astype(np.float32),
        angles_rad=filtered_angles.astype(np.float32),
        fault_state=LIDAR_FAULT_NONE,
    )


# Project polar (angle, distance) pairs onto the flat y=0 plane as XYZ points.
def polar_scan_to_xyz(angles_rad: np.ndarray, distances_m: np.ndarray) -> np.ndarray:
    """Convert a 2D polar scan into simple 3D points."""

    x = distances_m * np.sin(angles_rad)
    y = np.zeros_like(distances_m)
    z = distances_m * np.cos(angles_rad)
    return np.column_stack((x, y, z)).astype(np.float32)


# Ensure angle/distance/quality arrays are 1D and shape-matched before filtering.
def _validate_scan_shapes(
    angles_rad: np.ndarray,
    distances_m: np.ndarray,
    qualities: np.ndarray | None,
) -> None:
    if angles_rad.ndim != 1 or distances_m.ndim != 1:
        raise ValueError("LiDAR angles and distances must be one-dimensional arrays.")
    if angles_rad.shape != distances_m.shape:
        raise ValueError("LiDAR angles and distances must have the same shape.")
    if qualities is not None and np.asarray(qualities).shape != distances_m.shape:
        raise ValueError("LiDAR qualities must match the distance array shape.")


# Build a zero-point ProcessedLidarFrame carrying the given fault state.
def _empty_processed_frame(timestamp_ms: int, fault_state: str) -> ProcessedLidarFrame:
    return ProcessedLidarFrame(
        timestamp_ms=timestamp_ms,
        points_xyz=np.empty((0, 3), dtype=np.float32),
        distances_m=np.empty((0,), dtype=np.float32),
        angles_rad=np.empty((0,), dtype=np.float32),
        fault_state=fault_state,
    )
