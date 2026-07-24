"""Shared result types used across the Haptos CV modules."""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

BBox = Tuple[int, int, int, int]


@dataclass(frozen=True)
class Detection:
    """One object detected in a frame.

    bbox uses OpenCV pixel coordinates: (x1, y1, x2, y2).
    region is filled during post-processing once image width is known.
    """

    class_name: str
    confidence: float
    bbox: BBox
    region: Optional[str] = None
    is_obstacle: bool = False
    median_depth_m: Optional[float] = None
    depth_pixel_count: int = 0
    depth_uncertainty_m: Optional[float] = None
    depth_valid_fraction: float = 0.0
    depth_fault_state: str = "not_available"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RawLidarScan:
    """Raw 2D LiDAR scan before filtering."""

    timestamp_ms: int
    angles_rad: np.ndarray
    distances_m: np.ndarray
    qualities: Optional[np.ndarray] = None


@dataclass(frozen=True)
class LidarFrameSummary:
    """Small JSON-safe summary of a filtered LiDAR frame."""

    timestamp_ms: int
    fault_state: str
    point_count: int
    nearest_distance_m: Optional[float] = None
    median_distance_m: Optional[float] = None
    farthest_distance_m: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StereoDepthSummary:
    """Small JSON-safe summary of one stereo disparity/depth estimate."""

    fault_state: str
    valid_pixel_count: int
    median_disparity_px: Optional[float] = None
    nearest_depth_m: Optional[float] = None
    median_depth_m: Optional[float] = None
    farthest_depth_m: Optional[float] = None
    valid_fraction: float = 0.0
    frame_skew_ms: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FilteredLidarFrame:
    """Cleaned LiDAR frame ready for projection or sensor fusion."""

    timestamp_ms: int
    points_xyz: np.ndarray
    distances_m: np.ndarray
    angles_rad: np.ndarray
    fault_state: str = "none"

    def point_count(self) -> int:
        return int(self.points_xyz.shape[0])

    def to_summary(self) -> LidarFrameSummary:
        if self.distances_m.size == 0:
            return LidarFrameSummary(
                timestamp_ms=self.timestamp_ms,
                fault_state=self.fault_state,
                point_count=0,
            )

        return LidarFrameSummary(
            timestamp_ms=self.timestamp_ms,
            fault_state=self.fault_state,
            point_count=self.point_count(),
            nearest_distance_m=float(np.min(self.distances_m)),
            median_distance_m=float(np.median(self.distances_m)),
            farthest_distance_m=float(np.max(self.distances_m)),
        )


@dataclass(frozen=True)
class FrameResult:
    """Structured summary for one processed frame."""

    frame_index: int
    command: str
    detections: List[Detection]
    fps: float
    cv_latency_ms: Optional[float] = None
    depth_latency_ms: Optional[float] = None
    lidar_summary: Optional[LidarFrameSummary] = None
    stereo_depth_summary: Optional[StereoDepthSummary] = None

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "frame_index": self.frame_index,
            "command": self.command,
            "fps": self.fps,
        }
        if self.cv_latency_ms is not None:
            data["cv_latency_ms"] = self.cv_latency_ms
        if self.depth_latency_ms is not None:
            data["depth_latency_ms"] = self.depth_latency_ms
        data["detections"] = [d.to_dict() for d in self.detections]
        if self.lidar_summary is not None:
            data["lidar"] = self.lidar_summary.to_dict()
        if self.stereo_depth_summary is not None:
            data["stereo_depth"] = self.stereo_depth_summary.to_dict()
        return data
