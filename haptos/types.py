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
    lidar_summary: Optional[LidarFrameSummary] = None

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "frame_index": self.frame_index,
            "command": self.command,
            "fps": self.fps,
        }
        data["detections"] = [d.to_dict() for d in self.detections]
        if self.lidar_summary is not None:
            data["lidar"] = self.lidar_summary.to_dict()
        return data
