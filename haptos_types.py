"""Shared result types used across the Haptos CV modules."""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

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
class FrameResult:
    """Structured summary for one processed frame."""

    frame_index: int
    command: str
    detections: List[Detection]
    fps: float

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["detections"] = [d.to_dict() for d in self.detections]
        return data
