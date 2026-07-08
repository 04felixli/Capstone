"""Small timestamp buffer for filtered LiDAR frames."""

from collections import deque
from typing import Optional

from haptos.types import FilteredLidarFrame


class LidarFrameBuffer:
    """Keep recent LiDAR frames and retrieve the closest one by timestamp."""

    def __init__(self, max_frames: int = 10):
        if max_frames <= 0:
            raise ValueError("max_frames must be positive.")
        self.frames = deque(maxlen=max_frames)

    def add(self, frame: FilteredLidarFrame) -> None:
        self.frames.append(frame)

    def closest_to(self, timestamp_ms: int) -> Optional[FilteredLidarFrame]:
        if not self.frames:
            return None

        return min(
            self.frames,
            key=lambda frame: abs(frame.timestamp_ms - timestamp_ms),
        )

    def __len__(self) -> int:
        return len(self.frames)
