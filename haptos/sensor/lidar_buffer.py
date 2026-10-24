"""Small timestamp buffer for processed LiDAR frames."""

from collections import deque
from typing import Optional

from haptos.types import ProcessedLidarFrame


class LidarFrameBuffer:
    """Keep recent LiDAR frames and retrieve the closest one by timestamp."""

    # Set up a bounded deque to hold the most recent frames.
    def __init__(self, max_frames: int = 10):
        if max_frames <= 0:
            raise ValueError("max_frames must be positive.")
        self.frames = deque(maxlen=max_frames)

    # Append a new processed frame, evicting the oldest if full.
    def add(self, frame: ProcessedLidarFrame) -> None:
        self.frames.append(frame)

    # Find the buffered frame with the timestamp nearest to the given one.
    def closest_to(self, timestamp_ms: int) -> Optional[ProcessedLidarFrame]:
        if not self.frames:
            return None

        return min(
            self.frames,
            key=lambda frame: abs(frame.timestamp_ms - timestamp_ms),
        )

    # Report how many frames are currently buffered.
    def __len__(self) -> int:
        return len(self.frames)
