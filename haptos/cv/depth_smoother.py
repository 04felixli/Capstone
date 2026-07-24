"""Short-lived temporal smoothing for object depth measurements."""

from collections import deque
from dataclasses import dataclass, replace
from typing import Deque, Dict, Optional

import numpy as np

from haptos.types import BBox, Detection


@dataclass
class _DepthTrack:
    class_name: str
    bbox: BBox
    depths_m: Deque[float]
    last_frame_index: int


class DetectionDepthSmoother:
    """Associate nearby detections and median-filter their recent depths."""

    def __init__(
        self,
        window_size: int = 5,
        min_iou: float = 0.3,
        max_track_age: int = 5,
    ):
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if min_iou < 0.0 or min_iou > 1.0:
            raise ValueError("min_iou must be between 0 and 1")
        if max_track_age < 0:
            raise ValueError("max_track_age must be 0 or greater")

        self.window_size = window_size
        self.min_iou = min_iou
        self.max_track_age = max_track_age
        self._frame_index = 0
        self._next_track_id = 1
        self._tracks: Dict[int, _DepthTrack] = {}

    def update(self, detections: list[Detection]) -> list[Detection]:
        self._frame_index += 1
        matched_track_ids: set[int] = set()
        smoothed: list[Detection] = []

        for detection in detections:
            track_id = self._find_match(detection, matched_track_ids)
            if track_id is None:
                track_id = self._create_track(detection)

            track = self._tracks[track_id]
            track.bbox = detection.bbox
            track.last_frame_index = self._frame_index
            matched_track_ids.add(track_id)

            if detection.median_depth_m is not None:
                track.depths_m.append(detection.median_depth_m)

            if not track.depths_m:
                smoothed.append(detection)
                continue

            depth_values = np.asarray(track.depths_m, dtype=np.float32)
            median_depth = float(np.median(depth_values))
            temporal_uncertainty = 1.4826 * float(np.median(np.abs(depth_values - median_depth)))
            current_uncertainty = detection.depth_uncertainty_m or 0.0
            smoothed.append(
                replace(
                    detection,
                    median_depth_m=median_depth,
                    depth_uncertainty_m=max(current_uncertainty, temporal_uncertainty),
                )
            )

        self._remove_stale_tracks()
        return smoothed

    def _find_match(self, detection: Detection, excluded: set[int]) -> Optional[int]:
        best_track_id = None
        best_iou = self.min_iou
        for track_id, track in self._tracks.items():
            if track_id in excluded or track.class_name != detection.class_name:
                continue
            iou = _bbox_iou(track.bbox, detection.bbox)
            if iou >= best_iou:
                best_track_id = track_id
                best_iou = iou
        return best_track_id

    def _create_track(self, detection: Detection) -> int:
        track_id = self._next_track_id
        self._next_track_id += 1
        self._tracks[track_id] = _DepthTrack(
            class_name=detection.class_name,
            bbox=detection.bbox,
            depths_m=deque(maxlen=self.window_size),
            last_frame_index=self._frame_index,
        )
        return track_id

    def _remove_stale_tracks(self) -> None:
        stale = [
            track_id
            for track_id, track in self._tracks.items()
            if self._frame_index - track.last_frame_index > self.max_track_age
        ]
        for track_id in stale:
            del self._tracks[track_id]


def _bbox_iou(first: BBox, second: BBox) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    first_area = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
    second_area = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
    union = first_area + second_area - intersection
    return float(intersection / union) if union else 0.0
