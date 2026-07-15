"""Utility helpers for drawing, FPS tracking, and JSONL logging."""

import json
import time
from pathlib import Path
from typing import Iterable, Optional

import cv2

from haptos.config import CENTER_REGION, LEFT_REGION, RIGHT_REGION
from haptos.types import Detection, FrameResult


class FPSCounter:
    """Small moving FPS helper based on time between processed frames."""

    def __init__(self, smoothing: float = 0.9):
        self.smoothing = smoothing
        self._last_time: Optional[float] = None
        self._fps = 0.0

    def update(self) -> float:
        now = time.perf_counter()
        if self._last_time is None:
            self._last_time = now
            return 0.0

        elapsed = now - self._last_time
        self._last_time = now
        if elapsed <= 0:
            return self._fps

        instant_fps = 1.0 / elapsed
        if self._fps == 0.0:
            self._fps = instant_fps
        else:
            self._fps = self.smoothing * self._fps + (1.0 - self.smoothing) * instant_fps
        return self._fps


class JsonlLogger:
    """Append frame results as newline-delimited JSON for later analysis."""

    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("a", encoding="utf-8")

    def write(self, result: FrameResult) -> None:
        self._file.write(json.dumps(result.to_dict()) + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()


def draw_regions(frame) -> None:
    """Draw vertical dividers for LEFT/CENTER/RIGHT navigation regions."""

    height, width = frame.shape[:2]
    x1 = width // 3
    x2 = (2 * width) // 3
    color = (220, 220, 220)

    cv2.line(frame, (x1, 0), (x1, height), color, 1)
    cv2.line(frame, (x2, 0), (x2, height), color, 1)

    labels = [
        (LEFT_REGION, 10),
        (CENTER_REGION, x1 + 10),
        (RIGHT_REGION, x2 + 10),
    ]
    for label, x in labels:
        cv2.putText(frame, label, (x, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


def draw_detections(frame, detections: Iterable[Detection]) -> None:
    """Draw bounding boxes and labels on a frame in place."""

    for detection in detections:
        x1, y1, x2, y2 = detection.bbox
        color = (0, 0, 255) if detection.is_obstacle else (0, 180, 0)
        label = f"{detection.class_name} {detection.region} {detection.confidence:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            label,
            (x1, max(y1 - 8, 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )


def draw_overlay(frame, result: FrameResult) -> None:
    draw_regions(frame)
    draw_detections(frame, result.detections)
    status = f"command={result.command} fps={result.fps:.1f}"
    cv2.putText(frame, status, (10, frame.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    if result.lidar_summary is not None:
        lidar = result.lidar_summary
        nearest = _format_optional_distance(lidar.nearest_distance_m)
        lidar_status = f"lidar={lidar.fault_state} points={lidar.point_count} nearest={nearest}"
        cv2.putText(frame, lidar_status, (10, frame.shape[0] - 46), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)

    if result.stereo_depth_summary is not None:
        stereo = result.stereo_depth_summary
        depth = _format_optional_distance(stereo.median_depth_m)
        stereo_status = f"stereo={stereo.fault_state} pixels={stereo.valid_pixel_count} median={depth}"
        cv2.putText(frame, stereo_status, (10, frame.shape[0] - 76), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)


def format_console_result(result: FrameResult) -> str:
    if result.detections:
        detections = ", ".join(
            f"{d.class_name}:{d.region.lower()}:{d.confidence:.2f}"
            for d in result.detections
        )
    else:
        detections = "none"

    lidar = ""
    if result.lidar_summary is not None:
        summary = result.lidar_summary
        lidar = (
            f" | lidar={summary.fault_state}:"
            f"points={summary.point_count}:"
            f"nearest={_format_optional_distance(summary.nearest_distance_m)}"
        )

    stereo_depth = ""
    if result.stereo_depth_summary is not None:
        summary = result.stereo_depth_summary
        stereo_depth = (
            f" | stereo={summary.fault_state}:"
            f"pixels={summary.valid_pixel_count}:"
            f"median_disp={_format_optional_number(summary.median_disparity_px)}:"
            f"median_depth={_format_optional_distance(summary.median_depth_m)}"
        )

    return f"Frame {result.frame_index} | command={result.command} | detections={detections}{lidar}{stereo_depth}"


def _format_optional_distance(distance_m: Optional[float]) -> str:
    if distance_m is None:
        return "n/a"
    return f"{distance_m:.2f}m"


def _format_optional_number(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}"
