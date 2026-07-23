"""Stereo depth helpers for paired camera frames."""

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from haptos.types import Detection, StereoDepthSummary
from haptos.cv.stereo_calibration import StereoCalibration, load_stereo_rectification


@dataclass(frozen=True)
class StereoDepthFrame:
    """Stereo output for one frame pair."""

    summary: StereoDepthSummary
    disparity_px: np.ndarray
    depth_m: Optional[np.ndarray] = None


@dataclass(frozen=True)
class DetectionDepthMeasurement:
    """Depth measurement details for one detection bbox."""

    sample_bbox: tuple[int, int, int, int]
    median_depth_m: Optional[float]
    valid_pixel_count: int


class StereoDepthEstimator:
    """Compute a coarse stereo disparity/depth summary from left and right frames."""

    def __init__(
        self,
        num_disparities: int = 64,
        block_size: int = 5,
        baseline_m: Optional[float] = None,
        focal_px: Optional[float] = None,
        calibration_path: Optional[str] = None,
    ):
        if num_disparities <= 0 or num_disparities % 16 != 0:
            raise ValueError("--stereo-num-disparities must be a positive multiple of 16")
        if block_size < 3 or block_size % 2 == 0:
            raise ValueError("--stereo-block-size must be an odd integer >= 3")
        self.calibration = load_stereo_rectification(calibration_path) if calibration_path else None
        if self.calibration is not None:
            baseline_m = self.calibration.baseline_m or baseline_m
            focal_px = self.calibration.focal_px or focal_px

        if (baseline_m is None) != (focal_px is None):
            raise ValueError("--stereo-baseline-m and --stereo-focal-px must be provided together")
        if baseline_m is not None and baseline_m <= 0:
            raise ValueError("--stereo-baseline-m must be positive")
        if focal_px is not None and focal_px <= 0:
            raise ValueError("--stereo-focal-px must be positive")

        self.baseline_m = baseline_m
        self.focal_px = focal_px
        self.matcher = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=num_disparities,
            blockSize=block_size,
            P1=8 * 3 * block_size**2,
            P2=32 * 3 * block_size**2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )

    def estimate(self, left_frame, right_frame) -> StereoDepthSummary:
        """Return a compact depth summary for a pair of BGR frames."""

        return self.estimate_frame(left_frame, right_frame).summary

    def estimate_frame(self, left_frame, right_frame) -> StereoDepthFrame:
        """Return disparity, optional metric depth, and a compact summary."""

        if self.calibration is not None:
            left_frame, right_frame = self.calibration.rectify(left_frame, right_frame)

        left_gray = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)
        if right_gray.shape != left_gray.shape:
            right_gray = cv2.resize(right_gray, (left_gray.shape[1], left_gray.shape[0]))

        disparity = self.matcher.compute(left_gray, right_gray).astype(np.float32) / 16.0
        depth = disparity_to_depth(disparity, baseline_m=self.baseline_m, focal_px=self.focal_px)
        summary = summarize_disparity(
            disparity,
            baseline_m=self.baseline_m,
            focal_px=self.focal_px,
        )
        return StereoDepthFrame(summary=summary, disparity_px=disparity, depth_m=depth)


def attach_depth_to_detections(
    detections: list[Detection],
    depth_m: Optional[np.ndarray],
    min_valid_depth_m: float = 0.05,
    bbox_scale: float = 0.6,
) -> list[Detection]:
    """Attach median metric depth from the center area of each detection box."""

    if depth_m is None:
        return detections
    if bbox_scale <= 0.0 or bbox_scale > 1.0:
        raise ValueError("bbox_scale must be greater than 0 and less than or equal to 1")

    enriched: list[Detection] = []
    for detection in detections:
        measurement = measure_detection_depth(
            detection,
            depth_m,
            min_valid_depth_m=min_valid_depth_m,
            bbox_scale=bbox_scale,
        )
        if measurement.median_depth_m is None:
            enriched.append(detection)
            continue

        enriched.append(
            Detection(
                class_name=detection.class_name,
                confidence=detection.confidence,
                bbox=detection.bbox,
                region=detection.region,
                is_obstacle=detection.is_obstacle,
                median_depth_m=measurement.median_depth_m,
                depth_pixel_count=measurement.valid_pixel_count,
            )
        )

    return enriched


def measure_detection_depth(
    detection: Detection,
    depth_m: np.ndarray,
    min_valid_depth_m: float = 0.05,
    bbox_scale: float = 0.6,
) -> DetectionDepthMeasurement:
    """Measure median depth inside the center area of one detection box."""

    if bbox_scale <= 0.0 or bbox_scale > 1.0:
        raise ValueError("bbox_scale must be greater than 0 and less than or equal to 1")

    height, width = depth_m.shape[:2]
    x1, y1, x2, y2 = _scale_bbox_around_center(detection.bbox, bbox_scale)
    x1 = max(0, min(width, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height, y1))
    y2 = max(0, min(height, y2))

    crop = depth_m[y1:y2, x1:x2]
    valid_depth = crop[np.isfinite(crop) & (crop > min_valid_depth_m)]
    if valid_depth.size == 0:
        return DetectionDepthMeasurement(
            sample_bbox=(x1, y1, x2, y2),
            median_depth_m=None,
            valid_pixel_count=0,
        )

    return DetectionDepthMeasurement(
        sample_bbox=(x1, y1, x2, y2),
        median_depth_m=float(np.median(valid_depth)),
        valid_pixel_count=int(valid_depth.size),
    )


def _scale_bbox_around_center(bbox, scale: float):
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    half_width = (x2 - x1) * scale / 2.0
    half_height = (y2 - y1) * scale / 2.0
    return (
        int(round(center_x - half_width)),
        int(round(center_y - half_height)),
        int(round(center_x + half_width)),
        int(round(center_y + half_height)),
    )


def disparity_to_depth(
    disparity: np.ndarray,
    baseline_m: Optional[float],
    focal_px: Optional[float],
    min_valid_disparity_px: float = 0.5,
) -> Optional[np.ndarray]:
    """Convert disparity pixels to metric depth when calibration values exist."""

    if baseline_m is None or focal_px is None:
        return None

    depth = np.full(disparity.shape, np.nan, dtype=np.float32)
    valid = np.isfinite(disparity) & (disparity > min_valid_disparity_px)
    depth[valid] = (baseline_m * focal_px) / disparity[valid]
    return depth


def summarize_disparity(
    disparity: np.ndarray,
    baseline_m: Optional[float] = None,
    focal_px: Optional[float] = None,
    min_valid_disparity_px: float = 0.5,
) -> StereoDepthSummary:
    """Summarize a disparity map and optionally convert it to metric depth."""

    valid_disparity = disparity[np.isfinite(disparity) & (disparity > min_valid_disparity_px)]
    if valid_disparity.size == 0:
        return StereoDepthSummary(fault_state="no_valid_disparity", valid_pixel_count=0)

    median_disparity = float(np.median(valid_disparity))
    if baseline_m is None or focal_px is None:
        return StereoDepthSummary(
            fault_state="uncalibrated",
            valid_pixel_count=int(valid_disparity.size),
            median_disparity_px=median_disparity,
        )

    depth = (baseline_m * focal_px) / valid_disparity
    depth = depth[np.isfinite(depth) & (depth > 0.0)]
    if depth.size == 0:
        return StereoDepthSummary(
            fault_state="no_valid_depth",
            valid_pixel_count=int(valid_disparity.size),
            median_disparity_px=median_disparity,
        )

    return StereoDepthSummary(
        fault_state="none",
        valid_pixel_count=int(valid_disparity.size),
        median_disparity_px=median_disparity,
        nearest_depth_m=float(np.min(depth)),
        median_depth_m=float(np.median(depth)),
        farthest_depth_m=float(np.max(depth)),
    )
