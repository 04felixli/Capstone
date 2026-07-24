"""Stereo depth helpers for paired camera frames."""

from dataclasses import dataclass, replace
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
    valid_mask: Optional[np.ndarray] = None


@dataclass(frozen=True)
class DetectionDepthMeasurement:
    """Depth measurement details for one detection bbox."""

    sample_bbox: tuple[int, int, int, int]
    median_depth_m: Optional[float]
    valid_pixel_count: int
    depth_uncertainty_m: Optional[float] = None
    valid_fraction: float = 0.0
    fault_state: str = "none"


class StereoDepthEstimator:
    """Compute a coarse stereo disparity/depth summary from left and right frames."""

    def __init__(
        self,
        num_disparities: int = 64,
        block_size: int = 5,
        baseline_m: Optional[float] = None,
        focal_px: Optional[float] = None,
        calibration_path: Optional[str] = None,
        min_valid_disparity_px: float = 0.5,
        max_depth_m: float = 8.0,
        local_disparity_tolerance_px: float = 2.0,
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
        if min_valid_disparity_px <= 0:
            raise ValueError("min_valid_disparity_px must be positive")
        if max_depth_m <= 0:
            raise ValueError("max_depth_m must be positive")
        if local_disparity_tolerance_px <= 0:
            raise ValueError("local_disparity_tolerance_px must be positive")

        self.baseline_m = baseline_m
        self.focal_px = focal_px
        self.min_valid_disparity_px = min_valid_disparity_px
        self.max_depth_m = max_depth_m
        self.local_disparity_tolerance_px = local_disparity_tolerance_px
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

    def estimate_frame(
        self,
        left_frame,
        right_frame,
        *,
        frame_skew_ms: Optional[float] = None,
    ) -> StereoDepthFrame:
        """Return disparity, optional metric depth, and a compact summary."""

        if self.calibration is not None:
            left_frame, right_frame = self.calibration.rectify(left_frame, right_frame)

        left_gray = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)
        if right_gray.shape != left_gray.shape:
            right_gray = cv2.resize(right_gray, (left_gray.shape[1], left_gray.shape[0]))

        raw_disparity = self.matcher.compute(left_gray, right_gray).astype(np.float32) / 16.0
        disparity, valid_mask = filter_disparity_map(
            raw_disparity,
            min_valid_disparity_px=self.min_valid_disparity_px,
            local_tolerance_px=self.local_disparity_tolerance_px,
        )
        depth = disparity_to_depth(
            disparity,
            baseline_m=self.baseline_m,
            focal_px=self.focal_px,
            min_valid_disparity_px=self.min_valid_disparity_px,
            max_depth_m=self.max_depth_m,
        )
        summary = summarize_disparity(
            disparity,
            baseline_m=self.baseline_m,
            focal_px=self.focal_px,
            min_valid_disparity_px=self.min_valid_disparity_px,
            max_depth_m=self.max_depth_m,
        )
        summary = replace(summary, frame_skew_ms=frame_skew_ms)
        return StereoDepthFrame(
            summary=summary,
            disparity_px=disparity,
            depth_m=depth,
            valid_mask=valid_mask,
        )


def attach_depth_to_detections(
    detections: list[Detection],
    depth_m: Optional[np.ndarray],
    min_valid_depth_m: float = 0.05,
    max_valid_depth_m: float = 8.0,
    bbox_scale: float = 0.6,
    min_valid_fraction: float = 0.05,
    max_relative_uncertainty: float = 1.0,
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
            max_valid_depth_m=max_valid_depth_m,
            bbox_scale=bbox_scale,
            min_valid_fraction=min_valid_fraction,
            max_relative_uncertainty=max_relative_uncertainty,
        )
        enriched.append(
            replace(
                detection,
                median_depth_m=measurement.median_depth_m,
                depth_pixel_count=measurement.valid_pixel_count,
                depth_uncertainty_m=measurement.depth_uncertainty_m,
                depth_valid_fraction=measurement.valid_fraction,
                depth_fault_state=measurement.fault_state,
            )
        )

    return enriched


def measure_detection_depth(
    detection: Detection,
    depth_m: np.ndarray,
    min_valid_depth_m: float = 0.05,
    max_valid_depth_m: float = 8.0,
    bbox_scale: float = 0.6,
    min_valid_fraction: float = 0.05,
    max_relative_uncertainty: float = 1.0,
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

    return measure_depth_in_bbox(
        depth_m,
        (x1, y1, x2, y2),
        min_valid_depth_m=min_valid_depth_m,
        max_valid_depth_m=max_valid_depth_m,
        min_valid_fraction=min_valid_fraction,
        max_relative_uncertainty=max_relative_uncertainty,
    )


def measure_center_depth(
    depth_m: np.ndarray,
    box_size_px: int = 80,
    min_valid_depth_m: float = 0.05,
    max_valid_depth_m: float = 8.0,
    min_valid_fraction: float = 0.05,
    max_relative_uncertainty: float = 1.0,
) -> DetectionDepthMeasurement:
    """Measure median depth in a square centered in the depth image."""

    if box_size_px <= 0:
        raise ValueError("box_size_px must be positive")

    height, width = depth_m.shape[:2]
    center_x = width / 2.0
    center_y = height / 2.0
    half_size = box_size_px / 2.0
    bbox = (
        int(round(center_x - half_size)),
        int(round(center_y - half_size)),
        int(round(center_x + half_size)),
        int(round(center_y + half_size)),
    )
    return measure_depth_in_bbox(
        depth_m,
        bbox,
        min_valid_depth_m=min_valid_depth_m,
        max_valid_depth_m=max_valid_depth_m,
        min_valid_fraction=min_valid_fraction,
        max_relative_uncertainty=max_relative_uncertainty,
    )


def measure_depth_in_bbox(
    depth_m: np.ndarray,
    bbox,
    min_valid_depth_m: float = 0.05,
    max_valid_depth_m: float = 8.0,
    min_valid_fraction: float = 0.05,
    max_relative_uncertainty: float = 1.0,
) -> DetectionDepthMeasurement:
    """Measure median depth inside a pixel bbox."""

    height, width = depth_m.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(width, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height, y1))
    y2 = max(0, min(height, y2))

    if max_valid_depth_m <= min_valid_depth_m:
        raise ValueError("max_valid_depth_m must be greater than min_valid_depth_m")
    if min_valid_fraction < 0.0 or min_valid_fraction > 1.0:
        raise ValueError("min_valid_fraction must be between 0 and 1")
    if max_relative_uncertainty <= 0.0:
        raise ValueError("max_relative_uncertainty must be positive")

    crop = depth_m[y1:y2, x1:x2]
    total_pixel_count = int(crop.size)
    valid_depth = crop[
        np.isfinite(crop)
        & (crop > min_valid_depth_m)
        & (crop <= max_valid_depth_m)
    ]
    valid_fraction = float(valid_depth.size / total_pixel_count) if total_pixel_count else 0.0
    if valid_depth.size == 0:
        return DetectionDepthMeasurement(
            sample_bbox=(x1, y1, x2, y2),
            median_depth_m=None,
            valid_pixel_count=0,
            valid_fraction=0.0,
            fault_state="no_valid_depth",
        )

    median_depth = float(np.median(valid_depth))
    absolute_deviation = np.abs(valid_depth - median_depth)
    mad = float(np.median(absolute_deviation))
    uncertainty_m = 1.4826 * mad

    if mad > 0.0:
        inlier_limit = 3.0 * uncertainty_m
        inliers = valid_depth[absolute_deviation <= inlier_limit]
        if inliers.size:
            valid_depth = inliers
            median_depth = float(np.median(valid_depth))
            uncertainty_m = 1.4826 * float(np.median(np.abs(valid_depth - median_depth)))

    if valid_fraction < min_valid_fraction:
        return DetectionDepthMeasurement(
            sample_bbox=(x1, y1, x2, y2),
            median_depth_m=None,
            valid_pixel_count=int(valid_depth.size),
            depth_uncertainty_m=uncertainty_m,
            valid_fraction=valid_fraction,
            fault_state="insufficient_valid_pixels",
        )

    relative_uncertainty = uncertainty_m / max(median_depth, 1e-6)
    if relative_uncertainty > max_relative_uncertainty:
        return DetectionDepthMeasurement(
            sample_bbox=(x1, y1, x2, y2),
            median_depth_m=None,
            valid_pixel_count=int(valid_depth.size),
            depth_uncertainty_m=uncertainty_m,
            valid_fraction=valid_fraction,
            fault_state="high_uncertainty",
        )

    return DetectionDepthMeasurement(
        sample_bbox=(x1, y1, x2, y2),
        median_depth_m=median_depth,
        valid_pixel_count=int(valid_depth.size),
        depth_uncertainty_m=uncertainty_m,
        valid_fraction=valid_fraction,
        fault_state="none",
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
    max_depth_m: Optional[float] = None,
) -> Optional[np.ndarray]:
    """Convert disparity pixels to metric depth when calibration values exist."""

    if baseline_m is None or focal_px is None:
        return None

    depth = np.full(disparity.shape, np.nan, dtype=np.float32)
    valid = np.isfinite(disparity) & (disparity > min_valid_disparity_px)
    depth[valid] = (baseline_m * focal_px) / disparity[valid]
    if max_depth_m is not None:
        depth[depth > max_depth_m] = np.nan
    return depth


def summarize_disparity(
    disparity: np.ndarray,
    baseline_m: Optional[float] = None,
    focal_px: Optional[float] = None,
    min_valid_disparity_px: float = 0.5,
    max_depth_m: Optional[float] = None,
) -> StereoDepthSummary:
    """Summarize a disparity map and optionally convert it to metric depth."""

    valid_disparity = disparity[np.isfinite(disparity) & (disparity > min_valid_disparity_px)]
    valid_fraction = float(valid_disparity.size / disparity.size) if disparity.size else 0.0
    if valid_disparity.size == 0:
        return StereoDepthSummary(
            fault_state="no_valid_disparity",
            valid_pixel_count=0,
            valid_fraction=0.0,
        )

    median_disparity = float(np.median(valid_disparity))
    if baseline_m is None or focal_px is None:
        return StereoDepthSummary(
            fault_state="uncalibrated",
            valid_pixel_count=int(valid_disparity.size),
            median_disparity_px=median_disparity,
            valid_fraction=valid_fraction,
        )

    depth = (baseline_m * focal_px) / valid_disparity
    depth = depth[np.isfinite(depth) & (depth > 0.0)]
    if max_depth_m is not None:
        depth = depth[depth <= max_depth_m]
    if depth.size == 0:
        return StereoDepthSummary(
            fault_state="no_valid_depth",
            valid_pixel_count=int(valid_disparity.size),
            median_disparity_px=median_disparity,
            valid_fraction=valid_fraction,
        )

    return StereoDepthSummary(
        fault_state="none",
        valid_pixel_count=int(valid_disparity.size),
        median_disparity_px=median_disparity,
        nearest_depth_m=float(np.min(depth)),
        median_depth_m=float(np.median(depth)),
        farthest_depth_m=float(np.max(depth)),
        valid_fraction=valid_fraction,
    )


def filter_disparity_map(
    disparity: np.ndarray,
    *,
    min_valid_disparity_px: float = 0.5,
    local_tolerance_px: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Reject invalid and locally inconsistent disparity pixels."""

    if disparity.ndim != 2:
        raise ValueError("disparity must be a two-dimensional array")
    if min_valid_disparity_px <= 0:
        raise ValueError("min_valid_disparity_px must be positive")
    if local_tolerance_px <= 0:
        raise ValueError("local_tolerance_px must be positive")

    disparity = disparity.astype(np.float32, copy=False)
    finite_valid = np.isfinite(disparity) & (disparity > min_valid_disparity_px)
    median_input = np.where(finite_valid, disparity, 0.0).astype(np.float32)
    local_median = cv2.medianBlur(median_input, 5)
    adaptive_tolerance = np.maximum(local_tolerance_px, np.abs(local_median) * 0.15)
    locally_consistent = np.abs(disparity - local_median) <= adaptive_tolerance
    valid_mask = finite_valid & (local_median > min_valid_disparity_px) & locally_consistent

    filtered = np.full(disparity.shape, np.nan, dtype=np.float32)
    filtered[valid_mask] = disparity[valid_mask]
    return filtered, valid_mask
