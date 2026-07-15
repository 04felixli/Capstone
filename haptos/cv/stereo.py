"""Stereo depth helpers for paired camera frames."""

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from haptos.types import Detection, StereoDepthSummary


@dataclass(frozen=True)
class StereoDepthFrame:
    """Stereo output for one frame pair."""

    summary: StereoDepthSummary
    disparity_px: np.ndarray
    depth_m: Optional[np.ndarray] = None


class StereoDepthEstimator:
    """Compute a coarse stereo disparity/depth summary from left and right frames."""

    def __init__(
        self,
        num_disparities: int = 64,
        block_size: int = 5,
        baseline_m: Optional[float] = None,
        focal_px: Optional[float] = None,
    ):
        if num_disparities <= 0 or num_disparities % 16 != 0:
            raise ValueError("--stereo-num-disparities must be a positive multiple of 16")
        if block_size < 3 or block_size % 2 == 0:
            raise ValueError("--stereo-block-size must be an odd integer >= 3")
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
) -> list[Detection]:
    """Attach median metric depth from each detection bounding box."""

    if depth_m is None:
        return detections

    enriched: list[Detection] = []
    height, width = depth_m.shape[:2]
    for detection in detections:
        x1, y1, x2, y2 = detection.bbox
        x1 = max(0, min(width, x1))
        x2 = max(0, min(width, x2))
        y1 = max(0, min(height, y1))
        y2 = max(0, min(height, y2))

        crop = depth_m[y1:y2, x1:x2]
        valid_depth = crop[np.isfinite(crop) & (crop > min_valid_depth_m)]
        if valid_depth.size == 0:
            enriched.append(detection)
            continue

        enriched.append(
            Detection(
                class_name=detection.class_name,
                confidence=detection.confidence,
                bbox=detection.bbox,
                region=detection.region,
                is_obstacle=detection.is_obstacle,
                median_depth_m=float(np.median(valid_depth)),
                depth_pixel_count=int(valid_depth.size),
            )
        )

    return enriched


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
