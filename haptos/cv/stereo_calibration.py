"""Stereo checkerboard calibration helpers."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

RECTIFICATION_KIND_CHECKERBOARD = "checkerboard_calibration"
RECTIFICATION_KIND_UNCALIBRATED = "uncalibrated_rectification"


@dataclass(frozen=True)
class StereoCalibration:
    """Saved stereo camera calibration and rectification maps."""

    image_size: Tuple[int, int]
    camera_matrix_left: np.ndarray
    dist_coeffs_left: np.ndarray
    camera_matrix_right: np.ndarray
    dist_coeffs_right: np.ndarray
    rotation: np.ndarray
    translation: np.ndarray
    essential_matrix: np.ndarray
    fundamental_matrix: np.ndarray
    rectification_left: np.ndarray
    rectification_right: np.ndarray
    projection_left: np.ndarray
    projection_right: np.ndarray
    disparity_to_depth_map: np.ndarray
    reprojection_error: float
    left_maps: Optional[Tuple[np.ndarray, np.ndarray]] = None
    right_maps: Optional[Tuple[np.ndarray, np.ndarray]] = None

    def __post_init__(self) -> None:
        if self.left_maps is None:
            object.__setattr__(
                self,
                "left_maps",
                _rectification_maps(
                    self.camera_matrix_left,
                    self.dist_coeffs_left,
                    self.rectification_left,
                    self.projection_left,
                    self.image_size,
                ),
            )
        if self.right_maps is None:
            object.__setattr__(
                self,
                "right_maps",
                _rectification_maps(
                    self.camera_matrix_right,
                    self.dist_coeffs_right,
                    self.rectification_right,
                    self.projection_right,
                    self.image_size,
                ),
            )

    @property
    def focal_px(self) -> float:
        return float(self.projection_left[0, 0])

    @property
    def baseline_m(self) -> float:
        focal = self.focal_px
        if focal == 0:
            return 0.0
        return abs(float(self.projection_right[0, 3]) / focal)

    def save(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            output_path,
            kind=np.array([RECTIFICATION_KIND_CHECKERBOARD]),
            image_size=np.array(self.image_size, dtype=np.int32),
            camera_matrix_left=self.camera_matrix_left,
            dist_coeffs_left=self.dist_coeffs_left,
            camera_matrix_right=self.camera_matrix_right,
            dist_coeffs_right=self.dist_coeffs_right,
            rotation=self.rotation,
            translation=self.translation,
            essential_matrix=self.essential_matrix,
            fundamental_matrix=self.fundamental_matrix,
            rectification_left=self.rectification_left,
            rectification_right=self.rectification_right,
            projection_left=self.projection_left,
            projection_right=self.projection_right,
            disparity_to_depth_map=self.disparity_to_depth_map,
            reprojection_error=np.array([self.reprojection_error], dtype=np.float64),
        )

    @classmethod
    def load(cls, path: str | Path) -> "StereoCalibration":
        data = np.load(Path(path))
        return cls(
            image_size=tuple(int(v) for v in data["image_size"]),
            camera_matrix_left=data["camera_matrix_left"],
            dist_coeffs_left=data["dist_coeffs_left"],
            camera_matrix_right=data["camera_matrix_right"],
            dist_coeffs_right=data["dist_coeffs_right"],
            rotation=data["rotation"],
            translation=data["translation"],
            essential_matrix=data["essential_matrix"],
            fundamental_matrix=data["fundamental_matrix"],
            rectification_left=data["rectification_left"],
            rectification_right=data["rectification_right"],
            projection_left=data["projection_left"],
            projection_right=data["projection_right"],
            disparity_to_depth_map=data["disparity_to_depth_map"],
            reprojection_error=float(data["reprojection_error"][0]),
        )

    def rectify(self, left_frame, right_frame):
        """Undistort/rectify a stereo frame pair."""

        width, height = self.image_size
        if left_frame.shape[1] != width or left_frame.shape[0] != height:
            left_frame = cv2.resize(left_frame, self.image_size)
        if right_frame.shape[1] != width or right_frame.shape[0] != height:
            right_frame = cv2.resize(right_frame, self.image_size)

        left_map_x, left_map_y = self.left_maps
        right_map_x, right_map_y = self.right_maps
        left_rectified = cv2.remap(left_frame, left_map_x, left_map_y, cv2.INTER_LINEAR)
        right_rectified = cv2.remap(right_frame, right_map_x, right_map_y, cv2.INTER_LINEAR)
        return left_rectified, right_rectified


@dataclass(frozen=True)
class UncalibratedStereoRectification:
    """Saved rectification homographies estimated from scene feature matches."""

    image_size: Tuple[int, int]
    homography_left: np.ndarray
    homography_right: np.ndarray
    fundamental_matrix: np.ndarray
    inlier_count: int
    match_count: int

    @property
    def focal_px(self) -> Optional[float]:
        return None

    @property
    def baseline_m(self) -> Optional[float]:
        return None

    def save(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            output_path,
            kind=np.array([RECTIFICATION_KIND_UNCALIBRATED]),
            image_size=np.array(self.image_size, dtype=np.int32),
            homography_left=self.homography_left,
            homography_right=self.homography_right,
            fundamental_matrix=self.fundamental_matrix,
            inlier_count=np.array([self.inlier_count], dtype=np.int32),
            match_count=np.array([self.match_count], dtype=np.int32),
        )

    @classmethod
    def load(cls, path: str | Path) -> "UncalibratedStereoRectification":
        data = np.load(Path(path))
        return cls(
            image_size=tuple(int(v) for v in data["image_size"]),
            homography_left=data["homography_left"],
            homography_right=data["homography_right"],
            fundamental_matrix=data["fundamental_matrix"],
            inlier_count=int(data["inlier_count"][0]),
            match_count=int(data["match_count"][0]),
        )

    def rectify(self, left_frame, right_frame):
        """Rectify a stereo frame pair with projective homographies."""

        width, height = self.image_size
        if left_frame.shape[1] != width or left_frame.shape[0] != height:
            left_frame = cv2.resize(left_frame, self.image_size)
        if right_frame.shape[1] != width or right_frame.shape[0] != height:
            right_frame = cv2.resize(right_frame, self.image_size)

        left_rectified = cv2.warpPerspective(left_frame, self.homography_left, self.image_size)
        right_rectified = cv2.warpPerspective(right_frame, self.homography_right, self.image_size)
        return left_rectified, right_rectified


def load_stereo_rectification(path: str | Path):
    """Load either full checkerboard calibration or uncalibrated rectification."""

    data = np.load(Path(path))
    kind = _npz_kind(data)
    data.close()
    if kind == RECTIFICATION_KIND_UNCALIBRATED:
        return UncalibratedStereoRectification.load(path)
    return StereoCalibration.load(path)


def estimate_uncalibrated_rectification_from_images(
    left_path: str | Path,
    right_path: str | Path,
    max_features: int = 4000,
    keep_matches: int = 800,
    min_inliers: int = 80,
    ransac_reproj_threshold: float = 1.5,
) -> UncalibratedStereoRectification:
    """Estimate stereo rectification from ordinary scene feature matches."""

    left_gray = _read_gray(Path(left_path))
    right_gray = _read_gray(Path(right_path))
    if left_gray.shape != right_gray.shape:
        raise ValueError("Left and right images must have the same size")

    orb = cv2.ORB_create(nfeatures=max_features)
    left_keypoints, left_descriptors = orb.detectAndCompute(left_gray, None)
    right_keypoints, right_descriptors = orb.detectAndCompute(right_gray, None)
    if left_descriptors is None or right_descriptors is None:
        raise ValueError("Could not find enough feature descriptors in both images")

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(matcher.match(left_descriptors, right_descriptors), key=lambda match: match.distance)
    matches = matches[:keep_matches]
    if len(matches) < min_inliers:
        raise ValueError(f"Need at least {min_inliers} feature matches; found {len(matches)}")

    left_points = np.float32([left_keypoints[match.queryIdx].pt for match in matches])
    right_points = np.float32([right_keypoints[match.trainIdx].pt for match in matches])
    fundamental, inlier_mask = cv2.findFundamentalMat(
        left_points,
        right_points,
        cv2.FM_RANSAC,
        ransac_reproj_threshold,
        0.99,
    )
    if fundamental is None or inlier_mask is None:
        raise ValueError("Could not estimate a fundamental matrix")

    inliers = inlier_mask.ravel().astype(bool)
    inlier_count = int(np.count_nonzero(inliers))
    if inlier_count < min_inliers:
        raise ValueError(f"Need at least {min_inliers} inlier matches; found {inlier_count}")

    image_size = (left_gray.shape[1], left_gray.shape[0])
    ok, homography_left, homography_right = cv2.stereoRectifyUncalibrated(
        left_points[inliers],
        right_points[inliers],
        fundamental,
        image_size,
    )
    if not ok:
        raise ValueError("OpenCV could not compute uncalibrated stereo rectification")

    return UncalibratedStereoRectification(
        image_size=image_size,
        homography_left=homography_left,
        homography_right=homography_right,
        fundamental_matrix=fundamental,
        inlier_count=inlier_count,
        match_count=len(matches),
    )


def make_rectified_preview(rectification, left_path: str | Path, right_path: str | Path):
    """Create a side-by-side rectified preview with horizontal guide lines."""

    left = cv2.imread(str(left_path))
    right = cv2.imread(str(right_path))
    if left is None:
        raise ValueError(f"Could not read image: {left_path}")
    if right is None:
        raise ValueError(f"Could not read image: {right_path}")

    left_rectified, right_rectified = rectification.rectify(left, right)
    preview = cv2.hconcat([left_rectified, right_rectified])
    height, width = left_rectified.shape[:2]
    for y in range(40, height, 40):
        cv2.line(preview, (0, y), (width * 2, y), (0, 255, 255), 1)
    cv2.putText(preview, "LEFT RECTIFIED", (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(preview, "RIGHT RECTIFIED", (width + 12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    return preview


def calibrate_stereo_from_images(
    image_pairs: Sequence[Tuple[Path, Path]],
    pattern_size: Tuple[int, int],
    square_size_m: float,
) -> Tuple[StereoCalibration, int]:
    """Calibrate a stereo pair from checkerboard image pairs.

    pattern_size is the number of inner corners as (columns, rows).
    """

    if square_size_m <= 0:
        raise ValueError("square_size_m must be positive")

    object_template = _checkerboard_object_points(pattern_size, square_size_m)
    object_points = []
    left_points = []
    right_points = []
    image_size: Optional[Tuple[int, int]] = None

    for left_path, right_path in image_pairs:
        left_gray = _read_gray(left_path)
        right_gray = _read_gray(right_path)
        if left_gray.shape != right_gray.shape:
            raise ValueError(f"Image pair sizes differ: {left_path} and {right_path}")
        current_size = (left_gray.shape[1], left_gray.shape[0])
        if image_size is None:
            image_size = current_size
        elif image_size != current_size:
            raise ValueError("All calibration images must have the same size")

        left_found, left_corners = cv2.findChessboardCorners(left_gray, pattern_size)
        right_found, right_corners = cv2.findChessboardCorners(right_gray, pattern_size)
        if not left_found or not right_found:
            continue

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        left_corners = cv2.cornerSubPix(left_gray, left_corners, (11, 11), (-1, -1), criteria)
        right_corners = cv2.cornerSubPix(right_gray, right_corners, (11, 11), (-1, -1), criteria)

        object_points.append(object_template)
        left_points.append(left_corners)
        right_points.append(right_corners)

    if image_size is None:
        raise ValueError("No calibration images found")
    if len(object_points) < 8:
        raise ValueError(f"Need at least 8 valid checkerboard pairs; found {len(object_points)}")

    _, camera_left, dist_left, _, _ = cv2.calibrateCamera(
        object_points,
        left_points,
        image_size,
        None,
        None,
    )
    _, camera_right, dist_right, _, _ = cv2.calibrateCamera(
        object_points,
        right_points,
        image_size,
        None,
        None,
    )

    flags = cv2.CALIB_FIX_INTRINSIC
    error, camera_left, dist_left, camera_right, dist_right, rotation, translation, essential, fundamental = (
        cv2.stereoCalibrate(
            object_points,
            left_points,
            right_points,
            camera_left,
            dist_left,
            camera_right,
            dist_right,
            image_size,
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5),
            flags=flags,
        )
    )

    rect_left, rect_right, proj_left, proj_right, q_matrix, _, _ = cv2.stereoRectify(
        camera_left,
        dist_left,
        camera_right,
        dist_right,
        image_size,
        rotation,
        translation,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0,
    )

    calibration = StereoCalibration(
        image_size=image_size,
        camera_matrix_left=camera_left,
        dist_coeffs_left=dist_left,
        camera_matrix_right=camera_right,
        dist_coeffs_right=dist_right,
        rotation=rotation,
        translation=translation,
        essential_matrix=essential,
        fundamental_matrix=fundamental,
        rectification_left=rect_left,
        rectification_right=rect_right,
        projection_left=proj_left,
        projection_right=proj_right,
        disparity_to_depth_map=q_matrix,
        reprojection_error=float(error),
    )
    return calibration, len(object_points)


def find_image_pairs(image_dir: str | Path) -> list[Tuple[Path, Path]]:
    """Find left_###/right_### calibration image pairs in a directory."""

    directory = Path(image_dir)
    pairs = []
    for left_path in sorted(directory.glob("left_*.jpg")):
        suffix = left_path.name.removeprefix("left_")
        right_path = directory / f"right_{suffix}"
        if right_path.exists():
            pairs.append((left_path, right_path))
    return pairs


def _checkerboard_object_points(pattern_size: Tuple[int, int], square_size_m: float) -> np.ndarray:
    cols, rows = pattern_size
    points = np.zeros((rows * cols, 3), np.float32)
    points[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    points *= square_size_m
    return points


def _read_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image


def _rectification_maps(camera_matrix, dist_coeffs, rectification, projection, image_size):
    return cv2.initUndistortRectifyMap(
        camera_matrix,
        dist_coeffs,
        rectification,
        projection,
        image_size,
        cv2.CV_32FC1,
    )


def _npz_kind(data) -> str:
    if "kind" not in data.files:
        return RECTIFICATION_KIND_CHECKERBOARD
    kind = data["kind"][0]
    if isinstance(kind, bytes):
        return kind.decode("utf-8")
    return str(kind)
