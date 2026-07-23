"""Tests for stereo disparity/depth helpers."""

import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

import numpy as np

from haptos.cv.stereo import attach_depth_to_detections, disparity_to_depth, measure_detection_depth, summarize_disparity
from haptos.cv.stereo_calibration import (
    UncalibratedStereoRectification,
    StereoCalibration,
    find_image_pairs,
    load_stereo_rectification,
)
from haptos.types import Detection


class StereoDepthSummaryTests(unittest.TestCase):
    def test_summarize_disparity_reports_uncalibrated_disparity(self):
        disparity = np.array([[0.0, 1.0], [2.0, -1.0]], dtype=np.float32)

        summary = summarize_disparity(disparity)

        self.assertEqual(summary.fault_state, "uncalibrated")
        self.assertEqual(summary.valid_pixel_count, 2)
        self.assertAlmostEqual(summary.median_disparity_px, 1.5)
        self.assertIsNone(summary.median_depth_m)

    def test_summarize_disparity_converts_to_metric_depth_when_calibrated(self):
        disparity = np.array([[10.0, 20.0]], dtype=np.float32)

        summary = summarize_disparity(disparity, baseline_m=0.06, focal_px=600.0)

        self.assertEqual(summary.fault_state, "none")
        self.assertEqual(summary.valid_pixel_count, 2)
        self.assertAlmostEqual(summary.nearest_depth_m, 1.8, places=5)
        self.assertAlmostEqual(summary.median_depth_m, 2.7, places=5)
        self.assertAlmostEqual(summary.farthest_depth_m, 3.6, places=5)

    def test_summarize_disparity_handles_no_valid_pixels(self):
        disparity = np.zeros((2, 2), dtype=np.float32)

        summary = summarize_disparity(disparity)

        self.assertEqual(summary.fault_state, "no_valid_disparity")
        self.assertEqual(summary.valid_pixel_count, 0)

    def test_disparity_to_depth_requires_calibration(self):
        disparity = np.array([[10.0]], dtype=np.float32)

        self.assertIsNone(disparity_to_depth(disparity, baseline_m=None, focal_px=None))

    def test_attach_depth_to_detections_uses_bbox_median_depth(self):
        depth = np.array(
            [
                [10.0, 10.0, 10.0, 10.0],
                [10.0, 1.0, 2.0, 10.0],
                [10.0, 3.0, 4.0, 10.0],
                [10.0, 10.0, 10.0, 10.0],
            ],
            dtype=np.float32,
        )
        detections = [
            Detection(
                class_name="person",
                confidence=0.9,
                bbox=(1, 1, 3, 3),
                region="CENTER",
                is_obstacle=True,
            )
        ]

        enriched = attach_depth_to_detections(detections, depth, bbox_scale=1.0)

        self.assertEqual(len(enriched), 1)
        self.assertAlmostEqual(enriched[0].median_depth_m, 2.5)
        self.assertEqual(enriched[0].depth_pixel_count, 4)
        self.assertEqual(enriched[0].class_name, "person")

    def test_attach_depth_to_detections_can_use_center_bbox_region(self):
        depth = np.full((10, 10), 10.0, dtype=np.float32)
        depth[4:6, 4:6] = 1.0
        detections = [
            Detection(
                class_name="person",
                confidence=0.9,
                bbox=(0, 0, 10, 10),
                region="CENTER",
                is_obstacle=True,
            )
        ]

        enriched = attach_depth_to_detections(detections, depth, bbox_scale=0.2)

        self.assertAlmostEqual(enriched[0].median_depth_m, 1.0)
        self.assertEqual(enriched[0].depth_pixel_count, 4)

    def test_measure_detection_depth_reports_sample_bbox(self):
        depth = np.full((10, 10), 2.0, dtype=np.float32)
        detection = Detection(
            class_name="person",
            confidence=0.9,
            bbox=(0, 0, 10, 10),
            region="CENTER",
            is_obstacle=True,
        )

        measurement = measure_detection_depth(detection, depth, bbox_scale=0.6)

        self.assertEqual(measurement.sample_bbox, (2, 2, 8, 8))
        self.assertAlmostEqual(measurement.median_depth_m, 2.0)
        self.assertEqual(measurement.valid_pixel_count, 36)

    def test_attach_depth_to_detections_rejects_invalid_bbox_scale(self):
        detections = [Detection(class_name="person", confidence=0.9, bbox=(0, 0, 1, 1))]
        depth = np.ones((2, 2), dtype=np.float32)

        with self.assertRaises(ValueError):
            attach_depth_to_detections(detections, depth, bbox_scale=0.0)

    def test_attach_depth_to_detections_leaves_detections_when_depth_missing(self):
        detections = [Detection(class_name="person", confidence=0.9, bbox=(0, 0, 1, 1))]

        self.assertIs(attach_depth_to_detections(detections, None), detections)

    def test_stereo_calibration_round_trips_to_npz(self):
        calibration = _make_test_calibration()

        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "stereo_calibration.npz"
            calibration.save(path)
            loaded = load_stereo_rectification(path)

        self.assertEqual(loaded.image_size, (4, 3))
        self.assertAlmostEqual(loaded.baseline_m, 0.06)
        self.assertAlmostEqual(loaded.focal_px, 100.0)
        self.assertAlmostEqual(loaded.reprojection_error, 0.25)

    def test_uncalibrated_rectification_round_trips_to_npz(self):
        rectification = UncalibratedStereoRectification(
            image_size=(4, 3),
            homography_left=np.eye(3, dtype=np.float64),
            homography_right=np.eye(3, dtype=np.float64),
            fundamental_matrix=np.eye(3, dtype=np.float64),
            inlier_count=120,
            match_count=400,
        )

        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "uncalibrated_rectification.npz"
            rectification.save(path)
            loaded = load_stereo_rectification(path)

        self.assertIsInstance(loaded, UncalibratedStereoRectification)
        self.assertEqual(loaded.image_size, (4, 3))
        self.assertIsNone(loaded.baseline_m)
        self.assertIsNone(loaded.focal_px)
        self.assertEqual(loaded.inlier_count, 120)
        self.assertEqual(loaded.match_count, 400)

    def test_find_image_pairs_matches_left_and_right_files(self):
        with TemporaryDirectory() as tmp_dir:
            directory = Path(tmp_dir)
            (directory / "left_000.jpg").touch()
            (directory / "right_000.jpg").touch()
            (directory / "left_001.jpg").touch()

            pairs = find_image_pairs(directory)

        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0][0].name, "left_000.jpg")
        self.assertEqual(pairs[0][1].name, "right_000.jpg")


def _make_test_calibration() -> StereoCalibration:
    camera_matrix = np.array(
        [
            [100.0, 0.0, 2.0],
            [0.0, 100.0, 1.5],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    dist_coeffs = np.zeros((5, 1), dtype=np.float64)
    identity = np.eye(3, dtype=np.float64)
    projection_left = np.array(
        [
            [100.0, 0.0, 2.0, 0.0],
            [0.0, 100.0, 1.5, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    projection_right = projection_left.copy()
    projection_right[0, 3] = -6.0
    return StereoCalibration(
        image_size=(4, 3),
        camera_matrix_left=camera_matrix,
        dist_coeffs_left=dist_coeffs,
        camera_matrix_right=camera_matrix,
        dist_coeffs_right=dist_coeffs,
        rotation=identity,
        translation=np.array([[-0.06], [0.0], [0.0]], dtype=np.float64),
        essential_matrix=identity,
        fundamental_matrix=identity,
        rectification_left=identity,
        rectification_right=identity,
        projection_left=projection_left,
        projection_right=projection_right,
        disparity_to_depth_map=np.eye(4, dtype=np.float64),
        reprojection_error=0.25,
    )


if __name__ == "__main__":
    unittest.main()
