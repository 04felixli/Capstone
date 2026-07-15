"""Tests for stereo disparity/depth helpers."""

import unittest

import numpy as np

from haptos.cv.stereo import attach_depth_to_detections, disparity_to_depth, summarize_disparity
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

        enriched = attach_depth_to_detections(detections, depth)

        self.assertEqual(len(enriched), 1)
        self.assertAlmostEqual(enriched[0].median_depth_m, 2.5)
        self.assertEqual(enriched[0].depth_pixel_count, 4)
        self.assertEqual(enriched[0].class_name, "person")

    def test_attach_depth_to_detections_leaves_detections_when_depth_missing(self):
        detections = [Detection(class_name="person", confidence=0.9, bbox=(0, 0, 1, 1))]

        self.assertIs(attach_depth_to_detections(detections, None), detections)


if __name__ == "__main__":
    unittest.main()
