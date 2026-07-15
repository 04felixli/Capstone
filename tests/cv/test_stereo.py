"""Tests for stereo disparity/depth helpers."""

import unittest

import numpy as np

from haptos.cv.stereo import summarize_disparity


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


if __name__ == "__main__":
    unittest.main()
