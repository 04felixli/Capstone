"""Tests for live center depth debug formatting."""

import unittest

from haptos.cv.stereo import DetectionDepthMeasurement
from scripts.live_center_depth import format_center_depth


class LiveCenterDepthFormattingTests(unittest.TestCase):
    def test_format_center_depth_includes_distance_bbox_pixels_and_latency(self):
        measurement = DetectionDepthMeasurement(
            sample_bbox=(280, 200, 360, 280),
            median_depth_m=1.23,
            valid_pixel_count=4000,
        )

        self.assertEqual(
            format_center_depth(7, measurement, 42.89),
            "Frame 7 | center_depth=1.23m | sample_bbox=[280, 200, 360, 280] | valid_pixels=4000 | depth_latency=42.9ms",
        )

    def test_format_center_depth_handles_missing_measurement(self):
        self.assertEqual(
            format_center_depth(8, None, 50.0),
            "Frame 8 | center_depth=n/a | sample_bbox=n/a | valid_pixels=0 | depth_latency=50.0ms",
        )


if __name__ == "__main__":
    unittest.main()
