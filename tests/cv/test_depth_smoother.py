"""Tests for temporal object-depth smoothing."""

import unittest

from haptos.cv.depth_smoother import DetectionDepthSmoother
from haptos.types import Detection


class DetectionDepthSmootherTests(unittest.TestCase):
    def test_smooths_depth_for_matching_objects(self):
        smoother = DetectionDepthSmoother(window_size=3)

        first = smoother.update([_detection(1.0, (10, 10, 30, 40))])
        second = smoother.update([_detection(1.4, (11, 10, 31, 40))])
        third = smoother.update([_detection(1.1, (12, 10, 32, 40))])

        self.assertAlmostEqual(first[0].median_depth_m, 1.0)
        self.assertAlmostEqual(second[0].median_depth_m, 1.2)
        self.assertAlmostEqual(third[0].median_depth_m, 1.1)
        self.assertGreater(third[0].depth_uncertainty_m, 0.0)

    def test_does_not_mix_non_overlapping_objects(self):
        smoother = DetectionDepthSmoother(window_size=3)

        smoother.update([_detection(1.0, (0, 0, 10, 10))])
        result = smoother.update([_detection(4.0, (100, 100, 110, 110))])

        self.assertAlmostEqual(result[0].median_depth_m, 4.0)


def _detection(depth_m, bbox):
    return Detection(
        class_name="person",
        confidence=0.9,
        bbox=bbox,
        region="CENTER",
        is_obstacle=True,
        median_depth_m=depth_m,
        depth_pixel_count=100,
        depth_uncertainty_m=0.05,
        depth_valid_fraction=0.8,
        depth_fault_state="none",
    )


if __name__ == "__main__":
    unittest.main()
