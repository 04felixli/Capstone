"""Tests for camera/LiDAR hazard decisions."""

import unittest

from haptos.fusion.hazard_decision import generate_fused_navigation_hint
from haptos.types import Detection, LidarFrameSummary


class HazardDecisionTests(unittest.TestCase):
    def test_ignores_trusted_distant_obstacle(self):
        command = generate_fused_navigation_hint([_detection(4.0)])

        self.assertEqual(command, "FORWARD")

    def test_stops_for_trusted_near_obstacle_in_any_region(self):
        command = generate_fused_navigation_hint([_detection(0.6, region="LEFT")])

        self.assertEqual(command, "STOP")

    def test_keeps_uncertain_distant_center_detection_conservative(self):
        detection = _detection(4.0)
        detection = Detection(
            **{
                **detection.to_dict(),
                "depth_fault_state": "high_uncertainty",
            }
        )

        command = generate_fused_navigation_hint([detection])

        self.assertEqual(command, "STOP")

    def test_stops_for_near_lidar_return(self):
        lidar = LidarFrameSummary(
            timestamp_ms=100,
            fault_state="none",
            point_count=20,
            nearest_distance_m=0.5,
        )

        command = generate_fused_navigation_hint([], lidar)

        self.assertEqual(command, "STOP")


def _detection(depth_m, region="CENTER"):
    return Detection(
        class_name="person",
        confidence=0.9,
        bbox=(10, 10, 30, 40),
        region=region,
        is_obstacle=True,
        median_depth_m=depth_m,
        depth_pixel_count=100,
        depth_uncertainty_m=0.05,
        depth_valid_fraction=0.8,
        depth_fault_state="none",
    )


if __name__ == "__main__":
    unittest.main()
