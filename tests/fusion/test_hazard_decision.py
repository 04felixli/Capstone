"""Tests for camera/LiDAR hazard decisions."""

import unittest

from haptos.fusion.hazard_decision import generate_fused_navigation_hint
from haptos.types import Detection, LidarFrameSummary


class HazardDecisionTests(unittest.TestCase):
    def test_forward_with_no_obstacles(self):
        command = generate_fused_navigation_hint([])

        self.assertEqual(command, "FORWARD")

    def test_stops_for_center_region_obstacle(self):
        command = generate_fused_navigation_hint([_detection(region="CENTER")])

        self.assertEqual(command, "STOP")

    def test_routes_around_left_only_obstacle(self):
        command = generate_fused_navigation_hint([_detection(region="LEFT")])

        self.assertEqual(command, "GO_RIGHT")

    def test_stops_for_near_lidar_return(self):
        lidar = LidarFrameSummary(
            timestamp_ms=100,
            fault_state="none",
            point_count=20,
            nearest_distance_m=0.5,
        )

        command = generate_fused_navigation_hint([], lidar)

        self.assertEqual(command, "STOP")

    def test_ignores_lidar_return_beyond_emergency_distance(self):
        lidar = LidarFrameSummary(
            timestamp_ms=100,
            fault_state="none",
            point_count=20,
            nearest_distance_m=1.5,
        )

        command = generate_fused_navigation_hint([], lidar, emergency_stop_distance_m=0.8)

        self.assertEqual(command, "FORWARD")

    def test_ignores_lidar_return_with_fault_state(self):
        lidar = LidarFrameSummary(
            timestamp_ms=100,
            fault_state="no_valid_points",
            point_count=0,
        )

        command = generate_fused_navigation_hint([], lidar)

        self.assertEqual(command, "FORWARD")

    def test_rejects_non_positive_emergency_stop_distance(self):
        with self.assertRaises(ValueError):
            generate_fused_navigation_hint([], emergency_stop_distance_m=0.0)


def _detection(region="CENTER"):
    return Detection(
        class_name="person",
        confidence=0.9,
        bbox=(10, 10, 30, 40),
        region=region,
        is_obstacle=True,
    )


if __name__ == "__main__":
    unittest.main()
