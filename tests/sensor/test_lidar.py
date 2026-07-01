"""Tests for LiDAR parsing, filtering, and buffering."""

import unittest

import numpy as np

from haptos.config import LIDAR_FAULT_NO_VALID_POINTS, LIDAR_FAULT_NONE
from haptos.sensor.lidar_buffer import LidarFrameBuffer
from haptos.sensor.lidar_filter import filter_lidar_scan, polar_scan_to_xyz
from haptos.sensor.lidar_reader import parse_lidar_sample_line
from haptos.types import RawLidarScan


class LidarFilterTests(unittest.TestCase):
    def test_filter_keeps_only_useful_range(self):
        scan = RawLidarScan(
            timestamp_ms=100,
            angles_rad=np.zeros(7, dtype=np.float32),
            distances_m=np.array([0.1, 0.3, 1.0, 2.0, 2.1, np.nan, np.inf], dtype=np.float32),
        )

        filtered = filter_lidar_scan(scan)

        self.assertEqual(filtered.fault_state, LIDAR_FAULT_NONE)
        np.testing.assert_allclose(filtered.distances_m, np.array([0.3, 1.0, 2.0], dtype=np.float32))
        self.assertEqual(filtered.points_xyz.shape, (3, 3))

    def test_filter_returns_fault_when_no_points_are_in_range(self):
        scan = RawLidarScan(
            timestamp_ms=100,
            angles_rad=np.zeros(3, dtype=np.float32),
            distances_m=np.array([0.1, 2.5, np.nan], dtype=np.float32),
        )

        filtered = filter_lidar_scan(scan)

        self.assertEqual(filtered.fault_state, LIDAR_FAULT_NO_VALID_POINTS)
        self.assertEqual(filtered.point_count(), 0)

    def test_polar_scan_to_xyz_uses_forward_z_axis(self):
        points = polar_scan_to_xyz(
            angles_rad=np.array([0.0, np.pi / 2.0], dtype=np.float32),
            distances_m=np.array([1.0, 2.0], dtype=np.float32),
        )

        np.testing.assert_allclose(points[0], np.array([0.0, 0.0, 1.0]), atol=1e-6)
        np.testing.assert_allclose(points[1], np.array([2.0, 0.0, 0.0]), atol=1e-6)


class LidarParserTests(unittest.TestCase):
    def test_parser_accepts_comma_separated_samples(self):
        sample = parse_lidar_sample_line("12.5,840,15")

        self.assertIsNotNone(sample)
        self.assertEqual(sample.angle_deg, 12.5)
        self.assertEqual(sample.distance_mm, 840.0)
        self.assertEqual(sample.quality, 15)

    def test_parser_accepts_whitespace_separated_samples(self):
        sample = parse_lidar_sample_line("-8.0 1250")

        self.assertIsNotNone(sample)
        self.assertEqual(sample.angle_deg, -8.0)
        self.assertEqual(sample.distance_mm, 1250.0)
        self.assertEqual(sample.quality, 1)

    def test_parser_ignores_scan_boundary_and_bad_lines(self):
        self.assertIsNone(parse_lidar_sample_line("SCAN"))
        self.assertIsNone(parse_lidar_sample_line("not,a,sample"))


class LidarFrameBufferTests(unittest.TestCase):
    def test_closest_to_returns_nearest_timestamp(self):
        buffer = LidarFrameBuffer(max_frames=3)

        first = filter_lidar_scan(
            RawLidarScan(
                timestamp_ms=100,
                angles_rad=np.array([0.0], dtype=np.float32),
                distances_m=np.array([1.0], dtype=np.float32),
            )
        )
        second = filter_lidar_scan(
            RawLidarScan(
                timestamp_ms=200,
                angles_rad=np.array([0.0], dtype=np.float32),
                distances_m=np.array([1.5], dtype=np.float32),
            )
        )

        buffer.add(first)
        buffer.add(second)

        self.assertIs(buffer.closest_to(180), second)


if __name__ == "__main__":
    unittest.main()
