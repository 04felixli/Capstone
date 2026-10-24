"""Tests for LiDAR parsing, filtering, and buffering."""

import unittest

import numpy as np

import struct

from haptos.config import LIDAR_FAULT_NO_VALID_POINTS, LIDAR_FAULT_NONE, LIDAR_SOURCE_NONE
from haptos.sensor.lidar_buffer import LidarFrameBuffer
from haptos.sensor.lidar_process import process_lidar_scan, polar_scan_to_xyz
from haptos.sensor.ld19_reader import _ScanAccumulator, create_lidar_reader, parse_ld19_packet
from haptos.types import RawLidarScan


def _build_ld19_packet(start_angle_deg: float, end_angle_deg: float, distances_mm, intensity=200) -> bytes:
    """Build a 47-byte LD19 packet (header/verlen/speed + 12 points + end angle/timestamp/last byte)."""
    body = bytes([0x54, 0x2C]) + struct.pack("<H", 1000)
    body += struct.pack("<H", int(round(start_angle_deg * 100)))
    for distance_mm in distances_mm:
        body += struct.pack("<HB", int(distance_mm), intensity)
    body += struct.pack("<H", int(round(end_angle_deg * 100)))
    body += struct.pack("<H", 0)
    return body + bytes([0])  # last byte is unused (no CRC check)


class LidarProcessTests(unittest.TestCase):
    def test_process_keeps_only_useful_range(self):
        scan = RawLidarScan(
            timestamp_ms=100,
            angles_rad=np.zeros(7, dtype=np.float32),
            distances_m=np.array([0.1, 0.3, 1.0, 2.0, 2.1, np.nan, np.inf], dtype=np.float32),
        )

        processed = process_lidar_scan(scan)

        self.assertEqual(processed.fault_state, LIDAR_FAULT_NONE)
        np.testing.assert_allclose(processed.distances_m, np.array([0.3, 1.0, 2.0], dtype=np.float32))
        self.assertEqual(processed.points_xyz.shape, (3, 3))

    def test_process_returns_fault_when_no_points_are_in_range(self):
        scan = RawLidarScan(
            timestamp_ms=100,
            angles_rad=np.zeros(3, dtype=np.float32),
            distances_m=np.array([0.1, 2.5, np.nan], dtype=np.float32),
        )

        processed = process_lidar_scan(scan)

        self.assertEqual(processed.fault_state, LIDAR_FAULT_NO_VALID_POINTS)
        self.assertEqual(processed.point_count(), 0)

    def test_polar_scan_to_xyz_uses_forward_z_axis(self):
        points = polar_scan_to_xyz(
            angles_rad=np.array([0.0, np.pi / 2.0], dtype=np.float32),
            distances_m=np.array([1.0, 2.0], dtype=np.float32),
        )

        np.testing.assert_allclose(points[0], np.array([0.0, 0.0, 1.0]), atol=1e-6)
        np.testing.assert_allclose(points[1], np.array([2.0, 0.0, 0.0]), atol=1e-6)

    def test_summary_relevant_nearest_distance_ignores_side_point(self):
        # A near point straight to the side (90 degrees) must not count as the
        # relevant-nearest distance, only the farther point ahead (0 rad) should.
        scan = RawLidarScan(
            timestamp_ms=100,
            angles_rad=np.array([0.0, np.pi / 2.0], dtype=np.float32),
            distances_m=np.array([1.2, 0.35], dtype=np.float32),
        )

        summary = process_lidar_scan(scan).to_summary()

        self.assertAlmostEqual(summary.nearest_distance_m, 0.35, places=5)
        self.assertAlmostEqual(summary.relevant_nearest_distance_m, 1.2, places=5)

    def test_summary_relevant_nearest_distance_is_none_when_no_points_are_relevant(self):
        scan = RawLidarScan(
            timestamp_ms=100,
            angles_rad=np.array([np.pi / 2.0], dtype=np.float32),
            distances_m=np.array([0.35], dtype=np.float32),
        )

        summary = process_lidar_scan(scan).to_summary()

        self.assertIsNone(summary.relevant_nearest_distance_m)


class Ld19PacketParserTests(unittest.TestCase):
    def test_parse_packet_interpolates_angle_per_point(self):
        packet = _build_ld19_packet(30.0, 31.1, [800 + i for i in range(12)])

        points = parse_ld19_packet(packet)

        self.assertEqual(len(points), 12)
        self.assertAlmostEqual(points[0][0], 30.0, places=2)
        self.assertAlmostEqual(points[-1][0], 31.1, places=2)
        self.assertEqual(points[0][1], 800.0)
        self.assertEqual(points[0][2], 200)

    def test_parse_packet_handles_angle_wrap_within_one_packet(self):
        packet = _build_ld19_packet(359.5, 0.6, [500] * 12)

        points = parse_ld19_packet(packet)

        self.assertAlmostEqual(points[0][0], 359.5, places=2)
        self.assertAlmostEqual(points[-1][0], 0.6, places=2)


class ScanAccumulatorTests(unittest.TestCase):
    def test_feed_accumulates_without_emitting_mid_rotation(self):
        accumulator = _ScanAccumulator()

        self.assertIsNone(accumulator.feed(parse_ld19_packet(_build_ld19_packet(0.0, 8.0, [100] * 12))))
        self.assertIsNone(accumulator.feed(parse_ld19_packet(_build_ld19_packet(8.0, 16.0, [100] * 12))))

    def test_feed_emits_a_scan_on_angle_wrap_around(self):
        accumulator = _ScanAccumulator()
        accumulator.feed(parse_ld19_packet(_build_ld19_packet(340.0, 348.0, [100] * 12)))
        accumulator.feed(parse_ld19_packet(_build_ld19_packet(348.0, 356.0, [100] * 12)))

        scan = accumulator.feed(parse_ld19_packet(_build_ld19_packet(4.0, 12.0, [200] * 12)))

        self.assertIsNotNone(scan)
        self.assertEqual(scan.distances_m.size, 24)
        np.testing.assert_allclose(scan.distances_m, 0.1, atol=1e-6)

    def test_zero_distance_points_are_dropped_by_filtering(self):
        # 11 invalid (0 mm) returns and one real 1.0 m return, positioned at the end of a
        # rotation so the next packet's low start angle triggers the wrap and emits this scan.
        accumulator = _ScanAccumulator()
        accumulator.feed(parse_ld19_packet(_build_ld19_packet(350.0, 358.0, [0] * 11 + [1000])))
        scan = accumulator.feed(parse_ld19_packet(_build_ld19_packet(4.0, 12.0, [100] * 12)))

        processed = process_lidar_scan(scan)

        self.assertEqual(processed.point_count(), 1)
        np.testing.assert_allclose(processed.distances_m, [1.0])


class CreateLidarReaderTests(unittest.TestCase):
    def test_none_source_returns_no_reader(self):
        # Must not touch pyserial/a real port: this source skips Ld19Reader entirely.
        self.assertIsNone(create_lidar_reader(LIDAR_SOURCE_NONE))

    def test_unsupported_source_raises_before_opening_a_port(self):
        with self.assertRaises(ValueError):
            create_lidar_reader("bogus-source")


class LidarFrameBufferTests(unittest.TestCase):
    def test_closest_to_returns_nearest_timestamp(self):
        buffer = LidarFrameBuffer(max_frames=3)

        first = process_lidar_scan(
            RawLidarScan(
                timestamp_ms=100,
                angles_rad=np.array([0.0], dtype=np.float32),
                distances_m=np.array([1.0], dtype=np.float32),
            )
        )
        second = process_lidar_scan(
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
