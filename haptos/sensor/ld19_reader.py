"""Reader for LDROBOT LD19 / STL-19P / D500 binary 2D LiDAR packets."""

import time

import numpy as np

from haptos.types import RawLidarScan

LD19_DEFAULT_BAUDRATE = 230400
LD19_PACKET_HEADER = 0x54
LD19_PACKET_VERLEN = 0x2C
LD19_POINTS_PER_PACKET = 12
LD19_PACKET_LEN = 47


class Ld19Reader:
    """Read full 360-degree scans from an LD19-protocol LiDAR on a serial port.

    The sensor spins continuously and emits 12-point packets with no scan markers.
    A scan is the run of points between two successive wrap-arounds of the angle
    (for example 359.x degrees back to 0.x degrees).
    """

    def __init__(self, port: str, baudrate: int = LD19_DEFAULT_BAUDRATE, serial_timeout_s: float = 0.5):
        if not port:
            raise ValueError("A LiDAR serial port is required, for example /dev/lidar or COM5.")
        if baudrate <= 0:
            raise ValueError("LiDAR baudrate must be positive.")

        try:
            import serial
        except ModuleNotFoundError as exc:
            raise RuntimeError("pyserial is not installed. Run: pip install -r requirements.txt") from exc

        self.port = port
        self.baudrate = baudrate
        self._serial = serial.Serial(port=port, baudrate=baudrate, timeout=serial_timeout_s)
        self._scan_points: list[tuple[float, float, int]] = []
        self._last_angle_deg: float | None = None

    def read(self) -> RawLidarScan:
        while True:
            packet = self._read_packet()
            if packet is None:
                continue

            points = parse_ld19_packet(packet)
            if self._last_angle_deg is not None and points[0][0] < self._last_angle_deg - 180.0:
                scan = _points_to_scan(int(time.time() * 1000), self._scan_points)
                self._scan_points = points
                self._last_angle_deg = points[-1][0]
                return scan

            self._scan_points.extend(points)
            self._last_angle_deg = points[-1][0]

    def close(self) -> None:
        self._serial.close()

    def _read_packet(self) -> bytes | None:
        """Sync on the header byte and read one fixed-length packet."""
        header = self._serial.read(1)
        if not header or header[0] != LD19_PACKET_HEADER:
            return None

        verlen = self._serial.read(1)
        if not verlen or verlen[0] != LD19_PACKET_VERLEN:
            return None

        rest = self._serial.read(LD19_PACKET_LEN - 2)
        if len(rest) != LD19_PACKET_LEN - 2:
            return None

        return header + verlen + rest


def parse_ld19_packet(packet: bytes) -> list[tuple[float, float, int]]:
    """Return (angle_deg, distance_mm, intensity) for each point in one packet."""
    start_angle_raw = int.from_bytes(packet[4:6], "little")
    end_angle_raw = int.from_bytes(packet[6 + LD19_POINTS_PER_PACKET * 3 : 8 + LD19_POINTS_PER_PACKET * 3], "little")

    start_angle = start_angle_raw / 100.0
    end_angle = end_angle_raw / 100.0
    step = ((end_angle - start_angle) % 360.0) / max(LD19_POINTS_PER_PACKET - 1, 1)

    points = []
    for i in range(LD19_POINTS_PER_PACKET):
        offset = 6 + i * 3
        distance_mm = int.from_bytes(packet[offset : offset + 2], "little")
        intensity = packet[offset + 2]
        angle_deg = (start_angle + step * i) % 360.0
        points.append((angle_deg, float(distance_mm), intensity))
    return points


def _points_to_scan(timestamp_ms: int, points: list[tuple[float, float, int]]) -> RawLidarScan:
    angles_deg = np.array([p[0] for p in points], dtype=np.float32)
    distances_m = np.array([p[1] / 1000.0 for p in points], dtype=np.float32)
    qualities = np.array([p[2] for p in points], dtype=np.uint8)

    return RawLidarScan(
        timestamp_ms=timestamp_ms,
        angles_rad=np.deg2rad(angles_deg).astype(np.float32),
        distances_m=distances_m,
        qualities=qualities,
    )
