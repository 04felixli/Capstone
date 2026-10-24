"""Reader for LDROBOT LD19 / STL-19P / D500 binary 2D LiDAR packets."""

import time
from typing import Optional

import numpy as np

from haptos.config import (
    LIDAR_DEFAULT_BAUDRATE,
    LIDAR_READ_TIMEOUT_S,
    LIDAR_SOURCE_NONE,
    LIDAR_SOURCE_SERIAL,
)
from haptos.types import RawLidarScan

LD19_PACKET_HEADER = 0x54
LD19_PACKET_VERLEN = 0x2C
LD19_POINTS_PER_PACKET = 12
LD19_PACKET_LEN = 47


class _ScanAccumulator:
    """Collect points across packets and split them into full 360-degree scans.

    The sensor spins continuously and emits 12-point packets with no scan markers.
    A scan is the run of points between two successive wrap-arounds of the angle
    (for example 359.x degrees back to 0.x degrees).
    """

    # Initialize with no points buffered and no last-seen angle.
    def __init__(self):
        self._scan_points: list[tuple[float, float, int]] = []
        self._last_angle_deg: Optional[float] = None

    # Buffer a packet's points, emitting the prior scan once a wrap-around is detected.
    def feed(self, points: list[tuple[float, float, int]]) -> Optional[RawLidarScan]:
        """Add one packet's points; return the completed scan if this packet wrapped past 360, else None."""
        if self._last_angle_deg is not None and points[0][0] < self._last_angle_deg - 180.0:
            scan = _points_to_scan(int(time.time() * 1000), self._scan_points)
            self._scan_points = points
            self._last_angle_deg = points[-1][0]
            return scan

        self._scan_points.extend(points)
        self._last_angle_deg = points[-1][0]
        return None
    
# Pack a list of (angle_deg, distance_mm, intensity) points into a RawLidarScan.
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


class Ld19Reader:
    """Read full 360-degree scans from an LD19-protocol LiDAR on a serial port."""

    # Open the serial port and set up the packet accumulator.
    def __init__(
        self,
        port: str,
        baudrate: int = LIDAR_DEFAULT_BAUDRATE,
        serial_timeout_s: float = 0.5,
        read_timeout_s: float = LIDAR_READ_TIMEOUT_S,
    ):
        if not port:
            raise ValueError("A LiDAR serial port is required, for example /dev/lidar or COM5.")
        if baudrate <= 0:
            raise ValueError("LiDAR baudrate must be positive.")
        if read_timeout_s <= 0:
            raise ValueError("LiDAR read_timeout_s must be positive.")

        try:
            import serial
        except ModuleNotFoundError as exc:
            raise RuntimeError("pyserial is not installed. Run: pip install -r requirements.txt") from exc

        self.port = port
        self.baudrate = baudrate
        self._serial = serial.Serial(port=port, baudrate=baudrate, timeout=serial_timeout_s)
        self._accumulator = _ScanAccumulator()
        self._read_timeout_s = read_timeout_s

    # Keep reading packets until a full 360-degree scan has accumulated, or give
    # up once read_timeout_s has passed with no complete scan (e.g. sensor silence).
    def read(self) -> RawLidarScan:
        started_at = time.monotonic()
        while True:
            if time.monotonic() - started_at > self._read_timeout_s:
                raise TimeoutError(f"No LiDAR scan received within {self._read_timeout_s}s.")

            packet = self._read_packet()
            if packet is None:
                continue

            points = parse_ld19_packet(packet)
            scan = self._accumulator.feed(points)
            if scan is not None:
                return scan

    # Close the underlying serial connection.
    def close(self) -> None:
        self._serial.close()

    # Sync on the header/verlen bytes and read one fixed-length packet.
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


# Decode one LD19 packet's raw bytes into per-point (angle, distance, intensity) tuples.
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


# Build the configured LiDAR reader (or None) from CLI source/port/baudrate options.
def create_lidar_reader(
    source: str,
    port: Optional[str] = None,
    baudrate: int = LIDAR_DEFAULT_BAUDRATE,
    read_timeout_s: float = LIDAR_READ_TIMEOUT_S,
) -> Optional[Ld19Reader]:
    """Create a LiDAR reader from CLI configuration."""

    if source == LIDAR_SOURCE_NONE:
        return None
    if source == LIDAR_SOURCE_SERIAL:
        return Ld19Reader(port=port or "", baudrate=baudrate, read_timeout_s=read_timeout_s)
    raise ValueError(f"Unsupported LiDAR source: {source}")
