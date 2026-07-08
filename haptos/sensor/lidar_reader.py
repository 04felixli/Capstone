"""LiDAR scan sources for serial-connected hardware."""

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from haptos.config import LIDAR_SOURCE_NONE, LIDAR_SOURCE_SERIAL
from haptos.types import RawLidarScan


@dataclass(frozen=True)
class LidarSample:
    """One parsed 2D LiDAR sample."""

    angle_deg: float
    distance_mm: float
    quality: int = 1


class SerialLidarReader:
    """Read 2D LiDAR samples from a serial port.

    Expected input format is one sample per line:

        angle_deg,distance_mm,quality

    The quality field is optional. A line containing SCAN, START, or END marks a
    scan boundary. This works well with a microcontroller or vendor driver that
    converts a LiDAR's native protocol into simple text samples.
    """

    def __init__(
        self,
        port: str,
        baudrate: int = 115200,
        serial_timeout_s: float = 0.05,
        scan_timeout_s: float = 0.20,
        min_samples: int = 5,
    ):
        if not port:
            raise ValueError("A LiDAR serial port is required, for example COM5.")
        if baudrate <= 0:
            raise ValueError("LiDAR baudrate must be positive.")
        if scan_timeout_s <= 0:
            raise ValueError("LiDAR scan timeout must be positive.")
        if min_samples <= 0:
            raise ValueError("LiDAR minimum sample count must be positive.")

        try:
            import serial
        except ModuleNotFoundError as exc:
            raise RuntimeError("pyserial is not installed. Run: pip install -r requirements.txt") from exc

        self.port = port
        self.baudrate = baudrate
        self.scan_timeout_s = scan_timeout_s
        self.min_samples = min_samples
        self._serial = serial.Serial(
            port=port,
            baudrate=baudrate,
            timeout=serial_timeout_s,
        )

    def read(self) -> RawLidarScan:
        samples = []
        started_at = time.perf_counter()

        while time.perf_counter() - started_at < self.scan_timeout_s:
            raw_line = self._serial.readline()
            if not raw_line:
                continue

            line = raw_line.decode("utf-8", errors="ignore").strip()
            if not line:
                continue

            if _is_scan_boundary(line):
                if len(samples) >= self.min_samples:
                    break
                continue

            sample = parse_lidar_sample_line(line)
            if sample is not None:
                samples.append(sample)

        timestamp_ms = int(time.time() * 1000)
        return _samples_to_scan(timestamp_ms, samples)

    def close(self) -> None:
        self._serial.close()


def parse_lidar_sample_line(line: str) -> Optional[LidarSample]:
    """Parse a serial LiDAR sample line.

    Supported separators are comma, semicolon, or whitespace:

        12.5,840,15
        12.5 840 15
        12.5;840
    """

    cleaned = line.strip()
    if not cleaned or _is_scan_boundary(cleaned):
        return None

    for separator in (",", ";"):
        cleaned = cleaned.replace(separator, " ")

    parts = cleaned.split()
    if len(parts) < 2:
        return None

    try:
        angle_deg = float(parts[0])
        distance_mm = float(parts[1])
        quality = int(float(parts[2])) if len(parts) >= 3 else 1
    except ValueError:
        return None

    return LidarSample(
        angle_deg=angle_deg,
        distance_mm=distance_mm,
        quality=quality,
    )


def create_lidar_reader(
    source: str,
    port: Optional[str] = None,
    baudrate: int = 115200,
    scan_timeout_s: float = 0.20,
    min_samples: int = 5,
) -> Optional[SerialLidarReader]:
    """Create a LiDAR reader from CLI configuration."""

    if source == LIDAR_SOURCE_NONE:
        return None
    if source == LIDAR_SOURCE_SERIAL:
        return SerialLidarReader(
            port=port or "",
            baudrate=baudrate,
            scan_timeout_s=scan_timeout_s,
            min_samples=min_samples,
        )
    raise ValueError(f"Unsupported LiDAR source: {source}")


def _samples_to_scan(timestamp_ms: int, samples: list[LidarSample]) -> RawLidarScan:
    angles_deg = np.array([sample.angle_deg for sample in samples], dtype=np.float32)
    distances_m = np.array([sample.distance_mm / 1000.0 for sample in samples], dtype=np.float32)
    qualities = np.array([sample.quality for sample in samples], dtype=np.uint8)

    return RawLidarScan(
        timestamp_ms=timestamp_ms,
        angles_rad=np.deg2rad(angles_deg).astype(np.float32),
        distances_m=distances_m,
        qualities=qualities,
    )


def _is_scan_boundary(line: str) -> bool:
    return line.strip().upper() in {"SCAN", "START", "END"}
