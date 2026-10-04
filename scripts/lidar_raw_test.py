"""Bare-bones console test for an LDROBOT LD19 / D500 series 2D LiDAR.

Reads raw UART packets straight off the serial port and prints angle/distance
readings to the console. No dependency on the rest of the haptos pipeline --
this is just to confirm the sensor is wired up and talking before wiring it
into haptos/sensor/lidar_reader.py.

Usage (on the Raspberry Pi):
    python3 scripts/lidar_raw_test.py --port /dev/ttyUSB0

If you don't know the port, run `ls /dev/ttyUSB* /dev/ttyACM*` first.
"""

import argparse
import struct
import time

import serial

PACKET_HEADER = 0x54
POINTS_PER_PACKET = 12
PACKET_LEN = 47  # header + verlen + speed(2) + start_angle(2) + 12*3 + end_angle(2) + timestamp(2) + crc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print raw LD19/D500 LiDAR readings to the console")
    parser.add_argument("--port", required=True, help="Serial port, e.g. /dev/ttyUSB0 or COM5")
    parser.add_argument("--baudrate", type=int, default=230400, help="LD19/D500 default baudrate")
    parser.add_argument(
        "--print-interval",
        type=float,
        default=0.0,
        help="Minimum seconds between printed lines (default 0 = print every reading). "
        "Serial is still read continuously regardless of this value, so the parser stays in sync.",
    )
    args = parser.parse_args()
    return args


def read_packet(ser: serial.Serial) -> bytes | None:
    """Sync on the header byte and read one fixed-length packet."""
    b = ser.read(1)
    if not b or b[0] != PACKET_HEADER:
        return None

    verlen = ser.read(1)
    if not verlen or verlen[0] != 0x2C:
        return None  # not a real header, keep resyncing byte-by-byte

    rest = ser.read(PACKET_LEN - 2)
    if len(rest) != PACKET_LEN - 2:
        return None

    return b + verlen + rest


def parse_packet(packet: bytes) -> list[tuple[float, float, int]]:
    """Return (angle_deg, distance_mm, intensity) for each point in the packet."""
    speed, start_angle_raw = struct.unpack_from("<HH", packet, 2)
    points_raw = packet[6 : 6 + POINTS_PER_PACKET * 3]
    end_angle_raw = struct.unpack_from("<H", packet, 6 + POINTS_PER_PACKET * 3)[0]

    start_angle = start_angle_raw / 100.0
    end_angle = end_angle_raw / 100.0
    angle_span = (end_angle - start_angle) % 360.0
    step = angle_span / max(POINTS_PER_PACKET - 1, 1)

    readings = []
    for i in range(POINTS_PER_PACKET):
        offset = i * 3
        distance_mm, intensity = struct.unpack_from("<HB", points_raw, offset)
        angle_deg = (start_angle + step * i) % 360.0
        readings.append((angle_deg, float(distance_mm), intensity))
    return readings


def main() -> None:
    args = parse_args()
    ser = serial.Serial(port=args.port, baudrate=args.baudrate, timeout=0.5)
    print(f"Listening on {args.port} @ {args.baudrate} baud. Ctrl+C to stop.")

    last_print = 0.0
    try:
        while True:
            packet = read_packet(ser)
            if packet is None:
                continue
            for angle_deg, distance_mm, intensity in parse_packet(packet):
                now = time.monotonic()
                if now - last_print < args.print_interval:
                    continue
                last_print = now
                print(f"angle={angle_deg:6.2f} deg  distance={distance_mm:7.1f} mm  intensity={intensity}")
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        ser.close()


if __name__ == "__main__":
    main()
