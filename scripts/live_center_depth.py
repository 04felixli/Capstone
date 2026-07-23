"""Live center-screen stereo depth debug command."""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from haptos.cv.camera import VideoSource  # noqa: E402
from haptos.cv.stereo import StereoDepthEstimator, measure_center_depth  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print live stereo depth at the center of the screen")
    parser.add_argument("--source", default="picamera0", help="Left/source camera")
    parser.add_argument("--stereo-right-source", default="picamera1", help="Right camera")
    parser.add_argument("--stereo-baseline-m", type=float, help="Rough stereo baseline in meters")
    parser.add_argument("--stereo-focal-px", type=float, help="Rough focal length in pixels")
    parser.add_argument("--stereo-calibration", help="Optional stereo calibration/rectification .npz file")
    parser.add_argument("--stereo-num-disparities", type=int, default=64)
    parser.add_argument("--stereo-block-size", type=int, default=5)
    parser.add_argument("--center-box-size", type=int, default=80, help="Center square size in pixels")
    parser.add_argument("--fps", type=float, default=1.0, help="Maximum reporting FPS. Use 0 to run as fast as possible.")
    parser.add_argument("--warmup", type=float, default=1.0, help="Seconds to let cameras settle before readings")
    args = parser.parse_args()
    if args.center_box_size <= 0:
        parser.error("--center-box-size must be positive")
    if args.fps < 0:
        parser.error("--fps must be 0 or greater")
    if args.warmup < 0:
        parser.error("--warmup must be 0 or greater")
    return args


def main() -> int:
    args = parse_args()
    left_source = VideoSource(args.source)
    right_source = VideoSource(args.stereo_right_source)
    try:
        if args.warmup:
            time.sleep(args.warmup)

        depth_estimator = StereoDepthEstimator(
            num_disparities=args.stereo_num_disparities,
            block_size=args.stereo_block_size,
            baseline_m=args.stereo_baseline_m,
            focal_px=args.stereo_focal_px,
            calibration_path=args.stereo_calibration,
        )
        frame_interval_s = 1.0 / args.fps if args.fps > 0 else 0.0
        frame_index = 0

        while True:
            loop_started_at = time.monotonic()
            left_ok, left_frame = left_source.read()
            right_ok, right_frame = right_source.read()
            if not left_ok or left_frame is None:
                raise RuntimeError(f"No frame received from source '{args.source}'")
            if not right_ok or right_frame is None:
                raise RuntimeError(f"No frame received from stereo right source '{args.stereo_right_source}'")

            depth_started_at = time.perf_counter()
            stereo_frame = depth_estimator.estimate_frame(left_frame, right_frame)
            depth_latency_ms = (time.perf_counter() - depth_started_at) * 1000.0

            frame_index += 1
            measurement = None
            if stereo_frame.depth_m is not None:
                measurement = measure_center_depth(stereo_frame.depth_m, box_size_px=args.center_box_size)

            print(format_center_depth(frame_index, measurement, depth_latency_ms), flush=True)

            if frame_interval_s > 0:
                elapsed_s = time.monotonic() - loop_started_at
                sleep_s = frame_interval_s - elapsed_s
                if sleep_s > 0:
                    time.sleep(sleep_s)

    except KeyboardInterrupt:
        print("Interrupted by user.", file=sys.stderr)
        return 130
    finally:
        left_source.release()
        right_source.release()


def format_center_depth(frame_index: int, measurement, depth_latency_ms: float) -> str:
    if measurement is None:
        center_depth = "n/a"
        sample_bbox = "n/a"
        valid_pixels = 0
    else:
        center_depth = "n/a" if measurement.median_depth_m is None else f"{measurement.median_depth_m:.2f}m"
        sample_bbox = str(list(measurement.sample_bbox))
        valid_pixels = measurement.valid_pixel_count

    return (
        f"Frame {frame_index} | "
        f"center_depth={center_depth} | "
        f"sample_bbox={sample_bbox} | "
        f"valid_pixels={valid_pixels} | "
        f"depth_latency={depth_latency_ms:.1f}ms"
    )


if __name__ == "__main__":
    raise SystemExit(main())
