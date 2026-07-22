"""Capture left/right image pairs for stereo checkerboard calibration."""

import argparse
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from haptos.cv.camera import VideoSource


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture stereo calibration image pairs")
    parser.add_argument("--left-source", default="picamera0", help="Left camera source")
    parser.add_argument("--right-source", default="picamera1", help="Right camera source")
    parser.add_argument("--output-dir", default="calibration/images", help="Directory for captured pairs")
    parser.add_argument("--pairs", type=int, default=25, help="Number of pairs to capture")
    parser.add_argument("--interval", type=float, default=1.5, help="Seconds between captures")
    parser.add_argument("--warmup", type=float, default=2.0, help="Seconds to let cameras settle before capture")
    parser.add_argument("--pattern-cols", type=int, help="Checkerboard inner corner columns for validation")
    parser.add_argument("--pattern-rows", type=int, help="Checkerboard inner corner rows for validation")
    parser.add_argument(
        "--manual",
        action="store_true",
        help="Wait for Enter before each capture so saved previews can be inspected between pairs",
    )
    args = parser.parse_args()
    if args.pairs <= 0:
        parser.error("--pairs must be positive")
    if args.interval < 0:
        parser.error("--interval must be 0 or greater")
    if args.warmup < 0:
        parser.error("--warmup must be 0 or greater")
    if (args.pattern_cols is None) != (args.pattern_rows is None):
        parser.error("--pattern-cols and --pattern-rows must be provided together")
    if args.pattern_cols is not None and (args.pattern_cols <= 2 or args.pattern_rows <= 2):
        parser.error("checkerboard pattern dimensions must both be greater than 2")
    return args


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    left_source = VideoSource(args.left_source)
    right_source = VideoSource(args.right_source)
    try:
        if args.warmup:
            print(f"Warming up cameras for {args.warmup:.1f}s...")
            time.sleep(args.warmup)

        for index in range(args.pairs):
            if args.manual:
                input(f"Pair {index + 1}/{args.pairs}: position checkerboard, then press Enter to capture...")
            elif args.interval:
                print(f"Pair {index + 1}/{args.pairs}: position checkerboard, capturing in {args.interval:.1f}s...")
                time.sleep(args.interval)

            left_ok, left_frame = left_source.read()
            right_ok, right_frame = right_source.read()
            if not left_ok or left_frame is None or not right_ok or right_frame is None:
                raise RuntimeError("Could not read both camera frames")

            left_path = output_dir / f"left_{index:03d}.jpg"
            right_path = output_dir / f"right_{index:03d}.jpg"
            preview_path = output_dir / f"preview_{index:03d}.jpg"
            validation = validate_checkerboard_pair(left_frame, right_frame, args.pattern_cols, args.pattern_rows)
            cv2.imwrite(str(left_path), left_frame)
            cv2.imwrite(str(right_path), right_frame)
            cv2.imwrite(str(preview_path), make_side_by_side_preview(left_frame, right_frame, validation))
            print(f"Saved {left_path}, {right_path}, and {preview_path}")
            if validation is not None:
                print(format_validation_result(validation))

        return 0
    finally:
        left_source.release()
        right_source.release()


def make_side_by_side_preview(left_frame, right_frame, validation=None):
    """Create a side-by-side image for quick calibration-pair inspection."""

    if left_frame.shape[:2] != right_frame.shape[:2]:
        right_frame = cv2.resize(right_frame, (left_frame.shape[1], left_frame.shape[0]))

    left_preview = left_frame.copy()
    right_preview = right_frame.copy()
    cv2.putText(left_preview, "LEFT", (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    cv2.putText(right_preview, "RIGHT", (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    preview = cv2.hconcat([left_preview, right_preview])

    if validation is not None:
        label = "VALID" if validation["valid"] else "INVALID"
        color = (0, 220, 0) if validation["valid"] else (0, 0, 255)
        cv2.putText(preview, label, (12, preview.shape[0] - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    return preview


def validate_checkerboard_pair(left_frame, right_frame, pattern_cols, pattern_rows):
    """Return checkerboard detection status for a stereo pair, or None if disabled."""

    if pattern_cols is None or pattern_rows is None:
        return None

    pattern_size = (pattern_cols, pattern_rows)
    left_found = detect_checkerboard(left_frame, pattern_size)
    right_found = detect_checkerboard(right_frame, pattern_size)
    return {
        "left_found": left_found,
        "right_found": right_found,
        "valid": left_found and right_found,
    }


def detect_checkerboard(frame, pattern_size):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    found, _ = cv2.findChessboardCorners(gray, pattern_size)
    return bool(found)


def format_validation_result(validation):
    if validation["valid"]:
        return "Checkerboard: VALID in both cameras"
    missing = []
    if not validation["left_found"]:
        missing.append("left")
    if not validation["right_found"]:
        missing.append("right")
    camera_label = "camera" if len(missing) == 1 else "cameras"
    return f"Checkerboard: INVALID ({', '.join(missing)} {camera_label} did not detect the full pattern)"


if __name__ == "__main__":
    raise SystemExit(main())
