"""Create a stereo calibration file from checkerboard image pairs."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from haptos.cv.stereo_calibration import calibrate_stereo_from_images, find_image_pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate stereo cameras from checkerboard pairs")
    parser.add_argument("--image-dir", default="calibration/images", help="Directory with left_###.jpg/right_###.jpg pairs")
    parser.add_argument("--output", default="calibration/stereo_calibration.npz", help="Output .npz calibration file")
    parser.add_argument("--pattern-cols", type=int, required=True, help="Checkerboard inner corner columns")
    parser.add_argument("--pattern-rows", type=int, required=True, help="Checkerboard inner corner rows")
    parser.add_argument("--square-size-m", type=float, required=True, help="Checkerboard square size in meters")
    args = parser.parse_args()
    if args.pattern_cols <= 0 or args.pattern_rows <= 0:
        parser.error("pattern dimensions must be positive")
    if args.square_size_m <= 0:
        parser.error("--square-size-m must be positive")
    return args


def main() -> int:
    args = parse_args()
    image_pairs = find_image_pairs(args.image_dir)
    if not image_pairs:
        raise RuntimeError(f"No left/right image pairs found in {args.image_dir}")

    calibration, valid_pair_count = calibrate_stereo_from_images(
        image_pairs=image_pairs,
        pattern_size=(args.pattern_cols, args.pattern_rows),
        square_size_m=args.square_size_m,
    )
    calibration.save(args.output)

    print(f"Used {valid_pair_count}/{len(image_pairs)} valid checkerboard pairs")
    print(f"Saved calibration to {args.output}")
    print(f"Reprojection error: {calibration.reprojection_error:.4f}")
    print(f"Calibrated baseline: {calibration.baseline_m:.4f} m")
    print(f"Calibrated focal length: {calibration.focal_px:.2f} px")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
