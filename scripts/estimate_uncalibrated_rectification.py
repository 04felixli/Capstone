"""Estimate stereo rectification from a normal textured image pair."""

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from haptos.cv.stereo_calibration import (  # noqa: E402
    estimate_uncalibrated_rectification_from_images,
    make_rectified_preview,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate uncalibrated stereo rectification from one image pair")
    parser.add_argument("--left", required=True, help="Left image path")
    parser.add_argument("--right", required=True, help="Right image path")
    parser.add_argument("--output", default="calibration/uncalibrated_rectification.npz", help="Output .npz file")
    parser.add_argument(
        "--preview-output",
        default="calibration/uncalibrated_rectification_preview.jpg",
        help="Output side-by-side rectified preview image",
    )
    parser.add_argument("--max-features", type=int, default=4000, help="Maximum ORB features to detect")
    parser.add_argument("--keep-matches", type=int, default=800, help="Best feature matches to keep before RANSAC")
    parser.add_argument("--min-inliers", type=int, default=80, help="Minimum RANSAC inlier matches required")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rectification = estimate_uncalibrated_rectification_from_images(
        left_path=args.left,
        right_path=args.right,
        max_features=args.max_features,
        keep_matches=args.keep_matches,
        min_inliers=args.min_inliers,
    )
    rectification.save(args.output)

    preview = make_rectified_preview(rectification, args.left, args.right)
    preview_path = Path(args.preview_output)
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(preview_path), preview)

    print(f"Saved uncalibrated rectification to {args.output}")
    print(f"Saved rectified preview to {args.preview_output}")
    print(f"Feature matches: {rectification.match_count}")
    print(f"RANSAC inliers: {rectification.inlier_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
