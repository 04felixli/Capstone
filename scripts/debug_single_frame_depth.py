"""Capture one stereo pair and print per-detection depth debug details."""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from haptos.config import DEFAULT_CONFIDENCE, DEFAULT_DETECTOR_BACKEND, DEFAULT_MODEL  # noqa: E402
from haptos.cv.camera import VideoSource  # noqa: E402
from haptos.cv.detector import create_detector  # noqa: E402
from haptos.cv.postprocess import filter_and_enrich_detections  # noqa: E402
from haptos.cv.stereo import StereoDepthEstimator, measure_detection_depth  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug one-frame object depth measurements")
    parser.add_argument("--source", default="picamera0", help="Left/source camera")
    parser.add_argument("--stereo-right-source", default="picamera1", help="Right camera")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="YOLO model path/name")
    parser.add_argument("--backend", default=DEFAULT_DETECTOR_BACKEND, choices=["ultralytics", "ncnn"])
    parser.add_argument("--conf", type=float, default=DEFAULT_CONFIDENCE, help="Confidence threshold")
    parser.add_argument("--stereo-baseline-m", type=float, help="Rough stereo baseline in meters")
    parser.add_argument("--stereo-focal-px", type=float, help="Rough focal length in pixels")
    parser.add_argument("--stereo-calibration", help="Optional stereo calibration/rectification .npz file")
    parser.add_argument("--stereo-num-disparities", type=int, default=64)
    parser.add_argument("--stereo-block-size", type=int, default=5)
    parser.add_argument("--depth-bbox-scale", type=float, default=0.6)
    parser.add_argument("--warmup", type=float, default=1.0, help="Seconds to let cameras settle before capture")
    parser.add_argument("--output-image", help="Optional debug image with detection and depth-sampling boxes")
    args = parser.parse_args()
    if args.warmup < 0:
        parser.error("--warmup must be 0 or greater")
    if args.depth_bbox_scale <= 0.0 or args.depth_bbox_scale > 1.0:
        parser.error("--depth-bbox-scale must be greater than 0 and less than or equal to 1")
    return args


def main() -> int:
    args = parse_args()
    left_source = VideoSource(args.source)
    right_source = VideoSource(args.stereo_right_source)
    try:
        if args.warmup:
            time.sleep(args.warmup)

        left_ok, left_frame = left_source.read()
        right_ok, right_frame = right_source.read()
        if not left_ok or left_frame is None:
            raise RuntimeError(f"No frame received from source '{args.source}'")
        if not right_ok or right_frame is None:
            raise RuntimeError(f"No frame received from stereo right source '{args.stereo_right_source}'")

        depth_estimator = StereoDepthEstimator(
            num_disparities=args.stereo_num_disparities,
            block_size=args.stereo_block_size,
            baseline_m=args.stereo_baseline_m,
            focal_px=args.stereo_focal_px,
            calibration_path=args.stereo_calibration,
        )
        detector = create_detector(args.backend, args.model, args.conf)

        depth_started_at = time.perf_counter()
        stereo_frame = depth_estimator.estimate_frame(left_frame, right_frame)
        depth_latency_ms = (time.perf_counter() - depth_started_at) * 1000.0

        cv_started_at = time.perf_counter()
        raw_detections = detector.detect(left_frame)
        cv_latency_ms = (time.perf_counter() - cv_started_at) * 1000.0

        _, frame_width = left_frame.shape[:2]
        detections = filter_and_enrich_detections(raw_detections, frame_width, args.conf)
        result = {
            "source": args.source,
            "stereo_right_source": args.stereo_right_source,
            "image_shape": list(left_frame.shape),
            "cv_latency_ms": cv_latency_ms,
            "depth_latency_ms": depth_latency_ms,
            "stereo_depth_summary": stereo_frame.summary.to_dict(),
            "detections": [],
        }

        debug_frame = left_frame.copy()
        for detection in detections:
            measurement = None
            if stereo_frame.depth_m is not None:
                measurement = measure_detection_depth(
                    detection,
                    stereo_frame.depth_m,
                    bbox_scale=args.depth_bbox_scale,
                )

            result["detections"].append(
                {
                    "class_name": detection.class_name,
                    "confidence": detection.confidence,
                    "region": detection.region,
                    "bbox": list(detection.bbox),
                    "depth_sample_bbox": list(measurement.sample_bbox) if measurement else None,
                    "median_depth_m": measurement.median_depth_m if measurement else None,
                    "depth_valid_pixel_count": measurement.valid_pixel_count if measurement else 0,
                }
            )
            draw_debug_boxes(debug_frame, detection.bbox, measurement.sample_bbox if measurement else None)

        if args.output_image:
            output_path = Path(args.output_image)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), debug_frame)
            result["output_image"] = str(output_path)

        print(json.dumps(result, indent=2))
        return 0
    finally:
        left_source.release()
        right_source.release()


def draw_debug_boxes(frame, detection_bbox, sample_bbox) -> None:
    x1, y1, x2, y2 = detection_bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    if sample_bbox is not None:
        sx1, sy1, sx2, sy2 = sample_bbox
        cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), (0, 255, 255), 2)


if __name__ == "__main__":
    raise SystemExit(main())
