"""Command-line entry point for the Haptos computer vision subsystem."""

import argparse
import sys
import time

import cv2

from haptos.cv.camera import VideoSource
from haptos.cv.stereo import StereoDepthEstimator, attach_depth_to_detections
from haptos.config import (
    DEFAULT_DETECTOR_BACKEND,
    DEFAULT_CONFIDENCE,
    DEFAULT_MODEL,
    DETECTOR_BACKEND_NCNN,
    DETECTOR_BACKEND_ULTRALYTICS,
    LIDAR_DEFAULT_BAUDRATE,
    LIDAR_DEFAULT_MIN_SAMPLES,
    LIDAR_DEFAULT_SCAN_TIMEOUT_S,
    LIDAR_SOURCE_NONE,
    LIDAR_SOURCE_SERIAL,
)
from haptos.cv.postprocess import filter_and_enrich_detections, generate_navigation_hint
from haptos.cv.utils import FPSCounter, JsonlLogger, draw_overlay, format_console_result
from haptos.sensor.lidar_buffer import LidarFrameBuffer
from haptos.sensor.lidar_filter import filter_lidar_scan
from haptos.sensor.lidar_reader import create_lidar_reader
from haptos.types import FrameResult


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Haptos laptop-testable CV module")
    parser.add_argument("--source", required=True, help="'webcam' or path to a video/image file")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="YOLO model path/name")
    parser.add_argument(
        "--backend",
        choices=[DETECTOR_BACKEND_ULTRALYTICS, DETECTOR_BACKEND_NCNN],
        default=DEFAULT_DETECTOR_BACKEND,
        help="Object detector backend",
    )
    parser.add_argument("--conf", type=float, default=DEFAULT_CONFIDENCE, help="Confidence threshold")
    parser.add_argument(
        "--fps",
        type=float,
        default=0.0,
        help="Optional maximum processing FPS. Use 1 for low-load Pi testing; 0 runs as fast as possible.",
    )
    parser.add_argument("--show", action="store_true", help="Display annotated frames")
    parser.add_argument("--save-log", help="Optional path for JSONL frame results")
    parser.add_argument(
        "--lidar-source",
        choices=[LIDAR_SOURCE_NONE, LIDAR_SOURCE_SERIAL],
        default=LIDAR_SOURCE_NONE,
        help="Optional LiDAR source",
    )
    parser.add_argument(
        "--lidar-port",
        help="Serial port for LiDAR data, for example COM5",
    )
    parser.add_argument(
        "--lidar-baudrate",
        type=int,
        default=LIDAR_DEFAULT_BAUDRATE,
        help="Serial baudrate for LiDAR data",
    )
    parser.add_argument(
        "--lidar-scan-timeout",
        type=float,
        default=LIDAR_DEFAULT_SCAN_TIMEOUT_S,
        help="Maximum seconds to collect samples for one LiDAR scan",
    )
    parser.add_argument(
        "--lidar-min-samples",
        type=int,
        default=LIDAR_DEFAULT_MIN_SAMPLES,
        help="Minimum samples needed before accepting a scan boundary",
    )
    parser.add_argument(
        "--lidar-buffer-size",
        type=int,
        default=10,
        help="Number of recent filtered LiDAR frames to retain",
    )
    parser.add_argument(
        "--stereo-depth",
        action="store_true",
        help="Enable stereo disparity/depth estimation using a second camera source",
    )
    parser.add_argument(
        "--stereo-left-source",
        help="Left camera source for stereo depth. Defaults to --source.",
    )
    parser.add_argument(
        "--stereo-right-source",
        default="picamera1",
        help="Right camera source for stereo depth",
    )
    parser.add_argument(
        "--stereo-num-disparities",
        type=int,
        default=64,
        help="Stereo matcher disparity range. Must be a positive multiple of 16.",
    )
    parser.add_argument(
        "--stereo-block-size",
        type=int,
        default=5,
        help="Stereo matcher block size. Must be an odd integer >= 3.",
    )
    parser.add_argument(
        "--stereo-baseline-m",
        type=float,
        help="Optional distance between camera centers in meters. Requires --stereo-focal-px.",
    )
    parser.add_argument(
        "--stereo-focal-px",
        type=float,
        help="Optional focal length in pixels. Requires --stereo-baseline-m.",
    )
    parser.add_argument(
        "--stereo-calibration",
        help="Optional .npz calibration file from scripts/calibrate_stereo.py. Overrides rough baseline/focal values.",
    )
    args = parser.parse_args()
    if args.fps < 0:
        parser.error("--fps must be 0 or greater")
    if args.stereo_depth and args.stereo_left_source is None:
        args.stereo_left_source = args.source
    return args


def main() -> int:
    args = parse_args()

    source = None
    stereo_right_source = None
    logger = None
    try:
        source_name = args.stereo_left_source if args.stereo_depth else args.source
        source = VideoSource(source_name)
        stereo_estimator = None
        if args.stereo_depth:
            stereo_right_source = VideoSource(args.stereo_right_source)
            stereo_estimator = StereoDepthEstimator(
                num_disparities=args.stereo_num_disparities,
                block_size=args.stereo_block_size,
                baseline_m=args.stereo_baseline_m,
                focal_px=args.stereo_focal_px,
                calibration_path=args.stereo_calibration,
            )

        try:
            from haptos.cv.detector import create_detector
        except ModuleNotFoundError as exc:
            if exc.name == "ultralytics":
                raise RuntimeError("Ultralytics is not installed. Run: pip install -r requirements.txt") from exc
            raise

        detector = create_detector(args.backend, args.model, args.conf)
        fps_counter = FPSCounter()
        logger = JsonlLogger(args.save_log) if args.save_log else None
        lidar_reader = create_lidar_reader(
            source=args.lidar_source,
            port=args.lidar_port,
            baudrate=args.lidar_baudrate,
            scan_timeout_s=args.lidar_scan_timeout,
            min_samples=args.lidar_min_samples,
        )
        lidar_buffer = LidarFrameBuffer(args.lidar_buffer_size) if lidar_reader is not None else None

        frame_index = 0
        frame_interval_s = 1.0 / args.fps if args.fps > 0 else 0.0
        while True:
            loop_started_at = time.monotonic()
            ok, frame = source.read()
            if not ok or frame is None:
                break

            stereo_depth_frame = None
            stereo_depth_summary = None
            if stereo_right_source is not None and stereo_estimator is not None:
                right_ok, right_frame = stereo_right_source.read()
                if not right_ok or right_frame is None:
                    break
                stereo_depth_frame = stereo_estimator.estimate_frame(frame, right_frame)
                stereo_depth_summary = stereo_depth_frame.summary

            frame_index += 1
            raw_detections = detector.detect(frame)
            frame_height, frame_width = frame.shape[:2]
            detections = filter_and_enrich_detections(raw_detections, frame_width, args.conf)
            if stereo_depth_frame is not None:
                detections = attach_depth_to_detections(detections, stereo_depth_frame.depth_m)
            command = generate_navigation_hint(detections)
            lidar_summary = None

            if lidar_reader is not None and lidar_buffer is not None:
                raw_lidar_scan = lidar_reader.read()
                filtered_lidar = filter_lidar_scan(raw_lidar_scan)
                lidar_buffer.add(filtered_lidar)
                lidar_summary = filtered_lidar.to_summary()

            fps = fps_counter.update()

            result = FrameResult(
                frame_index=frame_index,
                command=command,
                detections=detections,
                fps=fps,
                lidar_summary=lidar_summary,
                stereo_depth_summary=stereo_depth_summary,
            )

            print(format_console_result(result))
            if logger is not None:
                logger.write(result)

            if args.show:
                draw_overlay(frame, result)
                cv2.imshow("Haptos CV", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if source.is_image:
                break

            if frame_interval_s > 0:
                elapsed_s = time.monotonic() - loop_started_at
                sleep_s = frame_interval_s - elapsed_s
                if sleep_s > 0:
                    time.sleep(sleep_s)

        return 0

    except KeyboardInterrupt:
        print("Interrupted by user.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    finally:
        if source is not None:
            source.release()
        if stereo_right_source is not None:
            stereo_right_source.release()
        if logger is not None:
            logger.close()
        if "lidar_reader" in locals() and lidar_reader is not None:
            lidar_reader.close()
        if args.show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
