"""Command-line entry point for the Haptos computer vision subsystem."""

import argparse
import sys
import time

import cv2

from haptos.cv.camera import StereoVideoSource, VideoSource
from haptos.cv.depth_smoother import DetectionDepthSmoother
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
    HAZARD_DEFAULT_EMERGENCY_STOP_DISTANCE_M,
    HAZARD_DEFAULT_MAX_DISTANCE_M,
    STEREO_DEFAULT_CAMERA_FPS,
    STEREO_DEFAULT_MAX_DEPTH_M,
    STEREO_DEFAULT_MAX_RELATIVE_UNCERTAINTY,
    STEREO_DEFAULT_MAX_SKEW_MS,
    STEREO_DEFAULT_MIN_VALID_FRACTION,
    STEREO_DEFAULT_SMOOTHING_WINDOW,
)
from haptos.cv.postprocess import filter_and_enrich_detections
from haptos.cv.utils import FPSCounter, JsonlLogger, draw_overlay, format_console_result
from haptos.fusion.hazard_decision import generate_fused_navigation_hint
from haptos.sensor.lidar_buffer import LidarFrameBuffer
from haptos.sensor.lidar_filter import filter_lidar_scan
from haptos.sensor.lidar_reader import create_lidar_reader
from haptos.types import FrameResult, StereoDepthSummary


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
    parser.add_argument(
        "--depth-bbox-scale",
        type=float,
        default=0.6,
        help="Center fraction of each detection bbox used for object depth. 1.0 uses the full box.",
    )
    parser.add_argument(
        "--stereo-camera-fps",
        type=float,
        default=STEREO_DEFAULT_CAMERA_FPS,
        help="Fixed capture FPS used to synchronize Raspberry Pi stereo cameras.",
    )
    parser.add_argument(
        "--stereo-max-skew-ms",
        type=float,
        default=STEREO_DEFAULT_MAX_SKEW_MS,
        help="Maximum accepted timestamp difference between left and right frames.",
    )
    parser.add_argument(
        "--stereo-max-depth-m",
        type=float,
        default=STEREO_DEFAULT_MAX_DEPTH_M,
        help="Discard stereo depth values beyond this distance.",
    )
    parser.add_argument(
        "--depth-min-valid-fraction",
        type=float,
        default=STEREO_DEFAULT_MIN_VALID_FRACTION,
        help="Minimum valid depth-pixel fraction required inside a detection box.",
    )
    parser.add_argument(
        "--depth-max-relative-uncertainty",
        type=float,
        default=STEREO_DEFAULT_MAX_RELATIVE_UNCERTAINTY,
        help="Reject object depths whose robust uncertainty is too large relative to distance.",
    )
    parser.add_argument(
        "--depth-smoothing-window",
        type=int,
        default=STEREO_DEFAULT_SMOOTHING_WINDOW,
        help="Number of recent matched object depths used for temporal median smoothing.",
    )
    parser.add_argument(
        "--hazard-distance-m",
        type=float,
        default=HAZARD_DEFAULT_MAX_DISTANCE_M,
        help="Trusted objects farther than this do not affect the navigation command.",
    )
    parser.add_argument(
        "--emergency-stop-distance-m",
        type=float,
        default=HAZARD_DEFAULT_EMERGENCY_STOP_DISTANCE_M,
        help="Any trusted camera or LiDAR obstacle at this distance triggers STOP.",
    )
    args = parser.parse_args()
    if args.fps < 0:
        parser.error("--fps must be 0 or greater")
    if args.depth_bbox_scale <= 0.0 or args.depth_bbox_scale > 1.0:
        parser.error("--depth-bbox-scale must be greater than 0 and less than or equal to 1")
    if args.stereo_camera_fps <= 0:
        parser.error("--stereo-camera-fps must be positive")
    if args.stereo_max_skew_ms < 0:
        parser.error("--stereo-max-skew-ms must be 0 or greater")
    if args.stereo_max_depth_m <= 0:
        parser.error("--stereo-max-depth-m must be positive")
    if args.depth_min_valid_fraction < 0.0 or args.depth_min_valid_fraction > 1.0:
        parser.error("--depth-min-valid-fraction must be between 0 and 1")
    if args.depth_max_relative_uncertainty <= 0:
        parser.error("--depth-max-relative-uncertainty must be positive")
    if args.depth_smoothing_window <= 0:
        parser.error("--depth-smoothing-window must be positive")
    if args.hazard_distance_m <= 0:
        parser.error("--hazard-distance-m must be positive")
    if args.emergency_stop_distance_m <= 0:
        parser.error("--emergency-stop-distance-m must be positive")
    if args.emergency_stop_distance_m > args.hazard_distance_m:
        parser.error("--emergency-stop-distance-m cannot exceed --hazard-distance-m")
    if args.stereo_depth and args.stereo_left_source is None:
        args.stereo_left_source = args.source
    return args


def main() -> int:
    args = parse_args()

    source = None
    stereo_source = None
    logger = None
    try:
        source_name = args.stereo_left_source if args.stereo_depth else args.source
        stereo_estimator = None
        if args.stereo_depth:
            stereo_source = StereoVideoSource(
                source_name,
                args.stereo_right_source,
                camera_fps=args.stereo_camera_fps,
                max_skew_ms=args.stereo_max_skew_ms,
            )
            stereo_estimator = StereoDepthEstimator(
                num_disparities=args.stereo_num_disparities,
                block_size=args.stereo_block_size,
                baseline_m=args.stereo_baseline_m,
                focal_px=args.stereo_focal_px,
                calibration_path=args.stereo_calibration,
                max_depth_m=args.stereo_max_depth_m,
            )
        else:
            source = VideoSource(source_name)

        try:
            from haptos.cv.detector import create_detector
        except ModuleNotFoundError as exc:
            if exc.name == "ultralytics":
                raise RuntimeError("Ultralytics is not installed. Run: pip install -r requirements.txt") from exc
            raise

        detector = create_detector(args.backend, args.model, args.conf)
        depth_smoother = DetectionDepthSmoother(window_size=args.depth_smoothing_window)
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
            stereo_pair = None
            if stereo_source is not None:
                ok, stereo_pair = stereo_source.read()
                frame = None if stereo_pair is None else stereo_pair.left
            else:
                ok, frame = source.read()
            if not ok or frame is None:
                if frame_index == 0:
                    raise RuntimeError(f"No frames received from source '{source_name}'.")
                break

            stereo_depth_frame = None
            stereo_depth_summary = None
            depth_latency_ms = None
            if stereo_pair is not None and stereo_estimator is not None:
                if stereo_pair.within_tolerance:
                    depth_started_at = time.perf_counter()
                    stereo_depth_frame = stereo_estimator.estimate_frame(
                        stereo_pair.left,
                        stereo_pair.right,
                        frame_skew_ms=stereo_pair.skew_ms,
                    )
                    depth_latency_ms = (time.perf_counter() - depth_started_at) * 1000.0
                    stereo_depth_summary = stereo_depth_frame.summary
                else:
                    stereo_depth_summary = StereoDepthSummary(
                        fault_state="frame_skew",
                        valid_pixel_count=0,
                        frame_skew_ms=stereo_pair.skew_ms,
                    )

            frame_index += 1
            cv_started_at = time.perf_counter()
            raw_detections = detector.detect(frame)
            cv_latency_ms = (time.perf_counter() - cv_started_at) * 1000.0
            _, frame_width = frame.shape[:2]
            detections = filter_and_enrich_detections(raw_detections, frame_width, args.conf)
            if stereo_depth_frame is not None:
                detections = attach_depth_to_detections(
                    detections,
                    stereo_depth_frame.depth_m,
                    max_valid_depth_m=args.stereo_max_depth_m,
                    bbox_scale=args.depth_bbox_scale,
                    min_valid_fraction=args.depth_min_valid_fraction,
                    max_relative_uncertainty=args.depth_max_relative_uncertainty,
                )
                detections = depth_smoother.update(detections)
            lidar_summary = None

            if lidar_reader is not None and lidar_buffer is not None:
                raw_lidar_scan = lidar_reader.read()
                filtered_lidar = filter_lidar_scan(raw_lidar_scan)
                lidar_buffer.add(filtered_lidar)
                lidar_summary = filtered_lidar.to_summary()

            command = generate_fused_navigation_hint(
                detections,
                lidar_summary,
                max_obstacle_distance_m=args.hazard_distance_m,
                emergency_stop_distance_m=args.emergency_stop_distance_m,
                min_depth_valid_fraction=args.depth_min_valid_fraction,
                max_relative_depth_uncertainty=args.depth_max_relative_uncertainty,
            )
            fps = fps_counter.update()

            result = FrameResult(
                frame_index=frame_index,
                command=command,
                detections=detections,
                fps=fps,
                cv_latency_ms=cv_latency_ms,
                depth_latency_ms=depth_latency_ms,
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

            is_image = stereo_source.is_image if stereo_source is not None else source.is_image
            if is_image:
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
        if stereo_source is not None:
            stereo_source.release()
        if logger is not None:
            logger.close()
        if "lidar_reader" in locals() and lidar_reader is not None:
            lidar_reader.close()
        if args.show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
