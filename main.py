"""Command-line entry point for the Haptos computer vision subsystem."""

import argparse
import sys
import time

import cv2

from haptos.cv.camera import VideoSource
from haptos.cv.detector import create_detector
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
)
from haptos.cv.postprocess import filter_and_enrich_detections
from haptos.cv.utils import FPSCounter, JsonlLogger, draw_overlay, format_console_result, sleep_to_maintain_rate
from haptos.fusion.hazard_decision import generate_fused_navigation_hint
from haptos.sensor.lidar_buffer import LidarFrameBuffer
from haptos.sensor.lidar_filter import filter_lidar_scan
from haptos.sensor.lidar_reader import create_lidar_reader
from haptos.types import FrameResult


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Haptos laptop-testable CV module")

    # Core detector settings.
    parser.add_argument("--source", required=True, help="'webcam' or path to a video file")
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

    # LiDAR is the only source of metric range; the camera is classification/region only.
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
    # Fusion: the only distance threshold, since the camera no longer carries depth.
    parser.add_argument(
        "--emergency-stop-distance-m",
        type=float,
        default=HAZARD_DEFAULT_EMERGENCY_STOP_DISTANCE_M,
        help="Any trusted LiDAR obstacle at this distance triggers STOP.",
    )
    args = parser.parse_args()
    if args.fps < 0:
        parser.error("--fps must be 0 or greater")
    if args.emergency_stop_distance_m <= 0:
        parser.error("--emergency-stop-distance-m must be positive")
    return args


def main() -> int:
    args = parse_args()

    # Declared before the try block so `finally` can safely release whatever
    # actually got constructed, even if setup fails partway through.
    source = None
    logger = None
    lidar_reader = None
    try:
        source = VideoSource(args.source)
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
        
        # Main loop: read a frame, detect, optionally read LiDAR, fuse, log, display.
        while True:
            loop_started_at = time.monotonic()

            # 1. Read a frame. End-of-stream on the very first frame means a bad
            # source; end-of-stream later just means the video/stream finished.
            ok, frame = source.read()
            if not ok or frame is None:
                if frame_index == 0:
                    raise RuntimeError(f"No frames received from source '{args.source}'.")
                break

            # 2. Detect, then enrich with region + is_obstacle (postprocess.py).
            frame_index += 1
            cv_started_at = time.perf_counter()
            raw_detections = detector.detect(frame)
            cv_latency_ms = (time.perf_counter() - cv_started_at) * 1000.0
            _, frame_width = frame.shape[:2]
            detections = filter_and_enrich_detections(raw_detections, frame_width, args.conf)
            lidar_summary = None

            # 3. LiDAR: independent channel, read/filtered/summarized every loop
            # iteration regardless of what the camera saw.
            if lidar_reader is not None and lidar_buffer is not None:
                raw_lidar_scan = lidar_reader.read()
                filtered_lidar = filter_lidar_scan(raw_lidar_scan)
                lidar_buffer.add(filtered_lidar)
                lidar_summary = filtered_lidar.to_summary()

            # 4. Fusion: LiDAR near-range override, else camera region logic.
            command = generate_fused_navigation_hint(
                detections,
                lidar_summary,
                emergency_stop_distance_m=args.emergency_stop_distance_m,
            )
            fps = fps_counter.update()

            # 5. Package + emit.
            result = FrameResult(
                frame_index=frame_index,
                command=command,
                detections=detections,
                fps=fps,
                cv_latency_ms=cv_latency_ms,
                lidar_summary=lidar_summary,
            )

            print(format_console_result(result))
            if logger is not None:
                logger.write(result)

            if args.show:
                draw_overlay(frame, result)
                cv2.imshow("Haptos CV", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            # 6. Loop control: cap the loop rate at --fps.
            sleep_to_maintain_rate(loop_started_at, frame_interval_s)

        return 0

    except KeyboardInterrupt:
        print("Interrupted by user.", file=sys.stderr)
        return 130
    except Exception as exc:
        # Any other failure becomes a clean message + exit code 1 instead of
        # a raw traceback.
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    finally:
        # Release whatever was actually constructed, regardless of where
        # setup or the loop failed.
        if source is not None:
            source.release()
        if logger is not None:
            logger.close()
        if lidar_reader is not None:
            lidar_reader.close()
        if args.show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
