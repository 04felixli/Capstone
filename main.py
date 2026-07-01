"""Command-line entry point for the Haptos computer vision subsystem."""

import argparse
import sys

import cv2

from haptos.cv.camera import VideoSource
from haptos.config import (
    DEFAULT_CONFIDENCE,
    DEFAULT_MODEL,
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
    parser.add_argument("--conf", type=float, default=DEFAULT_CONFIDENCE, help="Confidence threshold")
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    source = None
    logger = None
    try:
        source = VideoSource(args.source)
        try:
            from haptos.cv.detector import YoloDetector
        except ModuleNotFoundError as exc:
            if exc.name == "ultralytics":
                raise RuntimeError("Ultralytics is not installed. Run: pip install -r requirements.txt") from exc
            raise

        detector = YoloDetector(args.model, args.conf)
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
        while True:
            ok, frame = source.read()
            if not ok or frame is None:
                break

            frame_index += 1
            raw_detections = detector.detect(frame)
            frame_height, frame_width = frame.shape[:2]
            detections = filter_and_enrich_detections(raw_detections, frame_width, args.conf)
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
        if logger is not None:
            logger.close()
        if "lidar_reader" in locals() and lidar_reader is not None:
            lidar_reader.close()
        if args.show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
