"""Command-line entry point for the Haptos computer vision subsystem."""

import argparse
import sys

import cv2

from camera import VideoSource
from config import DEFAULT_CONFIDENCE, DEFAULT_MODEL
from detector import YoloDetector
from haptos_types import FrameResult
from postprocess import filter_and_enrich_detections, generate_navigation_hint
from utils import FPSCounter, JsonlLogger, draw_overlay, format_console_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Haptos laptop-testable CV module")
    parser.add_argument("--source", required=True, help="'webcam' or path to a video/image file")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="YOLO model path/name")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONFIDENCE, help="Confidence threshold")
    parser.add_argument("--show", action="store_true", help="Display annotated frames")
    parser.add_argument("--save-log", help="Optional path for JSONL frame results")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    source = None
    logger = None
    try:
        source = VideoSource(args.source)
        detector = YoloDetector(args.model, args.conf)
        fps_counter = FPSCounter()
        logger = JsonlLogger(args.save_log) if args.save_log else None

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
            fps = fps_counter.update()

            result = FrameResult(
                frame_index=frame_index,
                command=command,
                detections=detections,
                fps=fps,
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
        if args.show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
