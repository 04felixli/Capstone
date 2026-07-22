"""Tests for CV utility formatting."""

import unittest

from haptos.cv.utils import format_console_result
from haptos.types import Detection, FrameResult


class ConsoleFormattingTests(unittest.TestCase):
    def test_format_console_result_includes_detection_depth_and_latencies(self):
        result = FrameResult(
            frame_index=63,
            command="STOP",
            detections=[
                Detection(
                    class_name="person",
                    confidence=0.91,
                    bbox=(10, 20, 30, 40),
                    region="CENTER",
                    is_obstacle=True,
                    median_depth_m=0.83,
                    depth_pixel_count=100,
                )
            ],
            fps=1.0,
            cv_latency_ms=123.45,
            depth_latency_ms=67.89,
        )

        self.assertEqual(
            format_console_result(result),
            "Frame 63 | detections=person:center:0.91:0.83m | cv_latency=123.5ms | depth_latency=67.9ms",
        )

    def test_format_console_result_handles_missing_depth_latency(self):
        result = FrameResult(
            frame_index=64,
            command="FORWARD",
            detections=[],
            fps=1.0,
            cv_latency_ms=50.0,
        )

        self.assertEqual(
            format_console_result(result),
            "Frame 64 | detections=none | cv_latency=50.0ms | depth_latency=n/a",
        )


if __name__ == "__main__":
    unittest.main()
