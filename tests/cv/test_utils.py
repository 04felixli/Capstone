"""Tests for CV utility formatting."""

import unittest

from haptos.cv.utils import format_console_result
from haptos.types import Detection, FrameResult


class ConsoleFormattingTests(unittest.TestCase):
    def test_format_console_result_includes_detections_and_latency(self):
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
                )
            ],
            fps=1.0,
            cv_latency_ms=123.45,
        )

        self.assertEqual(
            format_console_result(result),
            "Frame 63 | detections=person:center:0.91 | cv_latency=123.5ms",
        )

    def test_format_console_result_handles_missing_detections(self):
        result = FrameResult(
            frame_index=64,
            command="FORWARD",
            detections=[],
            fps=1.0,
            cv_latency_ms=50.0,
        )

        self.assertEqual(
            format_console_result(result),
            "Frame 64 | detections=none | cv_latency=50.0ms",
        )


if __name__ == "__main__":
    unittest.main()
