"""Tests for stereo calibration capture helpers."""

import unittest

import numpy as np

from scripts.capture_stereo_pairs import (
    detect_checkerboard,
    format_validation_result,
    make_side_by_side_preview,
    validate_checkerboard_pair,
)


class StereoPairCaptureTests(unittest.TestCase):
    def test_make_side_by_side_preview_concatenates_frames(self):
        left = np.zeros((4, 5, 3), dtype=np.uint8)
        right = np.zeros((4, 5, 3), dtype=np.uint8)

        preview = make_side_by_side_preview(left, right)

        self.assertEqual(preview.shape, (4, 10, 3))

    def test_make_side_by_side_preview_resizes_right_frame(self):
        left = np.zeros((4, 5, 3), dtype=np.uint8)
        right = np.zeros((2, 3, 3), dtype=np.uint8)

        preview = make_side_by_side_preview(left, right)

        self.assertEqual(preview.shape, (4, 10, 3))

    def test_detect_checkerboard_finds_synthetic_pattern(self):
        frame = _make_checkerboard_frame(inner_cols=4, inner_rows=3, square_px=30)

        self.assertTrue(detect_checkerboard(frame, (4, 3)))

    def test_validate_checkerboard_pair_requires_both_cameras(self):
        checkerboard = _make_checkerboard_frame(inner_cols=4, inner_rows=3, square_px=30)
        blank = np.zeros_like(checkerboard)

        validation = validate_checkerboard_pair(checkerboard, blank, 4, 3)

        self.assertFalse(validation["valid"])
        self.assertTrue(validation["left_found"])
        self.assertFalse(validation["right_found"])
        self.assertEqual(
            format_validation_result(validation),
            "Checkerboard: INVALID (right camera did not detect the full pattern)",
        )

    def test_validate_checkerboard_pair_can_be_disabled(self):
        frame = np.zeros((4, 5, 3), dtype=np.uint8)

        self.assertIsNone(validate_checkerboard_pair(frame, frame, None, None))


def _make_checkerboard_frame(inner_cols: int, inner_rows: int, square_px: int) -> np.ndarray:
    square_cols = inner_cols + 1
    square_rows = inner_rows + 1
    image = np.zeros((square_rows * square_px, square_cols * square_px), dtype=np.uint8)
    for row in range(square_rows):
        for col in range(square_cols):
            if (row + col) % 2 == 0:
                y1 = row * square_px
                x1 = col * square_px
                image[y1 : y1 + square_px, x1 : x1 + square_px] = 255
    return np.dstack([image, image, image])


if __name__ == "__main__":
    unittest.main()
