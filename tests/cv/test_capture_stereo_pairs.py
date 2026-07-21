"""Tests for stereo calibration capture helpers."""

import unittest

import numpy as np

from scripts.capture_stereo_pairs import make_side_by_side_preview


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


if __name__ == "__main__":
    unittest.main()
