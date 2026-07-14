"""Tests for camera source helpers."""

import unittest

from haptos.cv.camera import _parse_picamera_index


class PicameraSourceTests(unittest.TestCase):
    def test_parse_picamera_default(self):
        self.assertEqual(_parse_picamera_index("picamera"), 0)

    def test_parse_picamera_suffix(self):
        self.assertEqual(_parse_picamera_index("picamera0"), 0)
        self.assertEqual(_parse_picamera_index("picamera1"), 1)

    def test_parse_picamera_colon(self):
        self.assertEqual(_parse_picamera_index("picamera:0"), 0)
        self.assertEqual(_parse_picamera_index("picamera:1"), 1)

    def test_parse_picamera_rejects_invalid_source(self):
        with self.assertRaises(ValueError):
            _parse_picamera_index("picam")


if __name__ == "__main__":
    unittest.main()
