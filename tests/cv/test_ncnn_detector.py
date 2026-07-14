"""Tests for NCNN YOLO post-processing helpers."""

import unittest

import numpy as np

from haptos.cv.detector import _decode_yolov8_output, _preprocess_frame


class NcnnDetectorPostprocessTests(unittest.TestCase):
    def test_preprocess_frame_letterboxes_to_square_chw_tensor(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        tensor, scale, pad_left, pad_top = _preprocess_frame(frame, 640)

        self.assertEqual(tensor.shape, (3, 640, 640))
        self.assertEqual(tensor.dtype, np.float32)
        self.assertAlmostEqual(scale, 1.0)
        self.assertAlmostEqual(pad_left, 0.0)
        self.assertAlmostEqual(pad_top, 80.0)

    def test_decode_yolov8_output_returns_detection_in_original_frame_coordinates(self):
        # YOLOv8 NCNN output is normally shaped (84, anchors). This synthetic
        # output has one confident "person" box after 80 px top padding.
        predictions = np.zeros((84, 1), dtype=np.float32)
        predictions[0, 0] = 320.0  # center x
        predictions[1, 0] = 320.0  # center y in padded image
        predictions[2, 0] = 100.0  # width
        predictions[3, 0] = 200.0  # height
        predictions[4, 0] = 0.90  # class 0 score

        detections = _decode_yolov8_output(
            predictions=predictions,
            frame_shape=(480, 640),
            scale=1.0,
            pad_left=0.0,
            pad_top=80.0,
            confidence_threshold=0.4,
            iou_threshold=0.45,
            names={0: "person"},
        )

        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].class_name, "person")
        self.assertAlmostEqual(detections[0].confidence, 0.90, places=5)
        self.assertEqual(detections[0].bbox, (270, 140, 370, 340))

    def test_decode_yolov8_output_applies_classwise_nms(self):
        predictions = np.zeros((84, 2), dtype=np.float32)
        predictions[0, :] = [320.0, 322.0]
        predictions[1, :] = [320.0, 322.0]
        predictions[2, :] = [100.0, 100.0]
        predictions[3, :] = [100.0, 100.0]
        predictions[4, :] = [0.95, 0.80]

        detections = _decode_yolov8_output(
            predictions=predictions,
            frame_shape=(640, 640),
            scale=1.0,
            pad_left=0.0,
            pad_top=0.0,
            confidence_threshold=0.4,
            iou_threshold=0.45,
            names={0: "person"},
        )

        self.assertEqual(len(detections), 1)
        self.assertAlmostEqual(detections[0].confidence, 0.95, places=5)


if __name__ == "__main__":
    unittest.main()
