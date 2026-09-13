"""Bounding-box geometry helpers shared across detection code.

Kept free of cv2/heavier dependencies so pure-math consumers don't have to
pull in drawing/video libraries just for IoU math.
"""

import numpy as np


def box_iou(box: np.ndarray, other_boxes: np.ndarray) -> np.ndarray:
    """Vectorized IoU between one (x1, y1, x2, y2) box and an array of boxes."""

    x1 = np.maximum(box[0], other_boxes[:, 0])
    y1 = np.maximum(box[1], other_boxes[:, 1])
    x2 = np.minimum(box[2], other_boxes[:, 2])
    y2 = np.minimum(box[3], other_boxes[:, 3])

    intersection = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    box_area = max(0.0, float((box[2] - box[0]) * (box[3] - box[1])))
    other_areas = np.maximum(0.0, other_boxes[:, 2] - other_boxes[:, 0]) * np.maximum(
        0.0,
        other_boxes[:, 3] - other_boxes[:, 1],
    )
    union = box_area + other_areas - intersection
    return intersection / np.maximum(union, 1e-6)
