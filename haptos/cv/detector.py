"""Object detector backends for Haptos CV."""

from pathlib import Path
import re
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np

from haptos.config import DETECTOR_BACKEND_NCNN, DETECTOR_BACKEND_ULTRALYTICS
from haptos.types import Detection

COCO_NAMES: Dict[int, str] = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    4: "airplane",
    5: "bus",
    6: "train",
    7: "truck",
    8: "boat",
    9: "traffic light",
    10: "fire hydrant",
    11: "stop sign",
    12: "parking meter",
    13: "bench",
    14: "bird",
    15: "cat",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
    20: "elephant",
    21: "bear",
    22: "zebra",
    23: "giraffe",
    24: "backpack",
    25: "umbrella",
    26: "handbag",
    27: "tie",
    28: "suitcase",
    29: "frisbee",
    30: "skis",
    31: "snowboard",
    32: "sports ball",
    33: "kite",
    34: "baseball bat",
    35: "baseball glove",
    36: "skateboard",
    37: "surfboard",
    38: "tennis racket",
    39: "bottle",
    40: "wine glass",
    41: "cup",
    42: "fork",
    43: "knife",
    44: "spoon",
    45: "bowl",
    46: "banana",
    47: "apple",
    48: "sandwich",
    49: "orange",
    50: "broccoli",
    51: "carrot",
    52: "hot dog",
    53: "pizza",
    54: "donut",
    55: "cake",
    56: "chair",
    57: "couch",
    58: "potted plant",
    59: "bed",
    60: "dining table",
    61: "toilet",
    62: "tv",
    63: "laptop",
    64: "mouse",
    65: "remote",
    66: "keyboard",
    67: "cell phone",
    68: "microwave",
    69: "oven",
    70: "toaster",
    71: "sink",
    72: "refrigerator",
    73: "book",
    74: "clock",
    75: "vase",
    76: "scissors",
    77: "teddy bear",
    78: "hair drier",
    79: "toothbrush",
}


class YoloDetector:
    """Thin wrapper around Ultralytics YOLO for laptop development."""

    def __init__(self, model_name: str, confidence: float):
        self.confidence = confidence
        try:
            from ultralytics import YOLO
        except ModuleNotFoundError as exc:
            raise RuntimeError("Ultralytics is not installed. Run: pip install -r requirements.txt") from exc

        try:
            self.model = YOLO(model_name)
        except Exception as exc:
            raise RuntimeError(f"Failed to load YOLO model '{model_name}': {exc}") from exc

    def detect(self, frame) -> List[Detection]:
        """Run YOLO and return class, confidence, and bounding box per object."""

        results = self.model.predict(frame, conf=self.confidence, verbose=False)
        if not results:
            return []

        result = results[0]
        detections: List[Detection] = []

        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = self.model.names.get(class_id, str(class_id))
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            detections.append(
                Detection(
                    class_name=class_name,
                    confidence=confidence,
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                )
            )

        return detections


class NcnnYoloDetector:
    """YOLOv8 detector for Ultralytics NCNN exports.

    This backend avoids importing Ultralytics and PyTorch at runtime. It expects
    an exported model directory containing model.ncnn.param, model.ncnn.bin, and
    optionally metadata.yaml.
    """

    def __init__(self, model_dir: str, confidence: float, iou_threshold: float = 0.45):
        self.model_dir = Path(model_dir)
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.img_size = _read_img_size(self.model_dir / "metadata.yaml")
        self.names = _read_names(self.model_dir / "metadata.yaml")

        param_path = self.model_dir / "model.ncnn.param"
        bin_path = self.model_dir / "model.ncnn.bin"
        if not param_path.exists() or not bin_path.exists():
            raise RuntimeError(
                f"NCNN model directory '{model_dir}' must contain model.ncnn.param and model.ncnn.bin."
            )

        try:
            import ncnn
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "The NCNN backend requires the ncnn Python package. On the Pi, activate the venv and run: "
                "pip install ncnn"
            ) from exc

        self.ncnn = ncnn
        self.net = ncnn.Net()
        self.net.load_param(str(param_path))
        self.net.load_model(str(bin_path))

    def detect(self, frame) -> List[Detection]:
        input_tensor, scale, pad_left, pad_top = _preprocess_frame(frame, self.img_size)

        with self.net.create_extractor() as extractor:
            extractor.input("in0", self.ncnn.Mat(input_tensor).clone())
            status, output = extractor.extract("out0")
            if status != 0:
                raise RuntimeError(f"NCNN inference failed with status {status}.")

        predictions = np.array(output)
        detections = _decode_yolov8_output(
            predictions=predictions,
            frame_shape=frame.shape[:2],
            scale=scale,
            pad_left=pad_left,
            pad_top=pad_top,
            confidence_threshold=self.confidence,
            iou_threshold=self.iou_threshold,
            names=self.names,
        )
        return detections


def create_detector(backend: str, model_name: str, confidence: float):
    """Create a detector backend from CLI configuration."""

    if backend == DETECTOR_BACKEND_ULTRALYTICS:
        return YoloDetector(model_name, confidence)
    if backend == DETECTOR_BACKEND_NCNN:
        return NcnnYoloDetector(model_name, confidence)
    raise ValueError(f"Unsupported detector backend: {backend}")


def _preprocess_frame(frame, img_size: int) -> Tuple[np.ndarray, float, float, float]:
    height, width = frame.shape[:2]
    scale = min(img_size / height, img_size / width)
    resized_width = int(round(width * scale))
    resized_height = int(round(height * scale))

    resized = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    pad_left = (img_size - resized_width) / 2.0
    pad_top = (img_size - resized_height) / 2.0
    pad_right = img_size - resized_width - int(round(pad_left - 0.1))
    pad_bottom = img_size - resized_height - int(round(pad_top - 0.1))
    pad_left_int = int(round(pad_left - 0.1))
    pad_top_int = int(round(pad_top - 0.1))

    padded = cv2.copyMakeBorder(
        resized,
        pad_top_int,
        pad_bottom,
        pad_left_int,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    tensor = rgb.astype(np.float32) / 255.0
    tensor = np.transpose(tensor, (2, 0, 1))
    return np.ascontiguousarray(tensor), scale, float(pad_left_int), float(pad_top_int)


def _decode_yolov8_output(
    predictions: np.ndarray,
    frame_shape: Tuple[int, int],
    scale: float,
    pad_left: float,
    pad_top: float,
    confidence_threshold: float,
    iou_threshold: float,
    names: Dict[int, str],
) -> List[Detection]:
    pred = np.squeeze(predictions)
    if pred.ndim == 1:
        pred = pred.reshape(1, -1)
    if pred.ndim != 2:
        raise RuntimeError(f"Unexpected NCNN output shape: {predictions.shape}")

    if pred.shape[1] < 5 and pred.shape[0] >= 5:
        pred = pred.T
    elif pred.shape[1] > 100 and pred.shape[0] <= 100:
        pred = pred.T

    boxes_xywh = pred[:, :4]
    class_scores = pred[:, 4:]
    class_ids = np.argmax(class_scores, axis=1)
    confidences = class_scores[np.arange(class_scores.shape[0]), class_ids]
    keep = confidences >= confidence_threshold

    if not np.any(keep):
        return []

    boxes_xyxy = _xywh_to_xyxy(boxes_xywh[keep])
    boxes_xyxy[:, [0, 2]] = (boxes_xyxy[:, [0, 2]] - pad_left) / scale
    boxes_xyxy[:, [1, 3]] = (boxes_xyxy[:, [1, 3]] - pad_top) / scale
    boxes_xyxy = _clip_boxes(boxes_xyxy, frame_shape)
    class_ids = class_ids[keep]
    confidences = confidences[keep]

    keep_indices = _classwise_nms(boxes_xyxy, confidences, class_ids, iou_threshold)
    detections: List[Detection] = []
    for index in keep_indices:
        x1, y1, x2, y2 = boxes_xyxy[index]
        if x2 <= x1 or y2 <= y1:
            continue
        class_id = int(class_ids[index])
        detections.append(
            Detection(
                class_name=names.get(class_id, str(class_id)),
                confidence=float(confidences[index]),
                bbox=(int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))),
            )
        )

    return detections


def _xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    converted = boxes.astype(np.float32).copy()
    converted[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
    converted[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
    converted[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
    converted[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
    return converted


def _clip_boxes(boxes: np.ndarray, frame_shape: Tuple[int, int]) -> np.ndarray:
    height, width = frame_shape
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, width - 1)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, height - 1)
    return boxes


def _classwise_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    iou_threshold: float,
) -> List[int]:
    selected: List[int] = []
    for class_id in np.unique(class_ids):
        indices = np.where(class_ids == class_id)[0]
        selected.extend(_nms(boxes[indices], scores[indices], iou_threshold, indices))
    return sorted(selected, key=lambda i: float(scores[i]), reverse=True)


def _nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float,
    original_indices: Sequence[int],
) -> List[int]:
    order = np.argsort(scores)[::-1]
    keep: List[int] = []

    while order.size > 0:
        current = order[0]
        keep.append(int(original_indices[current]))
        if order.size == 1:
            break

        ious = _box_iou(boxes[current], boxes[order[1:]])
        order = order[1:][ious <= iou_threshold]

    return keep


def _box_iou(box: np.ndarray, other_boxes: np.ndarray) -> np.ndarray:
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


def _read_img_size(metadata_path: Path) -> int:
    if not metadata_path.exists():
        return 640

    lines = metadata_path.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        if line.strip() == "imgsz:" and index + 1 < len(lines):
            match = re.search(r"-\s*(\d+)", lines[index + 1])
            if match:
                return int(match.group(1))
    return 640


def _read_names(metadata_path: Path) -> Dict[int, str]:
    if not metadata_path.exists():
        return COCO_NAMES

    names: Dict[int, str] = {}
    in_names = False
    for line in metadata_path.read_text(encoding="utf-8").splitlines():
        if line.strip() == "names:":
            in_names = True
            continue
        if in_names and line and not line.startswith(" "):
            break
        if in_names:
            match = re.match(r"\s*(\d+):\s*(.+?)\s*$", line)
            if match:
                names[int(match.group(1))] = match.group(2).strip("'\"")

    return names or COCO_NAMES
