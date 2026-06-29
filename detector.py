"""YOLO model loading and frame inference."""

from typing import List

from ultralytics import YOLO

from haptos_types import Detection


class YoloDetector:
    """Thin wrapper around Ultralytics YOLO for Haptos detections."""

    def __init__(self, model_name: str, confidence: float):
        self.confidence = confidence
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
