"""OpenCV video/image input helpers."""

from pathlib import Path
from typing import Optional, Tuple, Union

import cv2

from haptos.config import IMAGE_EXTENSIONS


class VideoSource:
    """Read frames from a webcam, video file, or single image.

    OpenCV returns images as NumPy arrays in BGR color order. The detector and
    drawing utilities can use that format directly.
    """

    def __init__(self, source: str):
        self.source = source
        self._capture: Optional[cv2.VideoCapture] = None
        self._image = None
        self._image_returned = False
        self._is_image = False
        self._open_source(source)

    def _open_source(self, source: str) -> None:
        if source.lower() == "webcam":
            self._capture = cv2.VideoCapture(0)
            if not self._capture.isOpened():
                raise RuntimeError("Could not open webcam index 0.")
            return

        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Input source does not exist: {source}")

        if path.suffix.lower() in IMAGE_EXTENSIONS:
            self._is_image = True
            self._image = cv2.imread(str(path))
            if self._image is None:
                raise RuntimeError(f"Could not read image file: {source}")
            return

        self._capture = cv2.VideoCapture(str(path))
        if not self._capture.isOpened():
            raise RuntimeError(f"Could not open video file: {source}")

    @property
    def is_image(self) -> bool:
        return self._is_image

    def read(self) -> Tuple[bool, Optional[Union[cv2.Mat, object]]]:
        """Return (success, frame). success becomes False when input is done."""

        if self._is_image:
            if self._image_returned:
                return False, None
            self._image_returned = True
            return True, self._image.copy()

        if self._capture is None:
            return False, None

        success, frame = self._capture.read()
        if not success or frame is None:
            return False, None
        return True, frame

    def release(self) -> None:
        if self._capture is not None:
            self._capture.release()
