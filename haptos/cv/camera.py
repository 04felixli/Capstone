"""Camera, video, and image input helpers."""

from pathlib import Path
import re
from typing import Any, Optional, Tuple

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
        self._picamera = None
        self._image = None
        self._image_returned = False
        self._is_image = False
        self._open_source(source)

    def _open_source(self, source: str) -> None:
        normalized_source = source.lower()
        if normalized_source == "webcam":
            self._capture = cv2.VideoCapture(0)
            if not self._capture.isOpened():
                raise RuntimeError("Could not open webcam index 0.")
            return

        if normalized_source.startswith("picamera"):
            camera_index = _parse_picamera_index(normalized_source)
            self._open_picamera(camera_index)
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

    def _open_picamera(self, camera_index: int) -> None:
        try:
            from picamera2 import Picamera2
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Picamera2 is not installed. On Raspberry Pi OS, run: sudo apt install -y python3-picamera2"
            ) from exc

        self._picamera = Picamera2(camera_index)
        config = self._picamera.create_video_configuration(
            main={"size": (640, 480), "format": "RGB888"}
        )
        self._picamera.configure(config)
        self._picamera.start()

    @property
    def is_image(self) -> bool:
        return self._is_image

    def read(self) -> Tuple[bool, Optional[Any]]:
        """Return (success, frame). success becomes False when input is done."""

        if self._is_image:
            if self._image_returned:
                return False, None
            self._image_returned = True
            return True, self._image.copy()

        if self._picamera is not None:
            frame_rgb = self._picamera.capture_array()
            if frame_rgb is None:
                return False, None
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            return True, frame_bgr

        if self._capture is None:
            return False, None

        success, frame = self._capture.read()
        if not success or frame is None:
            return False, None
        return True, frame

    def release(self) -> None:
        if self._capture is not None:
            self._capture.release()
        if self._picamera is not None:
            self._picamera.stop()
            self._picamera.close()
            self._picamera = None


def _parse_picamera_index(source: str) -> int:
    """Parse picamera source strings.

    Accepted examples:
        picamera
        picamera0
        picamera1
        picamera:0
        picamera:1
    """

    if source == "picamera":
        return 0

    match = re.fullmatch(r"picamera:?(\d+)", source)
    if not match:
        raise ValueError("Picamera source must be 'picamera', 'picamera0', 'picamera1', or 'picamera:<index>'.")
    return int(match.group(1))
