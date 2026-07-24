"""Camera, video, and image input helpers."""

from dataclasses import dataclass
from pathlib import Path
import re
import time
from typing import Any, Optional, Tuple

import cv2

from haptos.config import IMAGE_EXTENSIONS


@dataclass(frozen=True)
class TimedFrame:
    """One image plus the sensor or host timestamp associated with it."""

    image: Any
    timestamp_ns: int


@dataclass(frozen=True)
class StereoFramePair:
    """Left/right frames paired by capture timestamp."""

    left: Any
    right: Any
    timestamp_ns: int
    skew_ms: float
    within_tolerance: bool


class VideoSource:
    """Read frames from a webcam, video file, or single image.

    OpenCV returns images as NumPy arrays in BGR color order. The detector and
    drawing utilities can use that format directly.
    """

    def __init__(
        self,
        source: str,
        *,
        picamera_sync_role: Optional[str] = None,
        camera_fps: float = 30.0,
        auto_start: bool = True,
    ):
        self.source = source
        self.picamera_sync_role = picamera_sync_role
        self.camera_fps = camera_fps
        self.auto_start = auto_start
        self._capture: Optional[cv2.VideoCapture] = None
        self._picamera = None
        self._picamera_started = False
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
        controls_config = {"FrameRate": self.camera_fps}
        if self.picamera_sync_role is not None:
            try:
                from libcamera import controls
            except ModuleNotFoundError as exc:
                raise RuntimeError("libcamera Python controls are required for stereo camera synchronization.") from exc

            sync_modes = {
                "server": controls.rpi.SyncModeEnum.Server,
                "client": controls.rpi.SyncModeEnum.Client,
            }
            try:
                controls_config["SyncMode"] = sync_modes[self.picamera_sync_role]
            except KeyError as exc:
                raise ValueError("picamera_sync_role must be 'server', 'client', or None") from exc

        config = self._picamera.create_video_configuration(
            main={"size": (640, 480), "format": "RGB888"},
            controls=controls_config,
            buffer_count=4,
            queue=False,
        )
        self._picamera.configure(config)
        if self.auto_start:
            self.start()

    def start(self) -> None:
        if self._picamera is not None and not self._picamera_started:
            self._picamera.start()
            self._picamera_started = True

    @property
    def is_image(self) -> bool:
        return self._is_image

    def read(self) -> Tuple[bool, Optional[Any]]:
        """Return (success, frame). success becomes False when input is done."""

        success, timed_frame = self.read_timed()
        return success, None if timed_frame is None else timed_frame.image

    def read_timed(self) -> Tuple[bool, Optional[TimedFrame]]:
        """Return a frame with a matching capture timestamp."""

        if self._is_image:
            if self._image_returned:
                return False, None
            self._image_returned = True
            return True, TimedFrame(self._image.copy(), time.monotonic_ns())

        if self._picamera is not None:
            if not self._picamera_started:
                self.start()
            request = self._picamera.capture_request()
            try:
                frame_rgb = request.make_array("main")
                metadata = request.get_metadata()
                timestamp_ns = int(metadata.get("SensorTimestamp", time.monotonic_ns()))
            finally:
                request.release()
            if frame_rgb is None:
                return False, None
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            return True, TimedFrame(frame_bgr, timestamp_ns)

        if self._capture is None:
            return False, None

        success, frame = self._capture.read()
        if not success or frame is None:
            return False, None
        return True, TimedFrame(frame, time.monotonic_ns())

    def release(self) -> None:
        if self._capture is not None:
            self._capture.release()
        if self._picamera is not None:
            if self._picamera_started:
                self._picamera.stop()
            self._picamera.close()
            self._picamera = None
            self._picamera_started = False


class StereoVideoSource:
    """Capture timestamp-matched stereo pairs from Pi or generic sources."""

    def __init__(
        self,
        left_source: str,
        right_source: str,
        *,
        camera_fps: float = 30.0,
        max_skew_ms: float = 8.0,
        max_pair_attempts: int = 4,
    ):
        if camera_fps <= 0:
            raise ValueError("camera_fps must be positive")
        if max_skew_ms < 0:
            raise ValueError("max_skew_ms must be 0 or greater")
        if max_pair_attempts <= 0:
            raise ValueError("max_pair_attempts must be positive")

        self.max_skew_ms = max_skew_ms
        self.max_pair_attempts = max_pair_attempts
        both_picamera = _is_picamera_source(left_source) and _is_picamera_source(right_source)

        if both_picamera:
            # Start the client before the server, as required by libcamera's
            # multi-camera software synchronization protocol.
            self.right_source = VideoSource(
                right_source,
                picamera_sync_role="client",
                camera_fps=camera_fps,
                auto_start=False,
            )
            self.left_source = VideoSource(
                left_source,
                picamera_sync_role="server",
                camera_fps=camera_fps,
                auto_start=False,
            )
            self.right_source.start()
            self.left_source.start()
        else:
            self.left_source = VideoSource(left_source)
            self.right_source = VideoSource(right_source)

    @property
    def is_image(self) -> bool:
        return self.left_source.is_image and self.right_source.is_image

    def read(self) -> Tuple[bool, Optional[StereoFramePair]]:
        left_ok, left = self.left_source.read_timed()
        right_ok, right = self.right_source.read_timed()
        if not left_ok or left is None or not right_ok or right is None:
            return False, None

        for _ in range(self.max_pair_attempts - 1):
            skew_ms = abs(left.timestamp_ns - right.timestamp_ns) / 1_000_000.0
            if skew_ms <= self.max_skew_ms:
                break

            if left.timestamp_ns < right.timestamp_ns:
                left_ok, replacement = self.left_source.read_timed()
                if not left_ok or replacement is None:
                    break
                left = replacement
            else:
                right_ok, replacement = self.right_source.read_timed()
                if not right_ok or replacement is None:
                    break
                right = replacement

        skew_ms = abs(left.timestamp_ns - right.timestamp_ns) / 1_000_000.0
        return True, StereoFramePair(
            left=left.image,
            right=right.image,
            timestamp_ns=max(left.timestamp_ns, right.timestamp_ns),
            skew_ms=skew_ms,
            within_tolerance=skew_ms <= self.max_skew_ms,
        )

    def release(self) -> None:
        self.left_source.release()
        self.right_source.release()


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


def _is_picamera_source(source: str) -> bool:
    return source.lower().startswith("picamera")
