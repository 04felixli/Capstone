"""Configuration constants for the Haptos CV subsystem."""

DEFAULT_MODEL = "yolov8n.pt"
DEFAULT_CONFIDENCE = 0.4
DEFAULT_DETECTOR_BACKEND = "ultralytics"
DETECTOR_BACKEND_ULTRALYTICS = "ultralytics"
DETECTOR_BACKEND_NCNN = "ncnn"

HAZARD_DEFAULT_EMERGENCY_STOP_DISTANCE_M = 0.8

LEFT_REGION = "LEFT"
CENTER_REGION = "CENTER"
RIGHT_REGION = "RIGHT"

COMMAND_FORWARD = "FORWARD"
COMMAND_STOP = "STOP"
COMMAND_GO_LEFT = "GO_LEFT"
COMMAND_GO_RIGHT = "GO_RIGHT"

LIDAR_SOURCE_NONE = "none"
LIDAR_SOURCE_SERIAL = "serial"

LIDAR_FAULT_NONE = "none"
LIDAR_FAULT_NO_VALID_POINTS = "no_valid_points"

LIDAR_MIN_DISTANCE_M = 0.3
LIDAR_MAX_DISTANCE_M = 2.0
LIDAR_MIN_QUALITY = 1
LIDAR_DEFAULT_BAUDRATE = 115200
LIDAR_DEFAULT_SCAN_TIMEOUT_S = 0.20
LIDAR_DEFAULT_MIN_SAMPLES = 5

# A first-pass list of COCO classes that can matter for wearable navigation.
# This can become configurable as Haptos learns which objects are important.
OBSTACLE_CLASSES = {
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "bench",
    "chair",
    "couch",
    "potted plant",
    "backpack",
    "suitcase",
    "dog",
    "cat",
    "traffic light",
    "stop sign",
    "fire hydrant",
}
