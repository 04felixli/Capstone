"""Configuration constants for the Haptos CV subsystem."""

DEFAULT_MODEL = "yolov8n.pt"
DEFAULT_CONFIDENCE = 0.4

LEFT_REGION = "LEFT"
CENTER_REGION = "CENTER"
RIGHT_REGION = "RIGHT"

COMMAND_FORWARD = "FORWARD"
COMMAND_STOP = "STOP"
COMMAND_GO_LEFT = "GO_LEFT"
COMMAND_GO_RIGHT = "GO_RIGHT"

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

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
