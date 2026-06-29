# Haptos CV

Laptop-testable computer vision subsystem for **Haptos**, an embedded wearable navigation system.

This module reads frames from a webcam, video file, or image file, runs a lightweight Ultralytics YOLO model, maps detections into `LEFT`, `CENTER`, and `RIGHT` image regions, and emits structured frame results for later sensor fusion and haptic feedback.

No GPIO or hardware-specific code is included in this version.

## Project Layout

```text
haptos-cv/
README.md
requirements.txt
main.py
config.py
camera.py
detector.py
postprocess.py
types.py
utils.py
haptos_types.py
```

`types.py` is kept for the requested layout. Runtime imports use
`haptos_types.py` to avoid colliding with Python's standard-library `types`
module.

## Setup

Use Python 3.10 or newer if possible.

```bash
cd haptos-cv
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Ultralytics will download `yolov8n.pt` the first time it is used if the model file is not already present.

## Run

Single image:

```bash
python main.py --source path/to/test-image.jpg --show
```

Webcam:

```bash
python main.py --source webcam --show
```

Video file:

```bash
python main.py --source path/to/walking-video.mp4 --show
```

Save newline-delimited JSON logs:

```bash
python main.py --source webcam --save-log logs/session.jsonl
```

Use a different model or confidence threshold:

```bash
python main.py --source webcam --model yolov8n.pt --conf 0.5 --show
```

## Output

Console output is intentionally concise:

```text
Frame 120 | command=STOP | detections=person:center:0.91
```

Each logged JSONL row contains:

- frame index
- navigation command
- FPS estimate
- detections with class name, confidence, bounding box, region, and obstacle flag

## Navigation Logic

The image is split into thirds:

- `LEFT`
- `CENTER`
- `RIGHT`

For now, common classes such as `person`, `bicycle`, `chair`, `backpack`, `car`, and `dog` are treated as obstacles.

The first-pass command logic is:

- obstacle in `CENTER` -> `STOP`
- obstacle in `LEFT` only -> `GO_RIGHT`
- obstacle in `RIGHT` only -> `GO_LEFT`
- no obstacles -> `FORWARD`
- obstacles in multiple regions -> `STOP`

This is deliberately simple so later work can combine CV output with ultrasonic sensors, IMU data, and haptic feedback policies.

## Testing Plan

1. Test with a single image first.
   Confirm that YOLO loads, detections appear, regions are correct, and the command makes sense.

2. Test with the webcam.
   Walk objects through the left, center, and right portions of the frame and check command changes.

3. Test with recorded walking videos.
   Use hallway, sidewalk, and indoor clutter videos to compare detections against expected obstacles.

4. Measure rough latency and FPS.
   Run with `--show` for visual debugging, then without `--show` for a cleaner FPS estimate. Review printed FPS and JSONL logs.

5. Plan later integration.
   Feed each `FrameResult` into a sensor fusion layer. That layer can compare camera obstacle regions with ultrasonic distance readings, then choose vibration motor intensity and side-specific haptic cues.

## Raspberry Pi Notes

The code avoids laptop-only assumptions beyond OpenCV camera access. For a Raspberry Pi migration, keep the same module boundaries and swap only the input or model configuration if needed. Smaller models, lower input resolution, and no display window will usually improve Pi performance.
