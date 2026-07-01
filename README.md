# Haptos CV

Laptop-testable computer vision subsystem for **Haptos**, an embedded wearable navigation system.

This module reads frames from a webcam, video file, or image file, runs a lightweight Ultralytics YOLO model, maps detections into `LEFT`, `CENTER`, and `RIGHT` image regions, and can read serial LiDAR scans for distance sensing.

No GPIO-specific code is included. LiDAR integration uses a serial port so the sensor driver can run on the host computer, microcontroller, or embedded platform.

## Project Layout

```text
haptos-cv/
README.md
requirements.txt
main.py
haptos/
  config.py
  types.py
  cv/
    camera.py
    detector.py
    postprocess.py
    utils.py
  sensor/
    lidar_buffer.py
    lidar_filter.py
    lidar_reader.py
tests/
  sensor/
    test_lidar.py
```

Runtime imports use the `haptos` package layout so project modules do not
collide with Python standard-library modules.

## Setup

Use Python 3.10 or newer if possible.

```bash
cd Capstone
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

Run with a serial-connected LiDAR:

```bash
python -m serial.tools.list_ports
python main.py --source webcam --lidar-source serial --lidar-port COM5 --lidar-baudrate 115200
```

The serial LiDAR reader expects one 2D sample per line:

```text
angle_deg,distance_mm,quality
```

The quality field is optional. These are also accepted:

```text
12.5,840,15
12.5 840 15
12.5;840
```

A line containing `SCAN`, `START`, or `END` marks a scan boundary. This format is intended for a LiDAR vendor SDK or microcontroller layer that converts the sensor's native protocol into simple serial samples.

Run LiDAR-only unit tests:

```bash
python -m unittest tests.sensor.test_lidar
```

## Output

Console output is intentionally concise:

```text
Frame 120 | command=STOP | detections=person:center:0.91
```

With serial LiDAR enabled, console rows also include a filtered LiDAR summary:

```text
Frame 120 | command=STOP | detections=person:center:0.91 | lidar=none:points=33:nearest=1.17m
```

Each logged JSONL row contains:

- frame index
- navigation command
- FPS estimate
- detections with class name, confidence, bounding box, region, and obstacle flag
- optional LiDAR summary with fault state, point count, and nearest/median/farthest filtered distance

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

5. Test LiDAR serial input.
   Confirm that the LiDAR driver outputs angle/distance samples, filtered point counts are nonzero for nearby objects, and nearest distance changes when obstacles move.

## Raspberry Pi Notes

The code avoids laptop-only assumptions beyond OpenCV camera access. For a Raspberry Pi migration, keep the same module boundaries and swap only the input or model configuration if needed. Smaller models, lower input resolution, and no display window will usually improve Pi performance.
