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
    depth_smoother.py
    detector.py
    postprocess.py
    stereo.py
    stereo_calibration.py
    utils.py
  fusion/
    hazard_decision.py
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

## Raspberry Pi Stereo Depth

The Waveshare dual IMX219 module is treated as two Raspberry Pi cameras. On
Raspberry Pi 5, configure both sensors in `/boot/firmware/config.txt`:

```text
dtoverlay=imx219,cam0
dtoverlay=imx219,cam1
```

After rebooting, confirm that both cameras appear:

```bash
rpicam-hello --list-cameras
```

Capture 25-40 checkerboard pairs at the same 640x480 resolution used at
runtime. `pattern-cols` and `pattern-rows` are the checkerboard's inner corner
counts, not its square counts:

```bash
python scripts/capture_stereo_pairs.py \
  --left-source picamera0 \
  --right-source picamera1 \
  --pairs 30 \
  --pattern-cols 9 \
  --pattern-rows 6
```

Calibrate using the checkerboard's measured square size:

```bash
python scripts/calibrate_stereo.py \
  --image-dir calibration/images \
  --output calibration/stereo_calibration.npz \
  --pattern-cols 9 \
  --pattern-rows 6 \
  --square-size-m 0.024
```

The stereo runtime uses libcamera's server/client synchronization controls,
pairs frames by `SensorTimestamp`, rejects pairs over the skew limit, filters
locally inconsistent disparity pixels, and reports per-object depth
uncertainty.

Export the nano detector to NCNN on a development computer:

```bash
yolo export model=yolov8n.pt format=ncnn imgsz=512
```

For better accuracy without a larger Pi model, fine-tune the nano checkpoint
on chest-mounted Haptos images and export the best checkpoint in one command:

```bash
python scripts/train_detector.py \
  --data datasets/haptos/data.yaml \
  --base-model yolov8n.pt \
  --epochs 80 \
  --export-ncnn \
  --export-imgsz 512
```

Training should run on a laptop or GPU machine, not on the Pi. Include
hallways, sidewalks, people, chairs, curbs, motion blur, low light, and hard
negative images where no obstacle is present. Keep separate train, validation,
and test splits captured on different walks.

Copy the exported model directory to the Pi, then run:

```bash
python main.py \
  --source picamera0 \
  --stereo-depth \
  --stereo-right-source picamera1 \
  --stereo-calibration calibration/stereo_calibration.npz \
  --backend ncnn \
  --model yolov8n_ncnn_model \
  --conf 0.25 \
  --fps 8
```

Important stereo quality controls:

- `--stereo-max-skew-ms 8` rejects left/right images captured too far apart.
- `--depth-min-valid-fraction 0.15` rejects sparse depth inside an object box.
- `--depth-max-relative-uncertainty 0.35` rejects noisy object distances.
- `--depth-smoothing-window 5` median-filters recent depths for matching objects.
- `--hazard-distance-m 2.5` ignores distant objects only when their depth is trustworthy.
- `--emergency-stop-distance-m 0.8` forces `STOP` for a trusted near camera or LiDAR obstacle.

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
- per-detection depth, valid-pixel fraction, robust uncertainty, and fault state
- optional LiDAR summary with fault state, point count, and nearest/median/farthest filtered distance
- stereo frame skew, valid-pixel fraction, and depth summary

## Navigation Logic

The image is split into thirds:

- `LEFT`
- `CENTER`
- `RIGHT`

For now, common classes such as `person`, `bicycle`, `chair`, `backpack`, `car`, and `dog` are treated as obstacles.

The distance-aware command logic is:

- trusted obstacle at or below the emergency distance -> `STOP`
- trusted LiDAR return at or below the emergency distance -> `STOP`
- trusted object beyond the hazard distance -> ignore it for the current command
- uncertain or missing depth -> retain the conservative camera-only behavior
- actionable obstacle in `CENTER` -> `STOP`
- obstacle in `LEFT` only -> `GO_RIGHT`
- obstacle in `RIGHT` only -> `GO_LEFT`
- no obstacles -> `FORWARD`
- obstacles in multiple regions -> `STOP`

The thresholds are engineering defaults for prototype testing, not
safety-certified values. Validate them with measured indoor and outdoor test
courses before relying on haptic output.

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

Use a Raspberry Pi 5 supply capable of the recommended 5V/5A mode and active
cooling. A 3A supply restricts downstream USB peripheral power, which matters
when USB LiDAR hardware is attached. Use a powered USB hub when the sensors'
combined draw exceeds the Pi's peripheral budget.

Run without `--show` on the wearable. Prefer the NCNN nano model and cap
processing with `--fps` while measuring latency, temperature, throttling, and
missed detections. The runtime does not generate a dense 3D camera point cloud;
it computes a depth map and samples only the object regions needed for hazard
decisions.
