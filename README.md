# Haptos CV

Computer vision and LiDAR subsystem for **Haptos**, an embedded wearable navigation system running on a Raspberry Pi 5.

The binocular camera feeds a lightweight Ultralytics YOLO model that maps detections into `LEFT`, `CENTER`, and `RIGHT` image regions. Only one image stream is used for this; the camera produces no depth. A 2D LiDAR (LDROBOT STL-19P, sold as Dxtvate D500) is the sole source of distance.

The Pi drives every sensor and output directly: the LiDAR over USB-serial, the camera, and planned tilt servo, haptic driver, and speaker. There is no separate wristband board.

## Project Layout

```text
haptos-cv/
README.md
CLAUDE.md
requirements.txt
requirements-pi.txt
main.py
docs/
  research.md
  roadmap.md
firmware/           # placeholder, no code yet
scripts/
  lidar_raw_test.py # standalone LiDAR packet check, no haptos imports
  train_detector.py
haptos/
  config.py
  types.py
  cv/
    camera.py
    detector.py
    geometry.py
    postprocess.py
    utils.py
  fusion/
    hazard_decision.py
  sensor/
    ld19_reader.py    # binary LD19/STL-19P reader (not yet wired into main.py)
    lidar_buffer.py
    lidar_filter.py
    lidar_reader.py   # legacy text-format reader
  feedback/           # placeholder for planned Pi-side haptic/audio output
tests/
  cv/
  fusion/
  sensor/
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

`requirements.txt` installs `ultralytics`, needed by the default `--backend ultralytics` detector. It will download `yolov8n.pt` the first time it is used if the model file is not already present. If `ultralytics` isn't installed, running `main.py` fails immediately with a message telling you to install it — pass `--backend ncnn` instead if you don't need it (see Raspberry Pi Deployment below).

## Run

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

## Raspberry Pi Deployment

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
  --lidar-source serial \
  --lidar-port /dev/lidar \
  --backend ncnn \
  --model yolov8n_ncnn_model \
  --conf 0.25 \
  --fps 8
```

`--lidar-source serial` currently uses the legacy text-format reader, so it will not
work with the STL-19P's binary output. The LD19 reader in `haptos/sensor/ld19_reader.py`
is written but not yet wired into `create_lidar_reader()`; until it is, check the
sensor directly with the standalone script below.

`--emergency-stop-distance-m 0.8` forces `STOP` for a trusted near LiDAR
return, regardless of what the camera sees.

### Checking the LiDAR on the Pi

The STL-19P is a continuously spinning sensor, so each printed `angle` is the
direction of one laser sample at that instant, not a fixed orientation. Expect the
angle to cycle through 0–360° repeatedly. Use `--print-interval` to slow the output
down for reading; serial is still read continuously underneath.

```bash
python3 scripts/lidar_raw_test.py --port /dev/lidar --print-interval 0.2
```

`/dev/lidar` is a udev symlink to the CP2102 USB-serial adapter (see the
`99-lidar.rules` setup notes in `docs/roadmap.md`). Zero-distance points with
intensity 0 are invalid returns, not objects at zero range.

### Legacy text-format LiDAR input

The text reader expects one 2D sample per line:

```text
angle_deg,distance_mm,quality
```

The quality field is optional. These are also accepted:

```text
12.5,840,15
12.5 840 15
12.5;840
```

A line containing `SCAN`, `START`, or `END` marks a scan boundary. This format
is for a vendor driver or microcontroller that converts the sensor's native
protocol; it is not used by the STL-19P directly.

Run LiDAR unit tests:

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

The command logic is:

- trusted LiDAR return at or below the emergency distance -> `STOP`
- obstacle in `CENTER` -> `STOP`
- obstacle in `LEFT` only -> `GO_RIGHT`
- obstacle in `RIGHT` only -> `GO_LEFT`
- no obstacles -> `FORWARD`
- obstacles in multiple regions -> `STOP`

The threshold is an engineering default for prototype testing, not a
safety-certified value. Validate it with measured indoor and outdoor test
courses before relying on haptic output.

## Testing Plan

1. Test with the webcam first.
   Confirm that YOLO loads, detections appear, regions are correct, and the command makes sense. Walk objects through the left, center, and right portions of the frame and check command changes.

2. Test with recorded walking videos.
   Use hallway, sidewalk, and indoor clutter videos to compare detections against expected obstacles.

3. Measure rough latency and FPS.
   Run with `--show` for visual debugging, then without `--show` for a cleaner FPS estimate. Review printed FPS and JSONL logs.

4. Test LiDAR input.
   Confirm with `scripts/lidar_raw_test.py` that the STL-19P produces readings, then confirm that filtered point counts are nonzero for nearby objects and nearest distance changes when obstacles move.

## Raspberry Pi Notes

Use a Raspberry Pi 5 supply capable of the recommended 5V/5A mode and active
cooling. A 3A supply restricts downstream USB peripheral power, which matters
when USB LiDAR hardware is attached. Use a powered USB hub when the sensors'
combined draw exceeds the Pi's peripheral budget.

Run without `--show` on the wearable. Prefer the NCNN nano model and cap
processing with `--fps` while measuring latency, temperature, throttling, and
missed detections.

`requirements-pi.txt` currently lists `numpy`, `pyserial`, and `ncnn`. `opencv-python`
(imported by `haptos/cv/camera.py`) and `picamera2` are not listed and need to be
addressed before a clean install.
