# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

Python 3.10+ required. On Windows use `.venv\Scripts\activate` instead of `source`.

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/Mac
.venv\Scripts\activate           # Windows
pip install -r requirements.txt
```

`yolov8n.pt` is downloaded automatically by Ultralytics on first run if not already present locally.

For Raspberry Pi deployment (NCNN backend, no PyTorch/Ultralytics), use `requirements-pi.txt` instead.

## Running

```bash
# Single image (fastest way to verify a change)
python main.py --source path/to/image.jpg --show

# Webcam live
python main.py --source webcam --show

# Video file, save structured output
python main.py --source path/to/video.mp4 --save-log logs/session.jsonl

# Headless (for FPS measurement or CI-like runs)
python main.py --source path/to/video.mp4

# Stereo depth (two camera sources) + serial LiDAR together
python main.py --source picamera0 --stereo-depth --stereo-right-source picamera1 \
  --lidar-source serial --lidar-port COM5 --show
```

Press `q` to quit the display window. `Ctrl+C` exits cleanly with code 130.

```bash
# Run the test suite (unittest-style tests, no venv needed for most — pure Python logic)
# -t . is required: without it, tests/scripts/ collides with the top-level scripts/ package
python -m unittest discover -s tests -t .
# or, if pytest is installed (no extra flags needed):
pytest tests/
```

## Architecture

The pipeline is a straight-line data flow that merges two independent sensing channels — camera and LiDAR — at the end:

```
VideoSource / StereoVideoSource (cv/camera.py)
    → YoloDetector / NcnnYoloDetector.detect() (cv/detector.py)     # raw Detection list, no region/obstacle/depth info
    → filter_and_enrich_detections() (cv/postprocess.py)            # adds region + is_obstacle, re-filters by conf
    → attach_depth_to_detections() (cv/stereo.py)                   # adds per-object stereo depth, if --stereo-depth
    → DetectionDepthSmoother.update() (cv/depth_smoother.py)        # temporal median smoothing of matched objects
                                                                      ┐
filter_lidar_scan() → LidarFrameBuffer (sensor/*)                    ├─ independent channel, same loop iteration
                                                                      ┘
    → generate_fused_navigation_hint() (fusion/hazard_decision.py)  # combines trusted camera depth + LiDAR range
    → FrameResult (types.py)                                        # frozen dataclass, serialisable via .to_dict()
    → console print / JsonlLogger / draw_overlay() (cv/utils.py)
```

Key design decisions to preserve:
- **`Detection` is frozen and re-created at each stage, never mutated.** `YoloDetector`/`NcnnYoloDetector` emit bare `Detection` objects (no region/obstacle/depth). `filter_and_enrich_detections` fills region + is_obstacle. `attach_depth_to_detections` and `DetectionDepthSmoother` each use `dataclasses.replace` to add depth fields on top. Never mutate a `Detection` in place.
- **`types.py` is the only types module.** There is no `haptos_types.py` split — runtime code imports directly from `haptos.types`.
- **`frame_width` is passed explicitly.** Region mapping happens in post-processing, not inside the detector, so the detector stays model-agnostic and unit-testable without a real frame.
- **Navigation logic is layered, not single-function.** `generate_navigation_hint()` (postprocess.py) is the camera-only region logic: CENTER obstacle → STOP, multi-region → STOP, LEFT-only → GO_RIGHT, RIGHT-only → GO_LEFT, else FORWARD. `generate_fused_navigation_hint()` (fusion/hazard_decision.py) wraps it: a *trusted* near obstacle (camera depth or LiDAR range) at or below `emergency_stop_distance_m` forces STOP immediately; a trusted obstacle beyond `max_obstacle_distance_m` is dropped before falling through to the region logic; untrusted/missing depth is treated conservatively (kept actionable).
- **2D LiDAR only produces a flat horizontal slice today.** `sensor/lidar_filter.py::polar_scan_to_xyz()` hardcodes `y = 0` for every point — there is no vertical/tilt dimension yet. See "Current Status & Open Work" below.

## Key constants (config.py)

- `DEFAULT_CONFIDENCE = 0.4` — applied twice: once inside the detector's `.detect()` (passed to YOLO's `predict`) and once in `filter_and_enrich_detections()`. Both thresholds use the same CLI `--conf` value.
- `OBSTACLE_CLASSES` — set of COCO class name strings. Extend this set to change which detected objects trigger navigation commands without touching any other module.
- `IMAGE_EXTENSIONS` — used by `VideoSource` to distinguish single-image mode (one frame then stop) from video/webcam mode (loop until end-of-stream or `q`).
- `HAZARD_DEFAULT_MAX_DISTANCE_M` / `HAZARD_DEFAULT_EMERGENCY_STOP_DISTANCE_M` — the two fusion thresholds (2.5 m / 0.8 m by default); emergency-stop must never exceed max-distance, enforced in both `main.py` arg validation and `generate_fused_navigation_hint()`.
- `LIDAR_MIN_DISTANCE_M` / `LIDAR_MAX_DISTANCE_M` / `LIDAR_MIN_QUALITY` — stage-two LiDAR point filtering in `sensor/lidar_filter.py`.
- `STEREO_DEFAULT_*` — stereo camera sync tolerance, max trusted depth, min valid-pixel fraction, max relative uncertainty, and smoothing window.

## Current Status & Open Work

**Done:** YOLO detection (Ultralytics + NCNN backends) → LEFT/CENTER/RIGHT region mapping → obstacle flagging; stereo camera disparity/depth attached per-detection with valid-fraction and uncertainty gating, then temporally smoothed; 2D serial-LiDAR ingestion with range/quality filtering; a fusion layer (`fusion/hazard_decision.py`) that lets trusted camera depth or LiDAR range override the region-only command; JSONL logging of the full `FrameResult` schema. This is well ahead of what `docs/roadmap.md`'s Phase 3/4 describe (it assumed a single-point ultrasonic/IR sensor and a simpler fixed-threshold fusion rule) — treat the code, not that doc, as authoritative for how sensing/fusion actually works.

**Not yet started** (see `docs/roadmap.md` Phase 1 remaining and `docs/research.md` for option tradeoffs):
- Path boundary / walkable-surface detection (sidewalk edges).
- Surface hazard detection (curbs, steps).
- Low-light fault detection (spec F5).
- A latency benchmarking script (spec F4, target <300ms end-to-end).
- `haptos/feedback/` (wristband firmware) and `firmware/` (CPU/decision firmware) are empty placeholders — no code yet.

**Active goal — 2D LiDAR → 3D reconstruction:** build a 3D point cloud from the existing 2D LiDAR by physically sweeping/tilting the sensor up and down and accumulating scans taken at each tilt angle over time (a nodding/push-broom style scanner), rather than swapping in true 3D LiDAR hardware. Relevant existing code:
- `sensor/lidar_filter.py::polar_scan_to_xyz()` currently hardcodes `y = 0` for every point — it will need a tilt-angle input (in addition to azimuth) to place each scan's points at the correct height.
- `sensor/lidar_buffer.py::LidarFrameBuffer` already keeps a short history of recent `FilteredLidarFrame`s by timestamp; it may be extendable to accumulate multiple tilt-angle scans into one 3D frame, or a new module may be cleaner.
- No tilt/servo control code exists yet — this needs a hardware interface (set servo angle → read one 2D scan → repeat) before frames can be tagged with elevation and merged into a point cloud.

## Planned integration surface

`FrameResult.to_dict()` is the output contract for anything downstream (wristband firmware, logging, future 3D reconstruction). Keep the schema stable; add fields rather than renaming existing ones.

For Raspberry Pi deployment: use `requirements-pi.txt`, the NCNN detector backend, and drop `--show` (no display). The rest of the pipeline is Pi-compatible as written.
