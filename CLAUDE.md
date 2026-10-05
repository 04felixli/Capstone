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

`requirements.txt` installs `ultralytics`, which the default `--backend ultralytics` detector imports at construction time — `main.py` no longer wraps this import in its own try/except, so a missing `ultralytics` install surfaces as `haptos/cv/detector.py`'s own `RuntimeError` ("Ultralytics is not installed. Run: pip install -r requirements.txt"), not a raw `ModuleNotFoundError`. `yolov8n.pt` is downloaded automatically by Ultralytics on first run if not already present locally.

For Raspberry Pi deployment (NCNN backend, no PyTorch/Ultralytics), use `requirements-pi.txt` instead — pass `--backend ncnn` to avoid importing `ultralytics` at all.

## Running

```bash
# Webcam live (fastest way to verify a change)
python main.py --source webcam --show

# Video file, save structured output
python main.py --source path/to/video.mp4 --save-log logs/session.jsonl

# Headless (for FPS measurement or CI-like runs)
python main.py --source path/to/video.mp4

# Single camera + serial LiDAR together
python main.py --source picamera0 --lidar-source serial --lidar-port COM5 --show
```

Press `q` to quit the display window. `Ctrl+C` exits cleanly with code 130.

```bash
# Run the test suite (unittest-style tests, no venv needed for most — pure Python logic)
python -m unittest discover -s tests -t .
# or, if pytest is installed (no extra flags needed):
pytest tests/
```

## Architecture

**Target architecture: mono camera for classification/region only + LiDAR as the sole source of metric range.** Stereo depth (dual cameras, disparity/depth estimation, per-object depth smoothing) was removed from this codebase — see "Current Status & Open Work" below for why. The camera is binocular hardware, but only feeds region classification: it uses one image stream, produces no depth, and camera detections carry a `region` (LEFT/CENTER/RIGHT) and `is_obstacle` flag but no distance. LiDAR range is the only depth signal; tilting the LiDAR is how it becomes 3D.

The pipeline is a straight-line data flow that merges two independent sensing channels — camera and LiDAR — at the end:

```
VideoSource (cv/camera.py)
    → YoloDetector / NcnnYoloDetector.detect() (cv/detector.py)     # raw Detection list, no region/obstacle info
    → filter_and_enrich_detections() (cv/postprocess.py)            # adds region + is_obstacle, re-filters by conf
                                                                      ┐
filter_lidar_scan() → LidarFrameBuffer (sensor/*)                    ├─ independent channel, same loop iteration
                                                                      ┘
    → generate_fused_navigation_hint() (fusion/hazard_decision.py)  # LiDAR near-range override, else camera region logic
    → FrameResult (types.py)                                        # frozen dataclass, serialisable via .to_dict()
    → console print / JsonlLogger / draw_overlay() (cv/utils.py)
```

Key design decisions to preserve:
- **`Detection` is frozen and re-created at each stage, never mutated.** `YoloDetector`/`NcnnYoloDetector` emit bare `Detection` objects (no region/obstacle). `filter_and_enrich_detections` is the only stage that fills in region + is_obstacle, via `dataclasses.replace`-style reconstruction. Never mutate a `Detection` in place.
- **`types.py` is the only types module.** There is no `haptos_types.py` split — runtime code imports directly from `haptos.types`.
- **`frame_width` is passed explicitly.** Region mapping happens in post-processing, not inside the detector, so the detector stays model-agnostic and unit-testable without a real frame.
- **Navigation logic is layered, not single-function.** `generate_navigation_hint()` (postprocess.py) is the camera-only region logic: CENTER obstacle → STOP, multi-region → STOP, LEFT-only → GO_RIGHT, RIGHT-only → GO_LEFT, else FORWARD. `generate_fused_navigation_hint()` (fusion/hazard_decision.py) wraps it: a trusted LiDAR return at or below `emergency_stop_distance_m` forces STOP immediately; otherwise it falls through to the region logic unchanged.
- **2D LiDAR only produces a flat horizontal slice today, and the fusion layer only reads its single nearest-distance scalar.** `sensor/lidar_filter.py::polar_scan_to_xyz()` hardcodes `y = 0` for every point (no vertical/tilt dimension), and `generate_fused_navigation_hint()` only ever reads `LidarFrameSummary.nearest_distance_m` — the closest point across the *entire* scan, angle discarded — even though `FilteredLidarFrame` already retains full per-point `angles_rad`/`points_xyz`. At the fusion-decision layer this makes the 2D LiDAR behave like a single-point rangefinder today. See "Current Status & Open Work" below.

## Key constants (config.py)

- `DEFAULT_CONFIDENCE = 0.4` — applied twice: once inside the detector's `.detect()` (passed to YOLO's `predict`) and once in `filter_and_enrich_detections()`. Both thresholds use the same CLI `--conf` value.
- `OBSTACLE_CLASSES` — set of COCO class name strings. Extend this set to change which detected objects trigger navigation commands without touching any other module.
- `HAZARD_DEFAULT_EMERGENCY_STOP_DISTANCE_M` — the one fusion threshold (0.8 m by default); any trusted LiDAR return at or below it forces STOP.
- `LIDAR_MIN_DISTANCE_M` / `LIDAR_MAX_DISTANCE_M` / `LIDAR_MIN_QUALITY` — stage-two LiDAR point filtering in `sensor/lidar_filter.py`.

## Current Status & Open Work

**Done:** YOLO detection (Ultralytics + NCNN backends) → LEFT/CENTER/RIGHT region mapping → obstacle flagging; 2D LiDAR ingestion with range/quality filtering. The text-format reader (`sensor/lidar_reader.py`, `angle,distance,quality` lines) is legacy; the binary LDROBOT LD19 / STL-19P reader (`sensor/ld19_reader.py`) is written and returns the same `RawLidarScan`, but is not yet wired into `create_lidar_reader()` or covered by unit tests. A fusion layer (`fusion/hazard_decision.py`) that lets a trusted near LiDAR return override the region-only command; JSONL logging of the `FrameResult` schema. This is well ahead of what `docs/roadmap.md`'s Phase 3/4 describe (it assumed a single-point ultrasonic/IR sensor) — treat the code, not that doc, as authoritative for how sensing/fusion actually works.

**Removed:** stereo camera depth (dual-camera capture/sync, `StereoDepthEstimator` disparity/depth estimation, per-detection depth smoothing, stereo calibration scripts) was implemented and working, but has been deliberately removed. The target architecture is mono camera (classification/region only) + a tilt-swept 2D LiDAR (all metric ranging, see "Active goal" below) — LiDAR is the more trustworthy range source (doesn't degrade on textureless/low-light surfaces, doesn't burn CPU on disparity matching each frame), so stereo's dual-camera sync/calibration complexity is no longer needed. If this is ever revisited, the removed code is in git history on this branch prior to the removal commit.

**Hardware:** a Raspberry Pi 5 is the only computer. It drives the STL-19P LiDAR (USB-serial, `/dev/lidar` via a udev symlink), the binocular camera, the tilt servo (planned), the haptic driver (DRV2605L over I2C, planned), and the speaker (planned). There is no separate wristband board. Audio output (TTS vs tones) is undecided.

**Not yet started** (see `docs/roadmap.md` Phase 1 remaining and `docs/research.md` for option tradeoffs):
- Path boundary / walkable-surface detection (sidewalk edges).
- Surface hazard detection (curbs, steps).
- Low-light fault detection (spec F5).
- A latency benchmarking script (spec F4, target <300ms end-to-end).
- Wiring the LD19 reader into the pipeline, tilt servo control, haptic driver code, and speaker output.
- `haptos/feedback/` (planned Pi-side haptic/audio output) and `firmware/` are empty placeholders — no code yet.

`requirements-pi.txt` does not list `opencv-python` (which `haptos/cv/camera.py` imports) or `picamera2` (also imported by `camera.py`); both need to be addressed for a clean Pi install.

**Active goal — 2D LiDAR → 3D reconstruction:** build a 3D point cloud from the existing 2D LiDAR by physically tilting the sensor up and down (a nodding/push-broom style scanner) and accumulating scans taken at each tilt angle over time, rather than swapping in true 3D LiDAR hardware. The 2D LiDAR already sweeps azimuth internally (each scan already carries multiple `angle_deg` samples); the new motor axis is elevation/tilt, not a second azimuth sweep. Suggested phasing:
- **Phase 0 (no new hardware):** the fusion layer currently only reads `LidarFrameSummary.nearest_distance_m` — nearest point across the *whole* scan, angle discarded — even though `FilteredLidarFrame.points_xyz`/`.angles_rad` already retain every point. Add an angle-aware occupancy check (e.g. nearest distance within a walking-corridor arc) before building on top of this in 3D.
- **Phase 1:** tilt hardware interface (`set_tilt_angle()`/`get_tilt_angle()`) and extend `sensor/lidar_filter.py::polar_scan_to_xyz()` — currently hardcodes `y = 0` — with a `tilt_deg` parameter so each scan lands at the correct height.
- **Phase 2:** accumulate tilt-tagged scans into a 3D point cloud (rolling/aging buffer rather than wait-for-full-sweep, since a tilt sweep is much slower than the camera loop), with ground-plane removal and a 3D corridor-occupancy check generalizing Phase 0.
- **Phase 3:** feed 3D corridor occupancy into `generate_fused_navigation_hint()` as the LiDAR-side trust signal.
- **Phase 4 (stretch):** LiDAR↔camera extrinsic calibration + point-to-bbox projection for tight fusion (knowing *which* detected object a LiDAR return corresponds to).

`sensor/lidar_buffer.py::LidarFrameBuffer` already keeps a short history of recent `FilteredLidarFrame`s by timestamp; it may be extendable for Phase 2's accumulation, or a new module may be cleaner.

## Planned integration surface

`FrameResult.to_dict()` is the output contract for anything downstream (Pi-side haptic and audio output, logging, future 3D reconstruction). Keep the schema stable; add fields rather than renaming existing ones.

For Raspberry Pi deployment: use `requirements-pi.txt`, the NCNN detector backend, and drop `--show` (no display). The rest of the pipeline is Pi-compatible as written.
