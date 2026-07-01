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
```

Press `q` to quit the display window. `Ctrl+C` exits cleanly with code 130.

There are no automated tests yet. Verification is done manually per the README testing plan.

## Architecture

The pipeline is a straight-line data flow across five modules:

```
VideoSource (camera.py)
    → YoloDetector.detect() (detector.py)          # raw Detection list, no region/obstacle info
    → filter_and_enrich_detections() (postprocess.py) # adds region + is_obstacle, re-filters by conf
    → generate_navigation_hint() (postprocess.py)  # collapses enriched detections → one command string
    → FrameResult (haptos_types.py)                # frozen dataclass, serialisable via .to_dict()
    → console print / JsonlLogger / draw_overlay() (utils.py)
```

Key design decisions to preserve:
- **`Detection` is frozen twice.** `YoloDetector` emits `Detection` objects without `region` or `is_obstacle` (both default to `None`/`False`). `filter_and_enrich_detections` replaces each one with a new frozen `Detection` that fills those fields. Never mutate a `Detection` in place.
- **`haptos_types.py` vs `types.py`.** Runtime imports always use `haptos_types` to avoid shadowing Python's stdlib `types` module. `types.py` exists only to satisfy the project spec layout and re-exports from `haptos_types`.
- **`frame_width` is passed explicitly.** Region mapping happens in post-processing, not inside the detector, so the detector stays model-agnostic and unit-testable without a real frame.
- **Navigation logic lives entirely in `generate_navigation_hint`.** The command priority is: CENTER obstacle → STOP, multi-region → STOP, LEFT-only → GO_RIGHT, RIGHT-only → GO_LEFT, else FORWARD.

## Key constants (config.py)

- `DEFAULT_CONFIDENCE = 0.4` — applied twice: once inside `YoloDetector.detect()` (passed to YOLO's `predict`) and once in `filter_and_enrich_detections()`. Both thresholds use the same CLI `--conf` value.
- `OBSTACLE_CLASSES` — set of COCO class name strings. Extend this set to change which detected objects trigger navigation commands without touching any other module.
- `IMAGE_EXTENSIONS` — used by `VideoSource` to distinguish single-image mode (one frame then stop) from video/webcam mode (loop until end-of-stream or `q`).

## Planned integration surface

`FrameResult.to_dict()` is the intended output contract for the sensor fusion layer. Future work will pipe each JSONL row into a layer that combines CV obstacle regions with ultrasonic distance readings to drive haptic motor intensity. Keep the `FrameResult` schema stable; add fields rather than renaming existing ones.

For Raspberry Pi deployment: swap `VideoSource` input config and drop `--show` (no display). The rest of the pipeline is Pi-compatible as written.
