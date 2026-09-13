# Research Topics

Organized by which roadmap phase they unblock. Each topic has a concrete decision that needs to come out of it.

---

## CV Module (Phase 1–2)

### Path Boundary Detection
The current pipeline detects discrete objects but not the walkable surface boundary. The spec requires detecting "path boundaries" — sidewalk edges, grass-to-pavement transitions.

Options to evaluate:
- **Semantic segmentation** (e.g., DeepLabV3+, SegFormer-B0) — labels each pixel as road/sidewalk/obstacle. Accurate but slower than detection models. Check if a TFLite or NCNN export fits Pi latency budget.
- **Lane/edge-line detection** (e.g., LaneATT, UFLDv2) — originally for driving but applicable to sidewalk edge detection. Much faster than full segmentation.
- **Depth estimation** (e.g., MiDaS, Depth Anything v2 Small) — infer relative depth from monocular camera; walkable surface appears as a consistent flat plane. Useful for step/curb detection.

**Decision needed:** Pick one approach (or a hybrid) that runs within the latency budget on Pi. Benchmark each candidate before committing.

### Surface Hazard Detection (curbs, steps, potholes)
Object detection models trained on COCO don't include these classes. Options:

- **Fine-tune YOLOv8n** on a small custom dataset of curbs/steps. Needs labelled data — look at open datasets: EgoPath, Mapillary Vistas (has sidewalk/curb labels), or collect your own.
- **Depth + slope heuristic** — if using depth estimation above, large depth discontinuities at the base of the frame indicate a step or curb.
- **Floor-plane homography** — project the ground plane and flag deviations. Works well for indoor steps, less robust outdoors.

**Decision needed:** Whether to label a small custom dataset or derive hazards from depth cues. Custom labelling is more accurate; depth inference is more generalizable.

### Model Export and Pi Optimization
YOLOv8n in PyTorch is not the fastest path on Pi. Research the export pipeline:

- `model.export(format='ncnn')` — NCNN runs well on ARM without GPU. Compare latency vs PyTorch.
- `model.export(format='tflite', int8=True)` — quantized TFLite for Pi. Needs a calibration dataset.
- `model.export(format='onnx')` then run with ONNXRuntime — good portability.
- Pi 5 has more CPU headroom than Pi 4; confirm which hardware you're targeting before benchmarking.

**Decision needed:** Which export format to use as the production model on Pi.

### Low-Light / Fault Detection
Spec F5 requires detecting "dim lighting" as a fault condition.

- Compute mean luminance of the frame (convert to grayscale, take mean pixel value). If below threshold, emit a fault.
- Alternatively, use YOLO confidence distribution: if average confidence of all detections drops sharply, lighting may be a factor.
- Research threshold: 100 lux is the lower bound of the spec (F2). Calibrate threshold using reference images shot at known lux values.

**Decision needed:** Luminance threshold value and whether frame-level or rolling-average check is more reliable.

---

## 2D LiDAR Sensing (Phase 3 — largely resolved)

The single-point ultrasonic/IR sensor plan this section used to describe (HC-SR04 vs VL53L1X, multi-sensor array layout) was superseded: a 2D serial LiDAR was selected and implemented instead (`sensor/lidar_reader.py`, `sensor/lidar_filter.py`), since it gives per-angle range data across the whole scan rather than one scalar per sensor, at the same 0.3–2.0 m range spec F1 requires. Treat the code as authoritative here; the open research has moved on to reconstructing 3D structure from that 2D scan (below).

## 2D LiDAR → 3D Reconstruction (Phase 4)

**Active goal.** The 2D LiDAR sweeps azimuth internally already; the plan is to add elevation by physically tilting the sensor up and down (a nodding/push-broom scanner) and accumulating scans by tilt angle over time, rather than buying true 3D LiDAR hardware.

### Tilt Hardware Selection
- Servo vs stepper for the tilt axis — a stepper gives repeatable absolute angle without feedback wiring; a hobby servo is simpler to drive but less precise at the edges of its range.
- Sweep range and speed: how many degrees of tilt actually matter for a walking corridor (ground-to-head height at ~1–2 m), and how slow can the sweep be before it lags obstacle detection unacceptably.
- Where the tilt interface lives: `set_tilt_angle()`/`get_tilt_angle()` need to report actual (not just commanded) angle if the motor can stall or lag.

**Decision needed:** Servo vs stepper, and the tilt sweep range/rate to target for Phase 1.

### Point Cloud Accumulation Strategy
A full tilt sweep is much slower than the camera loop, so the point cloud has to be built incrementally rather than waiting for a complete sweep each time.

- **Rolling/aging buffer** — keep the last N tilt-tagged scans, drop points older than some age. Simple, matches how `LidarFrameBuffer` already works for 2D frames.
- **Fixed-grid accumulation** — bin points into a 3D occupancy grid and decay/refresh cells over time. More memory, cheaper to query for corridor occupancy.

**Decision needed:** Extend `LidarFrameBuffer` for tilt-tagged 3D frames, or build a separate accumulation module — and rolling buffer vs grid.

### Ground-Plane Removal
Points from the ground itself shouldn't count as obstacles.

- **Height threshold** — drop points below some fixed height once the sensor's mounting height and tilt geometry are known. Simple, works if the ground is flat.
- **Plane fitting (RANSAC)** — fit a ground plane per accumulation window and remove inliers. More robust on sloped or uneven ground, more compute.

**Decision needed:** Fixed height threshold is the right starting point given a fixed chest-mount height; revisit plane fitting only if outdoor slopes cause false positives.

### Corridor Occupancy Check
Phase 0 (no new hardware) needs an angle-aware occupancy check before any 3D work: restrict `nearest_distance_m`-style logic to points whose angle falls within a walking-corridor arc, instead of the nearest point across the entire 2D scan. Phase 2 generalizes this to 3D once points carry a tilt-derived height.

**Decision needed:** The corridor arc width (in `angles_rad`) that best matches the camera's CENTER region, so the two channels agree on what "ahead" means.

### LiDAR↔Camera Extrinsic Calibration (Phase 4 stretch)
Only needed once 3D corridor occupancy is feeding fusion and there's a reason to know *which* detected object a LiDAR return corresponds to (point-to-bbox projection).

- Requires known camera intrinsics (from Ultralytics/OpenCV) plus a measured or checkerboard-based rigid transform between the camera and LiDAR mounting points.

**Decision needed:** Defer until Phases 0–3 are working; not a blocker for the current roadmap.

---

## Sensor Fusion (Phase 4)

### Fusion Strategy
`fusion/hazard_decision.py` already implements the rule-based approach this section used to propose as a research question: a trusted LiDAR return at or below `emergency_stop_distance_m` (0.8 m by default, `HAZARD_DEFAULT_EMERGENCY_STOP_DISTANCE_M` in `config.py`) forces STOP; otherwise the camera's region-only logic (`generate_navigation_hint()`) decides. Remaining open question is whether this stays sufficient once LiDAR carries 3D corridor occupancy instead of a single nearest-distance scalar:

- **Confidence-weighted voting** — weight each source by its confidence score. More flexible but requires calibrated confidence values.
- **Kalman filter** — model obstacle position as a state; fuse CV detections and LiDAR readings as noisy measurements. Overkill for MVP but useful if smooth directional estimates over time become necessary.

**Decision needed:** Whether the flat threshold still holds once `generate_fused_navigation_hint()` consumes corridor occupancy (Phase 4 above) instead of `nearest_distance_m`, or whether it needs to become angle/region-aware too.

### Inter-Subsystem Communication Protocol
How does the CV module (Pi) talk to the wristband firmware?

- **Wired UART** — simple, reliable, low latency, no pairing required. Good choice if both devices are on the same physical assembly.
- **BLE (Bluetooth Low Energy)** — wireless, but adds pairing complexity and ~10–50 ms latency overhead. Necessary if the wristband is genuinely separate from the CPU.
- **I2C / SPI** — only practical if wristband MCU is on the same board or very short cable.

**Decision needed:** Wired vs wireless between CPU and wristband. Pick wired UART for the prototype to eliminate a variable; revisit for final product.

---

## Haptic Feedback (Phase 5)

### Vibration Pattern Design
Research shows users can reliably distinguish 4–6 distinct haptic patterns when the differences are in duration and rhythm rather than intensity alone.

- Look at prior work: "Tacton" pattern design principles (Brown et al.) — rhythm and envelope matter more than intensity.
- Recommended pattern set to test: single short pulse (FORWARD/clear), double pulse (GO_LEFT), double pulse offset (GO_RIGHT), continuous (STOP), long single (fault/warning).
- Test distinguishability with 5–10 people without looking at the wristband.

**Decision needed:** Final pattern set. Run recognition trials before finalizing firmware (otherwise you'll flash firmware multiple times).

### Audio Cue Approach
Spec mentions audio as a parallel feedback channel.

- Earpiece vs. bone conduction — bone conduction keeps the user's ears open to ambient sound, which is safer for navigation.
- Simple tones (beeps with distinct pitch/rhythm) vs. speech ("turn left") — speech is intuitive but requires TTS compute; tones are fast and offline.
- Audio as primary or backup — given the spec's language ("haptic and audio"), plan for audio as the redundant channel if vibration patterns are ambiguous.

**Decision needed:** Tone-based vs TTS, and earpiece vs bone conduction.

---

## Hardware Platform (cross-cutting)

### Raspberry Pi Model
- Pi 4 (4GB) is well-supported by Ultralytics and OpenCV. Pi 5 is faster but newer — check Ultralytics compatibility before committing.
- Pi Zero 2W is too slow for real-time YOLOv8 inference.
- Pi Camera Module 3 (12 MP, autofocus) or Camera Module 3 Wide (120° FoV) — wider FoV captures more lateral context for GO_LEFT/GO_RIGHT decisions.

**Decision needed:** Pi 4 vs Pi 5, and which camera module. Wide-angle lens is likely worth it.

### Camera Mounting Position
Where the camera sits determines what it sees:

- **Chest-mounted** — stable, covers ~1–3 m in front, natural forward view. Standard for navigation aids research.
- **Head/glasses-mounted** — follows gaze direction, but more movement noise and less stable mounting.

**Decision needed:** Chest mount for prototype (simpler, more stable). Document mount height for consistent test conditions.
