# Research Topics

Organized by which roadmap phase they unblock. Each open topic has a concrete decision that still needs to come out of it. Closed topics are kept briefly so the reasoning isn't lost.

---

## CV Module (Phase 1–2)

### Path Boundary Detection
The current pipeline detects discrete objects but not the walkable surface boundary. The spec requires detecting "path boundaries" — sidewalk edges, grass-to-pavement transitions.

Options to evaluate:
- **Semantic segmentation** (e.g., DeepLabV3+, SegFormer-B0) — labels each pixel as road/sidewalk/obstacle. Accurate but slower than detection models. Check if a TFLite or NCNN export fits Pi latency budget.
- **Lane/edge-line detection** (e.g., LaneATT, UFLDv2) — originally for driving but applicable to sidewalk edge detection. Much faster than full segmentation.
- **Depth estimation** (e.g., MiDaS, Depth Anything v2 Small) — infer relative depth from the camera image; walkable surface appears as a consistent flat plane. Useful for step/curb detection. Note: the project's metric depth comes from the LiDAR, so monocular depth would only be a cue for path shape, not a range source.

**Decision needed:** Pick one approach (or a hybrid) that runs within the latency budget on the Pi 5. Benchmark each candidate before committing.

### Surface Hazard Detection (curbs, steps, potholes)
Object detection models trained on COCO don't include these classes. Options:

- **Fine-tune YOLOv8n** on a small custom dataset of curbs/steps. Needs labelled data — look at open datasets: EgoPath, Mapillary Vistas (has sidewalk/curb labels), or collect your own.
- **LiDAR ground-plane heuristic** — once the tilt sweep (Phase 4) produces 3D points, a drop or rise in the ground height ahead indicates a step or curb. This uses the metric range the project already has.
- **Floor-plane homography** — project the ground plane and flag deviations. Works well for indoor steps, less robust outdoors.

**Decision needed:** Whether to label a small custom dataset, rely on the LiDAR ground-plane cue, or both.

### Model Export and Pi Optimization
YOLOv8n in PyTorch is not the fastest path on the Pi. The current export path is NCNN (`yolo export format=ncnn`), which `README.md` and `CLAUDE.md` use. Remaining work:

- Benchmark NCNN latency on the Pi 5 against the 300 ms budget.
- `model.export(format='tflite', int8=True)` — quantized TFLite for Pi. Needs a calibration dataset. Only pursue if NCNN misses the budget.
- `model.export(format='onnx')` then run with ONNXRuntime — good portability, not currently planned.

**Decision needed:** Confirm NCNN meets the latency budget on the Pi 5; if not, which fallback to try.

### Low-Light / Fault Detection
Spec F5 requires detecting "dim lighting" as a fault condition.

- Compute mean luminance of the frame (convert to grayscale, take mean pixel value). If below threshold, emit a fault.
- Alternatively, use YOLO confidence distribution: if average confidence of all detections drops sharply, lighting may be a factor.
- Research threshold: 100 lux is the lower bound of the spec (F2). Calibrate threshold using reference images shot at known lux values.

**Decision needed:** Luminance threshold value and whether frame-level or rolling-average check is more reliable.

---

## Camera Hardware (Phase 2)

### Binocular Camera Module and Lens
The camera is binocular hardware, but it is used for region classification only. Stereo depth is out of scope: the LiDAR provides all range, and the stereo pipeline was removed from this codebase.

Open questions:
- Which of the two image streams feeds the detector, and whether the other is used at all.
- Field of view. A wider lens captures more lateral context for GO_LEFT/GO_RIGHT decisions, but needs the region thresholds rechecked.

**Decision needed:** Which lens/stream is the region input, and the field of view for the region split.

### Camera Mounting Position
Where the camera sits determines what it sees:

- **Chest-mounted** — stable, covers ~1–3 m in front, natural forward view. Standard for navigation aids research.
- **Head/glasses-mounted** — follows gaze direction, but more movement noise and less stable mounting.

**Decision needed:** Chest mount for prototype (simpler, more stable). Document mount height for consistent test conditions.

---

## 2D LiDAR Sensing (Phase 3 — largely resolved)

The sensor is an LDROBOT STL-19P (sold as Dxtvate D500): a 360° 2D LiDAR with 12 m rated range, outputting binary packets at 230400 baud over a CP2102 USB-serial adapter. It is read by `sensor/ld19_reader.py`. The earlier text-format reader (`sensor/lidar_reader.py`) expects a vendor or microcontroller bridge and is being retired.

The single-point ultrasonic/IR sensor plan was superseded because a 2D LiDAR gives per-angle range data across the whole scan rather than one scalar per sensor, at the 0.3–2.0 m range spec F1 requires.

## 2D LiDAR → 3D Reconstruction (Phase 4)

**Active goal.** The 2D LiDAR sweeps azimuth internally already; elevation is added by tilting the sensor up and down with a servo (a nodding/push-broom scanner), accumulating scans by tilt angle over time, rather than buying true 3D LiDAR hardware.

### Tilt Hardware
- **Decided:** servo, not stepper. A hobby servo is simpler to drive; the trade-off is precision near the ends of its range.
- Sweep range and speed: how many degrees of tilt actually matter for a walking corridor (ground-to-head height at ~1–2 m), and how slow the sweep can be before it lags obstacle detection unacceptably.
- The tilt interface (`set_tilt_angle()`/`get_tilt_angle()`) should report actual (not just commanded) angle if the servo can stall or lag. Hobby servos usually have no position feedback, so this may need a measured or calibrated mapping instead.

**Decision needed:** Tilt sweep range and rate to target for Phase 1, and how to confirm the actual angle.

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
`fusion/hazard_decision.py` implements the rule-based approach: a trusted LiDAR return at or below `emergency_stop_distance_m` (0.8 m by default, `HAZARD_DEFAULT_EMERGENCY_STOP_DISTANCE_M` in `config.py`) forces STOP; otherwise the camera's region-only logic (`generate_navigation_hint()`) decides. The open question is whether this stays sufficient once LiDAR carries 3D corridor occupancy instead of a single nearest-distance scalar:

- **Confidence-weighted voting** — weight each source by its confidence score. More flexible but requires calibrated confidence values.
- **Kalman filter** — model obstacle position as a state; fuse CV detections and LiDAR readings as noisy measurements. Overkill for MVP but useful if smooth directional estimates over time become necessary.

**Decision needed:** Whether the flat threshold still holds once `generate_fused_navigation_hint()` consumes corridor occupancy (Phase 4 above) instead of `nearest_distance_m`, or whether it needs to become angle/region-aware too.

### Inter-Subsystem Communication (closed)
Previously an open question (wired UART vs BLE to a wristband). Closed: the Pi drives the LiDAR, camera, and haptic driver directly, with no separate wristband board, so no inter-board protocol is needed.

---

## Haptic and Audio Feedback (Phase 5)

### Vibration Pattern Design
Research shows users can reliably distinguish 4–6 distinct haptic patterns when the differences are in duration and rhythm rather than intensity alone.

- Look at prior work: "Tacton" pattern design principles (Brown et al.) — rhythm and envelope matter more than intensity.
- Recommended pattern set to test: single short pulse (FORWARD/clear), double pulse (GO_LEFT), double pulse offset (GO_RIGHT), continuous (STOP), long single (fault/warning).
- Test distinguishability with 5–10 people without looking at the device.
- Driver: DRV2605L over I2C from the Pi.

**Decision needed:** Final pattern set. Run recognition trials before finalizing the pattern code.

### Audio Cue Approach
Spec mentions audio as a parallel feedback channel. Open.

- Earpiece vs. bone conduction — bone conduction keeps the user's ears open to ambient sound, which is safer for navigation.
- Simple tones (beeps with distinct pitch/rhythm) vs. speech ("object 2 meters ahead") — speech is intuitive but requires TTS compute; tones are fast and offline. The plan leans toward speech with espeak or pyttsx3, but the decision is deferred.
- Audio as primary or backup — plan for audio as the redundant channel if vibration patterns are ambiguous.

**Decision needed:** Tone-based vs TTS, and earpiece vs bone conduction.

---

## Hardware Platform (cross-cutting)

### Raspberry Pi Model (closed)
Raspberry Pi 5 selected. It runs the detector, LiDAR, haptic driver, and speaker. Benchmarking against the latency budget is still in Phase 2.
