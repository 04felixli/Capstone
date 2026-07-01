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

## Distance Sensor Interface (Phase 3)

### Sensor Hardware Selection
Spec F1 requires 0.3 m to 2.0 m range.

- **HC-SR04 (ultrasonic)** — cheap, 2 cm to 4 m range, GPIO trigger/echo. Affected by soft/angled surfaces and background noise.
- **VL53L1X (IR/ToF, I2C)** — accurate up to ~4 m, faster update rate, more robust to acoustic noise. Higher cost.
- **GP2Y0A21YK0F (IR analog)** — 10–80 cm only; too short for spec F1.

**Decision needed:** HC-SR04 (simple, cheap) vs VL53L1X (more accurate, I2C). VL53L1X is likely the better fit given the 2 m range requirement and navigation accuracy spec.

### Multi-Sensor Array Layout
The spec mentions a "sensor array" (plural). Research:
- How many sensors, and at what body position (chest, waist, cane)?
- Horizontal coverage: one sensor covers ~15° cone. To cover left/center/right, three sensors may be needed.
- Crosstalk between simultaneous ultrasonic sensors — sensors must be triggered sequentially or use different frequencies.

**Decision needed:** Number of sensors, their mounting positions, and trigger sequencing strategy.

---

## Sensor Fusion (Phase 4)

### Fusion Strategy
When CV says "STOP" and distance sensor says 1.8 m (no immediate threat), or vice versa, what wins?

Approaches:
- **Rule-based priority** — distance sensor overrides CV below a threshold (e.g., < 0.5 m); CV leads above it. Simple and predictable. Start here.
- **Confidence-weighted voting** — weight each source by its confidence score. More flexible but requires calibrated confidence values.
- **Kalman filter** — model obstacle position as a state; fuse CV detections and sensor readings as noisy measurements. Overkill for MVP but useful if you need smooth directional estimates over time.

**Decision needed:** Rule-based fusion is the right starting point. Define the distance threshold that triggers sensor-only override.

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
