# Haptos Development Roadmap

Haptos is a single Raspberry Pi 5 that drives every sensor and output: a binocular camera (region classification only), an STL-19P 2D LiDAR (the sole source of depth, tilted by a servo for 3D), a DRV2605L haptic driver, and a speaker. There is no separate wristband board. Software runs as standalone Python on the Pi; ROS2 is not used.

The roadmap below is sequenced so each phase produces a testable artifact before the next phase begins.

---

## Phase 1 — CV Module: Complete Core Detection (current)

**Goal:** Reliable obstacle detection and region classification on laptop, ready to port to Pi.

Already done:
- YOLOv8n inference pipeline (Ultralytics and NCNN backends)
- LEFT/CENTER/RIGHT region mapping
- Navigation command generation (FORWARD, STOP, GO_LEFT, GO_RIGHT)
- JSONL frame logging

Remaining:
- [ ] Add **path boundary detection** — identify sidewalk edges and walkable surface so the system can detect when the user is drifting off path, not just when obstacles are ahead. (See `research.md` for approach options.)
- [ ] Add **surface hazard detection** — steps, curbs, and significant elevation changes in the forward path.
- [ ] Add **low-light fault detection** — detect when ambient lighting drops below a usable threshold and emit a fault frame result (maps to spec F5).
- [ ] Write a **benchmarking script** — run a video through the pipeline, compute per-frame latency, and report the 90th-percentile end-to-end time. Target: under 300 ms (spec F4).
- [ ] Build a **static test suite** — fixed images with known obstacle positions and expected commands. This is the only way to validate spec F3 (75% navigation accuracy) without hardware.

---

## Phase 2 — Pi Port and Latency Validation

**Goal:** Confirm the CV pipeline and LiDAR run within latency budget on the Pi 5.

Already done:
- Raspberry Pi 5 selected as the platform, running Raspberry Pi OS with SSH access.
- STL-19P LiDAR identified (sold as "Dxtvate D500"); it enumerates as a CP2102 USB-serial device.
- udev rule `/etc/udev/rules.d/99-lidar.rules` (`KERNEL=="ttyUSB*", ATTRS{idVendor}=="10c4", ATTRS{idProduct}=="ea60", MODE="0666", SYMLINK+="lidar"`) gives a stable `/dev/lidar` path. The `capstone` user is in the `dialout` group.
- Standalone LiDAR packet check (`scripts/lidar_raw_test.py`) confirms the sensor produces readings.

Remaining:
- [ ] Set up the Pi Python environment (venv, same dependencies). `requirements-pi.txt` is missing `opencv-python` and `picamera2`.
- [ ] Profile frame latency on Pi with `yolov8n` NCNN. If > 300 ms per frame, investigate:
  - Reduce input resolution
  - Switch export format (TFLite) if NCNN is not enough
  - Drop `--show` window (significant overhead on Pi without GPU)
- [ ] Validate that JSONL output format (the integration contract for later phases) is stable on Pi.
- [ ] Test under 100–50,000 lux range and measure detection rate to verify spec F2 (85%).

---

## Phase 3 — 2D LiDAR Sensing and Fusion (largely done; wiring remains)

**Goal:** Produce a range reading layer with the same output contract as the CV module.

The single-point ultrasonic/IR sensor plan was superseded by a 2D LiDAR, which gives per-angle data across a scan and covers spec F1's 0.3–2.0 m range. Already done:
- `sensor/ld19_reader.py` — binary LD19 / STL-19P packet reader that returns one full 360° scan per `read()` as a `RawLidarScan`. Written, not yet committed, and not yet wired into `main.py`.
- `sensor/lidar_filter.py` — range/quality filtering into `FilteredLidarFrame`, including fault state for a scan with no valid points.
- `sensor/lidar_buffer.py` — short rolling history of filtered frames by timestamp.
- `fusion/hazard_decision.py` — rule-based fusion: a trusted LiDAR return at or below `emergency_stop_distance_m` forces STOP, otherwise camera region logic decides.
- `sensor/lidar_reader.py` — legacy text-format reader (`angle_deg,distance_mm,quality` lines). Not used by the STL-19P; to be removed when `create_lidar_reader()` switches to the LD19 reader.

Remaining:
- [ ] Wire `Ld19Reader` into `create_lidar_reader()` and `main.py` as the serial LiDAR source, and retire the text-format reader with its tests.
- [ ] Add unit tests for the LD19 reader using a captured packet fixture: angle math, scan split on angle wrap-around, and zero-distance points being filtered.
- [ ] Add CRC checking for LD19 packets (the reader currently accepts any packet with the correct header bytes).
- [ ] Add a read timeout so `Ld19Reader.read()` returns or faults when the sensor is disconnected, instead of blocking.
- [ ] Validate 0.3 m to 2.0 m detection range (spec F1) with physical measurements against the LiDAR.
- [ ] Implement remaining fault states beyond "no valid points" (spec F5): sensor disconnected/timeout, low battery, CV pipeline error, dim lighting.

---

## Phase 4 — 2D→3D LiDAR Reconstruction and Decision Logic

**Goal:** Extend the 2D LiDAR into a 3D corridor-occupancy signal and fold it into the fused navigation command with confidence/fault handling. The LiDAR provides all depth; the tilt sweep is how it becomes 3D.

The 2D LiDAR already sweeps azimuth internally (each scan carries multiple `angle_deg` samples). Elevation comes from a tilt servo rather than 3D LiDAR hardware. In sequence:
- [ ] **Phase 0 (no new hardware):** angle-aware occupancy check in the fusion layer — today `generate_fused_navigation_hint()` only reads `LidarFrameSummary.nearest_distance_m` (nearest point across the whole scan, angle discarded) even though `FilteredLidarFrame.points_xyz`/`.angles_rad` retain every point. Restrict the near-range check to a walking-corridor arc before building further.
- [ ] **Phase 1:** tilt servo interface (`set_tilt_angle()`/`get_tilt_angle()`); extend `polar_scan_to_xyz()` with a `tilt_deg` parameter (it hardcodes `y = 0` today).
- [ ] **Phase 2:** accumulate tilt-tagged scans into a rolling 3D point cloud, with ground-plane removal and a 3D corridor-occupancy check generalizing Phase 0. `LidarFrameBuffer` may be extendable for this, or a new module may be cleaner.
- [ ] **Phase 3:** feed 3D corridor occupancy into `generate_fused_navigation_hint()` as the LiDAR-side trust signal.
- [ ] **Phase 4 (stretch):** LiDAR↔camera extrinsic calibration + point-to-bbox projection for tight fusion.
- [ ] Implement all fault states (spec F5): low battery, sensor failure, CV pipeline error, dim lighting. Each fault maps to a distinct haptic/audio output.
- [ ] Log all fusion decisions and fault events (spec F7).
- [ ] Validate spec F3 (75% navigation accuracy) end-to-end on a controlled obstacle course using fused output.

---

## Phase 5 — Haptic and Audio Output

**Goal:** Map the four navigation commands + fault states to distinct, user-distinguishable feedback patterns, driven directly by the Pi.

- [ ] Wire the DRV2605L haptic driver to the Pi over I2C and write a minimal left/right buzz test.
- [ ] Define the pattern library: FORWARD, GO_LEFT, GO_RIGHT, STOP, and each fault state each get a unique vibration duration/intensity/rhythm. Keep the set small (≤ 6 patterns).
- [ ] Implement Pi-side haptic output code in `haptos/feedback/` that maps `FrameResult` commands to those patterns.
- [ ] Decide the audio channel: TTS speech (e.g. espeak or pyttsx3) for distance announcements, or tones. **Open decision.**
- [ ] Run recognition trials: can a person without visual feedback correctly identify each pattern > 80% of the time? Iterate until yes.
- [ ] Verify response latency: time from command decision to haptic activation ≤ 300 ms end-to-end (spec F4).
- [ ] Add rate limiting on spoken or buzzed announcements so the user isn't flooded.

---

## Phase 6 — Power and Physical Integration

**Goal:** Confirm the assembled wearable meets physical constraints before final testing.

- [ ] Measure current draw for each active subsystem. Build a power budget.
- [ ] Verify 4h battery life under continuous operation (spec N1).
- [ ] Verify total wearable mass ≤ 1 kg (spec N2).
- [ ] Verify no skin-contact surface exceeds 37 °C during operation (spec N3).
- [ ] Verify all subsystems are independently testable without rewiring the full system (spec N5).

---

## Phase 7 — System Integration and Acceptance Testing

**Goal:** Run the full system against the spec acceptance criteria.

- [ ] Obstacle detection range test: place obstacles at 0.3 m, 1.0 m, 2.0 m and confirm detection (spec F1).
- [ ] Detection rate test: 85%+ across varied obstacle sizes and lighting conditions (spec F2).
- [ ] Navigation accuracy test: 75%+ correct command on standardized obstacle course (spec F3).
- [ ] Latency test: 90% of detection-to-feedback cycles < 300 ms (spec F4).
- [ ] Fault simulation test: trigger each fault state and verify user warning (spec F5).
- [ ] Offline test: disable network, confirm full operation (spec F6).
- [ ] Log inspection: confirm JSONL captures sensor readings, detections, decisions, and feedback events (spec F7).

---

## Minimum Viable System (if scope must be cut)

Per the risk assessment, if CV latency or integration time becomes a blocker:

1. LiDAR-only obstacle detection with GO_LEFT/GO_RIGHT/STOP/FORWARD commands
2. Haptic feedback for those four commands
3. CV treated as an enhancement layer, not the primary detection path
