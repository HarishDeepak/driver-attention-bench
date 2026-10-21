# THEORY — rules, theory, algorithms, keywords

Ordered by project stage. Tags: **[SD-202]** = read from Euro NCAP SD-202 v1.2 (July 2026) text; **[DE v1.1]** = Driver Engagement
protocol, only skimmed via search snippets, not read in full; **[general]** = standard engineering/ML knowledge, not from a Euro NCAP
document.

---

## Stage 1 — In-cabin driver monitoring (DMS) bench

### 1.1 What the rules require
- **[SD-202] Sleep** = eyes closed 3 s (fixed). **Microsleep** (1-2 s closure) and **long distraction** (long glance away, 3-4 s) are
  thresholds the manufacturer declares inside those ranges. Warning tolerance [0.2] s.
- **[SD-202] Test stimuli:** microsleep = eyes closed up to 4 s; sleep = up to 7 s; long distraction = gaze held up to 5 s.
  Owl glance (head turns with gaze) move <= 0.48 s = 12 frames @ 25 Hz; lizard glance (eyes only) <= 0.16 s = 4 frames.
  Test video >= 25 Hz.
- **[SD-202] Occlusions:** cap, hat, sunglasses (< 15 % transmittance), face mask (Type 2R, EN 14683).
- **[SD-202] Unresponsive driver:** eyes closed >= 13 s or gaze to lap >= 14 s.
- **[SD-202] Intervention spot checks** at 30-80 km/h in CCRs / CMRs / CCRm; LKA e.g. 0.3 m/s lateral at 70 km/h.
- **[DE v1.1]** 25 driver-monitoring points in scoring.
- **[general]** EU General Safety Regulation (EU) 2019/2144: DDAW (drowsiness and attention warning) and ADDW (advanced driver
  distraction warning) become mandatory in phases 2024-2026; the exact phase dates and scope are not independently verified here.
- Keywords: Euro NCAP, SD-202, DMS, OMS, DDAW, ADDW, microsleep, LGA (long glance away), owl/lizard, occlusion, CCRs/CCRm/CMRs.

### 1.2 Signals from a camera
- **Eye aspect ratio (EAR) [general]:** with six eye landmarks p1..p6,
  `EAR = (|p2-p6| + |p3-p5|) / (2 |p1-p4|)`. Open ~0.25-0.35, closed -> ~0. Threshold is person- and camera-dependent: calibrate per driver.
- **PERCLOS [general]:** fraction of a time window (classically 1 min) in which the eye is >= 80 % (P80) closed. Slow drowsiness indicator;
  SD-202 events here are much shorter, so the bench uses direct closure duration, not PERCLOS.
- **MediaPipe Face Landmarker [general]:** 478 landmarks, 52 blendshape scores (0-1, e.g. `eyeBlinkLeft/Right`, `eyeLookInLeft`, ...),
  and a 4x4 facial transformation matrix (rotation+translation of canonical face model to image) -> head yaw/pitch/roll.
  Use `running_mode=VIDEO` with monotonically increasing `timestamp_ms`.
- **Blink vs closure:** blendshape `eyeBlink*` above threshold for both eyes. A normal blink lasts ~0.1-0.4 s -> must not warn.
- **Head pose [general]:** either from the transformation matrix (Euler angles) or classic PnP (`cv2.solvePnP`) on 6 landmarks
  with a generic 3D face model + approximate camera matrix.
- **Gaze zones [general]:** off-road = head yaw/pitch outside a calibrated forward cone OR eye-direction blendshapes beyond threshold.
  Owl = head-driven, lizard = eye-driven; head pose alone misses lizard glances.
- Keywords: EAR, PERCLOS, blendshapes, facial transformation matrix, PnP, Euler angles, gaze estimation, landmark jitter, One-Euro filter.

### 1.3 Warning logic (`src/dms_core.py`)
- Per-frame booleans `(eyes_closed, gaze_off_road)` -> **state machine with timers**: start time of the current episode, fire once when
  `t - start >= threshold`, re-arm when the state ends. One warning per episode.
- Thresholds: microsleep 1.5 s, sleep 3.0 s, long distraction 3.0 s (OEM-declared values chosen inside the SD-202 ranges), tolerance 0.2 s.
- **Timing error** = `t_warn - (t_event_start + required_s)`; pass if `|err| <= 0.2 s`. Note frame quantisation at 25 Hz = 0.04 s,
  plus label error from human key-press ground truth (~0.1-0.3 s) -> the bench can only claim "within label noise".

### 1.4 Evaluation with a tiny dataset
- Per-event outcome: warned in tolerance (TP), warned outside tolerance, missed (FN), warned with no event (FP).
- Metrics: precision, recall, mean/median absolute timing error. With n ~ 5-30 events report **Wilson 95 % interval** for a proportion:
  `(p + z²/2n ± z·sqrt(p(1-p)/n + z²/4n²)) / (1 + z²/n)`, z = 1.96. Never report a bare percentage.
- **Honest scope:** self-recorded, one subject, one webcam. Results are a functional test of the pipeline, not a validation of a product.
- Keywords: precision/recall, confusion matrix, Wilson interval, label noise, inter-annotator agreement, test-retest.

### 1.5 Digital scenario variants (synthetic data)
- OpenCV transforms on recorded clips: **IR/night look** (grayscale, gamma, noise, vignette), **glare** (additive bright blob/gradient),
  **resample to 25 Hz** (frame drop/duplicate by timestamp), **synthetic occlusion overlays** (sunglasses/mask/cap drawn on landmarks).
- **Domain gap [general]:** synthetic overlays differ from real optics (IR reflectance of sunglasses, mask fit). Report accuracy per
  variant against the clean baseline; do not claim the variants substitute for real occluded recordings.
- Keywords: data augmentation, domain gap, domain randomisation, sim-to-real, scenario-based testing, ODD.

---

## Stage 2 — Vehicle part in CARLA (add-on)

### 2.1 Euro NCAP link
- **[SD-202]** intervention checks at 30-80 km/h against stationary targets (CCRs). **[DE v1.1, not independently verified]** the idea of comparing the
  forward-collision warning time when the driver is attentive vs inattentive (project shorthand: TFSS_att vs TFSS_inatt).
  The bench uses a **simplified** FCW, not a certified function.

### 2.2 Kinematics
- Closing speed `v = v_ego` (target stationary), range `d`. **TTC = d / v** (constant speed).
- Braking distance `s = v² / (2a)`; required deceleration to stop in `d`: `a_req = v² / (2d)`. Comfortable ~3-4 m/s², max ~8-9 m/s² on dry asphalt.
- Example 50 km/h = 13.9 m/s: TTC 2.5 s at d = 34.7 m; a_req = 5.6 m/s² at that point.
- **FCW logic:** warn when `TTC < TTC_thr`. DMS-gated variant: `TTC_thr = TTC_att` if attentive, larger (`TTC_inatt`) if driver is
  distracted/asleep -> warning fires earlier. Metric = difference of warning times (or of range at warning) between the two conditions.
- AEB [general]: staged (warning -> partial brake -> full brake) driven by TTC / required deceleration.
- Keywords: TTC, required deceleration, FCW, AEB, CCRs/CCRm/CCRb, reaction time, driver-in-the-loop.

### 2.3 CARLA 0.9.16
- **Client/server:** UE4 server (`CarlaUE4.exe`, port 2000) + Python client (`carla` wheel). `client.get_world()`, `load_world("Town01")`.
- **Synchronous mode:** `settings.synchronous_mode=True; fixed_delta_seconds=0.05`, then `world.tick()`; required for reproducible sweeps and
  sensor/data alignment. Async mode (used in the smoke test) is non-deterministic.
- **Actors/blueprints:** `world.get_blueprint_library()`, `spawn_actor`, `vehicle.*`, `sensor.camera.rgb`, `sensor.other.collision`,
  `apply_control(VehicleControl(throttle, brake))`; set velocity with `set_target_velocity`.
- **Sensors:** RGB, depth, semantic, radar, lidar, collision, lane invasion, GNSS, IMU. **No in-cabin camera** -> DMS state comes from the
  webcam pipeline (Stage 1), not from CARLA. DReyeVR needs CARLA 0.9.13 + VR headset: not used.
- **Maps:** OpenDRIVE (`.xodr`) road networks; Town01-Town10HD built in.
- **Scenarios:** plain Python sweep over (speed, initial range, DMS state); optionally **Scenic** (probabilistic scenario language).
- **Rendering:** `-quality-level=Low`; `-RenderOffScreen` for headless with sensors. Measured on RTX 4050 Laptop: see `STAGE0.md`.
- Keywords: UE4, Traffic Manager, synchronous mode, fixed time step, OpenDRIVE, Scenic, scenario sweep, ScenarioRunner, OpenSCENARIO.

---

## Stage 3 — Glue: ROS 2, replay, report

### 3.1 ROS 2 (Jazzy, in WSL Ubuntu 24.04)
- **Nodes** (processes with a name), **topics** (typed pub/sub), **messages** (`sensor_msgs/Image`, `std_msgs/String`, custom), **QoS**
  (reliability, durability, history/depth; sensor data usually `BEST_EFFORT`), **rclpy** Python client, **launch** files, **colcon** build,
  **ament_python** packages, **parameters** (thresholds as node params).
- **rosbag2:** `ros2 bag record/play`; replaying a recorded camera topic gives deterministic, repeatable test input; use bag time
  (`use_sim_time`) so the evaluator uses message stamps, not wall clock.
- **CARLA link:** CARLA's native ROS 2 support lists Foxy/Galactic/Humble, not Jazzy -> use a plain Python client node publishing/subscribing topics.
- **Windows <-> WSL:** NAT mode, host reachable at the default-gateway IP; Windows firewall can block WSL -> Windows ports; mirrored networking
  makes `127.0.0.1` shared. See `STAGE0.md`.
- Keywords: pub/sub, DDS, QoS profiles, rosbag2/mcap, launch, colcon, `use_sim_time`, message_filters, lifecycle.

### 3.2 Reporting
- Evaluator output: table per scenario (event, required time, warning time, error, pass/fail), per-variant summary with Wilson intervals,
  plots, short "limitations" section. PDF via matplotlib `PdfPages` or similar.
- Keywords: test report, traceability (clip id -> label -> result), reproducibility (seed, versions, commit hash).

## Source list
- Euro NCAP SD-202 v1.2: https://cdn.euroncap.com/cars/assets/SD_202_Driver_Monitoring_Test_Procedure_v1_2_d3420cb629.pdf
- CARLA docs (sensors, sync mode): https://carla.readthedocs.io
- MediaPipe Face Landmarker: https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker
