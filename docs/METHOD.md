# Method

How the bench turns a face video into warnings, and how the warnings are scored. Formulas and sources: `details/THEORY.md`.

## Pipeline

```
video (phone, DMD, Blender) -> variant (clean / IR / glare / dark / sunglasses / mask)
  -> MediaPipe Face Landmarker: blink score, eye aspect ratio (EAR), head yaw/pitch, eye direction
  -> Classifier: per-driver calibration -> per frame: eyes closed? looking away?
  -> Monitor (Euro NCAP SD-202 timers) -> warnings: microsleep / sleep / long distraction
  -> evaluate.py (vs labels) -> CSV + PDF report
  -> escalation.py -> CARLA car: warn, slow down, minimal-risk stop     -> ROS 2 topic /dms/state
```

## 1. Rules (Euro NCAP SD-202 v1.2)

| Event | Warning when | We use |
|---|---|---|
| Microsleep | eyes closed 1-2 s (maker declares) | 1.5 s |
| Sleep | eyes closed 3 s | 3 s |
| Long distraction | gaze off road 3-4 s (maker declares) | 3 s |
| Unresponsive driver | eyes closed >= 13 s or away >= 14 s | escalation to a stop |

A warning counts as on time within **±0.2 s** of the required moment.

## 2. Per-frame decision (`src/signals.py`, `Classifier`)

- **Calibration:** over the first `calib_s` seconds: head and gaze baseline = median; "eyes open" = the most-open 20 % of the window
  (robust: blinks or a glance down while calibrating no longer shift it; E11).
  2 s for live use; 20 s for DMD, because DMD clips often start with the driver looking away.
- **Eyes closed:** the blink score (or EAR) moved `closed_frac` of the way from the open baseline to fully closed (default 0.5).
- **Looking away:** head yaw > 25° or pitch > 20° from the baseline, or the eye direction left the baseline cone.
  When the eyes are closed or hidden (sunglasses), only head pose is used.
- **Eyes unknown:** eye region much darker than the forehead (dark lenses). Unknown is never counted as closed.
- A 3-frame median smooths single-frame noise; a lost face is held for 0.5 s.

**Comparison arms** (same pipeline, one part swapped): EAR instead of blink score (`--eye ear`), MobileGaze network instead of head pose
(`--gaze mobilegaze`), DINOv2 + linear probe live on the GPU (`--eye dino`), and offline only: CLIP probes, hosted VLMs (see `EXPERIMENTS.md`).

## 3. Timers and escalation

`src/dms_core.py` `Monitor` raises each warning once per episode. `src/escalation.py` turns the state into actions:
warning -> slowing to 60 km/h after 8 s -> minimal-risk manoeuvre at 13 / 14 s (hazards, 2 m/s² ramp; in CARLA: lane change
to the rightmost lane if free, stop at its edge) -> stopped, no move-off. 2 s attentive before standstill = recovery. Look-down lid flicker
(closed gaps <= 0.5 s) does not end a distraction episode.

## 4. Scoring (`src/evaluate.py`)

- Each labelled event long enough for a warning expects one: **hit** (within ±0.2 s), **early**, **late**, or **miss**.
- A warning matching no event is a **false alert**; reported per hour of scored video.
- Hit rate comes with a **Wilson 95 % interval**, because the event counts are small.
- Events that start during calibration are dropped. With `--scope`, a kind is only scored where the dataset labels that kind.
- `scripts/replay.py` re-runs the decision step on saved per-frame signals: knob changes take seconds, no new MediaPipe pass.
  `--loso` chooses a knob on the other drivers and scores it on the held-out one, so a tuned number is never tuned on its own test data.

## 5. Data

| Source | What | Notes |
|---|---|---|
| DMD (Vicomtech) | 13 face videos, 6 drivers, per-frame eye-state and gaze-zone labels | CC BY-NC-ND, academic use; kept local, never uploaded |
| Blender + MPFB2 | 6 synthetic driver clips with exact labels (`details/SYNTHETIC.md`) | covers sunglasses / mask safely |
| Own phone clips | protocol in `details/RECORDING.md` | not recorded yet; never while driving |
| CARLA 0.9.16 | road scenes for the vehicle side | no in-cabin camera in CARLA |
