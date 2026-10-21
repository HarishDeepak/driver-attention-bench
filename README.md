# driver-attention-bench

**New cars in Europe must watch the driver.** A dashboard camera checks whether the eyes are closed or off the road,
and Euro NCAP tests it with fixed rules: eyes closed 3 s = "sleep", and the warning must come within 0.2 s of that moment.

This project builds such a driver monitor from open tools, **tests it the way a car maker's validation team would**, and connects
it to a car in the CARLA simulator to see what the warning is worth on the road.

<p align="center">
  <img src="results/demo_live_phone.gif" width="600" alt="Live test with a phone camera: eyes closed, looking down, head turned; warnings fire">
</p>
<p align="center"><sub>Live phone camera. Green/red: the system's decision per frame. Yellow: truth, labelled by hand without seeing
the output. 4 of 4 warnings within ±0.2 s, 0 false alerts (one take: a demo, not a statistic).
<a href="https://raw.githubusercontent.com/HarishDeepak/driver-attention-bench/main/results/videos/live_phone_take5.mp4">Full take (MP4)</a></sub></p>

## How it works

```mermaid
flowchart LR
    A["Camera video<br/>of the driver"] --> B["MediaPipe face points<br/>eyelids · head · eye direction"]
    B --> C["Decide<br/>eyes closed? looking away?<br/>for how long?"]
    C --> D["Warn<br/>microsleep 1.5 s · sleep 3 s<br/>distraction 3 s"]
    D --> E["Car (CARLA)<br/>warn · slow · stop safely"]
    D --> F["Test<br/>every warning vs the true<br/>timing → score + report"]
```

Details: [architecture](docs/ARCHITECTURE.md) · [how warnings are scored](docs/METHOD.md) · [every experiment](docs/EXPERIMENTS.md).

## The story in six steps

**1. Build it and check it where the truth is exact.** Computer-generated drivers (Blender) have perfect labels: 15 of 18
warnings on time, 0 false alerts. The 3 misses are eyes behind dark sunglasses, reported as "unknown" on purpose.

**2. Real drivers: most failures were in the test, not the detector.** On the DMD dataset the first run caught 7 of 66 events
in time. Scoring only where labels exist, a longer calibration and defining "closed" as lids actually shut raised that to
22 of 49 with the same detector (30 of 49 with the eye-aspect-ratio signal).

<p align="center"><img src="results/dmd_steps.png" width="560" alt="DMD hit rate per measurement fix"></p>

**3. Big AI models did not beat simple signals.** Foundation models (DINOv2, CLIP) recognise closed eyes as well as a simple
eyelid measurement (≈ 97 % per frame) but not better; chat-style vision models were near guessing and took seconds per frame.
Infrared matters: a model trained on daylight drops to 0.69-0.76 on real infrared eyes, 0.93-0.94 when trained on infrared.

**4. Night drivers exposed a calibration flaw.** On real night-time videos (NITYMED) the system often said "open" while the eyes
were shut: the driver's "normal" face was learned during a blink. A robust calibration raised hand-labelled hits from 7 to 16 of 27
without hurting any other dataset.

<p align="center"><img src="results/demo_night_nitymed.gif" width="520" alt="Night-time driver: microsleep and sleep warnings"></p>
<p align="center"><sub>Night driving, real car (NITYMED, CC BY 4.0, Petrellis et al.); yellow = hand-labelled truth.
<a href="https://raw.githubusercontent.com/HarishDeepak/driver-attention-bench/main/results/videos/night_nitymed_P1042793.mp4">Full video (MP4)</a></sub></p>

**5. What is a warning worth on the road?** In CARLA, a driver who stays asleep gets warning → slow-down → a minimal-risk
manoeuvre (hazards, move right if the lane is free, stop at its edge, as in UN R157). For a distracted driver, warning earlier
because the camera saw the distraction avoids crashes at 30-50 km/h. Plugging the accuracy measured on real drivers into a
Monte Carlo: at 50 km/h the crash rate falls from 87 % (no camera) to 44 % (this detector) and 17 % (a perfect one).

<p align="center">
  <img src="results/demo_carla_safe_stop.gif" width="420" alt="CARLA: slowing, minimal-risk manoeuvre, lane change right, stop at the lane edge">
  <img src="results/dms_safety_mc.png" width="420" alt="Crash probability vs speed for no DMS, measured DMS and ideal DMS">
</p>
<p align="center"><sub>Sleeping driver: warning 3.5 s, slowing 10 s, manoeuvre 15 s, standstill 23 s.
<a href="https://raw.githubusercontent.com/HarishDeepak/driver-attention-bench/main/results/videos/carla_minimal_risk_manoeuvre.mp4">Full run (MP4)</a> ·
<a href="https://raw.githubusercontent.com/HarishDeepak/driver-attention-bench/main/results/videos/face_to_carla_take5.mp4">my recorded face driving the car (MP4)</a></sub></p>

**6. Drowsy is not asleep.** Euro NCAP wants a drowsy driver warned and the collision warning made earlier; only an
*unresponsive* driver gets an automatic stop. So a drowsiness level (0-3) now counts microsleeps over 5 minutes, long blinks and
the share of time the eyes are closed; level 3 caps the speed at 60 km/h (my assumption). Two findings: looking down lowers
the eyelids and was being counted as drowsiness (fixed: false levels on alert drivers 4/6 → 1/6); and on real-life video of 6
people (UTA-RLDD) the level never fired on an alert person, but caught only 2 of 6 self-rated drowsy ones: it detects repeated
microsleeps, not early, quiet drowsiness.

<p align="center"><img src="results/demo_drowsiness_carla.gif" width="700" alt="Night driver drives CARLA: repeated microsleeps raise the drowsiness level to 3 and the speed is capped at 60 km/h"></p>
<p align="center"><sub>A night driver's video drives the CARLA car: repeated microsleeps → level 3 at 37 s → 60 km/h.
<a href="https://raw.githubusercontent.com/HarishDeepak/driver-attention-bench/main/results/videos/drowsy_night_driver_to_carla.mp4">Full run (MP4)</a></sub></p>

## What I would do next

- Earlier drowsiness cues (blink rate and speed) tuned on a dataset with drowsiness rated over time (UL-DD, DROZY).
- A real infrared camera and more people; the eye "unknown" check needs real glasses and dim light, not only synthetic ones.
- A lead car in the CARLA scene so the earlier collision warning for a drowsy driver is shown in the loop, not only in the sweep.

## Run

```bash
pip install -r requirements.txt            # Python 3.12; models go in models/ (links below)
python src/dms_core.py && python src/signals.py && python src/drowsiness.py && python src/escalation.py   # self-checks

python scripts/run_clip.py <video.mp4 | camera-index> --show [--eye ear]           # one video or camera, live overlay
bash run_suite.sh scenarios/synthetic.toml                                        # suite → scores → PDF report (Linux / WSL)
python scripts/live_demo.py sim:2-40 --headless --video out.mp4                   # CARLA 0.9.16 running; or a face video
bash scripts/ros2/record.sh                                                       # ROS 2 Jazzy: /dms/state + rosbag
```

Models (not in git): [`face_landmarker.task`](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task);
optional `yunet.onnx` (opencv_zoo `face_detection_yunet_2023mar.onnx`) and `mobileone_s0_gaze.onnx` (MobileGaze, MIT).
Optional parts (ROS 2, Blender, Cosmos): [`docs/details/`](docs/details/).

## Data and licence

Code: MIT ([`LICENSE`](LICENSE)). Data keeps its own terms and is never redistributed here except NITYMED clips (CC BY 4.0,
Petrellis et al.): DMD (Vicomtech, CC BY-NC-ND 4.0; request it, then `python scripts/dmd_to_labels.py`), UTA-RLDD (Ghoddoosian
et al. 2019; numbers only), MRL Eye (numbers only).

## Limits

Small datasets (DMD 6 drivers, 2 hand-labelled night videos, 6 real-life drowsiness subjects, a few own phone takes); thresholds,
reaction times and braking values are stated assumptions, not product values; CARLA has no in-cabin camera, so the face comes
from video; simplified car physics. Personal project by Harish Deepak (TU Darmstadt), October 2026. No product claims.
