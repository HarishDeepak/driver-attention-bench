# incabin-dms-bench

A small test bench for camera-based **driver monitoring**: does a simple system warn on time by the rules of the
**Euro NCAP SD-202** test procedure, and where does it fail? Includes a **CARLA** link where the driver state slows and stops the car.

Personal project. Small datasets, simplified vehicle models, no product claims.

```
face video -> MediaPipe -> per-driver calibration -> eyes closed? looking away? -> SD-202 timers -> warnings
                                                                                     |-> scoring vs labels -> PDF test report
                                                                                     |-> escalation -> CARLA car (warn, slow, stop), ROS 2 topic
```

## Results in short

| Question | Answer | Where |
|---|---|---|
| Does it warn on time on exactly labelled synthetic drivers? | 15/18 warnings within ±0.2 s, 0 false alerts; the 3 misses are eyes behind sunglasses, reported as "unknown" on purpose | E1 |
| On real drivers (DMD dataset, 6 people)? | First run 7/66. Fixing the **measurement** (scoring scope, 20 s calibration, label definition) gave 22/49 (blink score) and 30/49 (EAR) | E2, E3 |
| Do foundation models (DINOv2, CLIP) see closed eyes better? | No: ~0.97 balanced accuracy per frame, same as the cheap signals. Zero-shot CLIP: 0.70 | E4 |
| Can a hosted VLM do the job? | No: Llama 3.2 Vision and Nemotron VL score ~0.59 on eye state, 3-9 s per frame. Offline use only | E5 |
| Does driver state matter for the car? | Inattentive driver + standard collision warning crashes at 30 and 50 km/h; warning earlier when the DMS says "inattentive" stops in time | E6 |

![CARLA collision-warning sweep](docs/img/carla_fcw.png)

Full results, including what did not work: [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md). How it works and how it is scored:
[`docs/METHOD.md`](docs/METHOD.md).

## Run

```bash
# self-checks (every module with logic has one)
python src/dms_core.py && python src/signals.py && python src/evaluate.py && python scripts/replay.py

# one video or a camera, with live view
python scripts/run_clip.py <video.mp4 | camera index> --show

# a whole scenario suite in WSL / Linux: clips x variants -> scoring -> out/<name>/report.pdf
bash run_suite.sh scenarios/synthetic.toml        # or scenarios/dmd.toml, scenarios/suite.toml

# CARLA 0.9.16 (start CarlaUE4.exe -quality-level=Low first)
python scripts/carla_fcw.py                       # collision-warning sweep
python scripts/live_demo.py sim:2-40              # escalation with a simulated driver (or a camera index)
```

Setup: Python 3.12 with `mediapipe opencv-python onnxruntime carla matplotlib`. Models go in `models/` (not in git):
- `face_landmarker.task`: https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
- `yunet.onnx` (optional face fallback): `face_detection_yunet_2023mar.onnx` from opencv_zoo
- `mobileone_s0_gaze.onnx` (optional `--gaze mobilegaze` arm): MobileGaze MobileOne-S0 export, MIT licence
ROS 2, Blender and other optional parts: [`docs/details/`](docs/details/).

## Layout

| Path | Role |
|---|---|
| `src/dms_core.py` | SD-202 warning timers |
| `src/signals.py` | MediaPipe signals, calibration, per-frame decision |
| `src/variants.py` | image variants: IR look, glare, dark, sunglasses, mask |
| `src/evaluate.py`, `src/report.py` | scoring and PDF report |
| `src/escalation.py` | warn -> slow -> minimal-risk stop |
| `scripts/` | run a clip, replay decisions, DMD conversion, CARLA, ROS 2, foundation-model and VLM tests |
| `scenarios/*.toml` | which clips, variants and settings a suite runs |
| `tools/blender/` | synthetic driver renders |
| `docs/` | `METHOD.md`, `EXPERIMENTS.md`, `details/` (setup and background) |

## Limitations

- DMD: 6 drivers, RGB face camera. Synthetic: 3 rendered identities with simple eyelids. Own phone clips not recorded yet.
- Thresholds, reaction times and braking values are stated assumptions, not values from any product.
- CARLA has no in-cabin camera: the face always comes from video, the car from the simulator.
- DMD data is not included (CC BY-NC-ND); `scripts/dmd_to_labels.py` rebuilds the labels from the official download.
