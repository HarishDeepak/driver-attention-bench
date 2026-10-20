# Experiments

All results, including the ones that did not work. Numbers are small-sample: read the 95 % intervals, not the point values.
Scoring rules: `METHOD.md` section 4.

## E1. Synthetic drivers (Blender, 6 clips, exact labels)

| Run | Hits | Rate (95 % CI) | Median timing error | False alerts |
|---|---|---|---|---|
| first run | 15/18 | 0.83 (0.61-0.94) | 0.04 s | 3 |
| after the closed-eyes fix | 15/18 | 0.83 (0.61-0.94) | 0.04 s | 0 |

- The 3 misses are eye closures behind dark sunglasses: the eyes are "unknown", which is the intended behaviour (never guess closed).
- The 3 false alerts were "long distraction" while the eyes were closed: closed lids made the eye-direction cue look away.
  Fix: use head pose only when the eyes are closed or hidden.

## E2. Real drivers: DMD, from first run to the fixed measurement

Default arm (blink score), 6 drivers. Each row changes one thing.

| Step | What changed | Hits | False alerts/h |
|---|---|---|---|
| 1 | first run: 2 s calibration, every second scored | 7/66 | 107 |
| 2 | score a kind only where DMD labels it (`--scope`) | 7/62 | 18.9 |
| 3 | 20 s calibration (clips start looking away / eyes closed) | 10/58 | 9.7 |
| 4 | label convention: "closed" = lids shut (`close`), not `closing` + `close` | 22/49 | 19.4 |

- Steps 2-4 fix the **measurement**, not the detector. Most of the gain is in how ground truth is defined.
- Step 4: with `closing` counted as closed, the timer starts while the lids are still moving, so most warnings came 0.2-1 s "late".
  SD-202 times eye closure; both conventions are reported because the choice changes the result a lot.
- Even per frame the eye signals are about 97 % correct (E4), so the remaining misses are mostly timing at the ±0.2 s limit.

## E3. Comparison arms on DMD (20 s calibration, scoped)

| Arm | Labels | Hits | False alerts/h |
|---|---|---|---|
| blink score (default) | closing + close | 10/58 | 9.7 |
| EAR | closing + close | 16/58 | 24.2 |
| blink score | close only | 22/49 | 19.4 |
| EAR | close only | 30/49 | 33.9 |
| head pose + eye direction (default gaze) | gaze clips | 2/12 | 42.6 |
| MobileGaze network | gaze clips | 3/12 | 32.0 |

**Threshold tuning (`closed_frac`), chosen on the other drivers (leave-one-subject-out):**

| Arm | Labels | Default 0.5 | Tuned on other drivers |
|---|---|---|---|
| blink score | closing + close | 10/58, 9.7/h | 16/58, 33.9/h |
| EAR | closing + close | 16/58, 24.2/h | 15/58, 36.4/h |
| blink score | close only | 22/49, 19.4/h | 25/49, 29.1/h |
| EAR | close only | 30/49, 33.9/h | 29/49, 31.5/h |

Tuning one threshold mostly trades hits for false alerts; it does not transfer clearly to a new driver. The default stays 0.5.

## E4. Foundation models for eye state (DMD, per frame, leave-one-subject-out)

5,665 frames (870 closed), 6 drivers; frames in the middle of opening/closing excluded.

| Method | Balanced accuracy | F1 (closed) | Time per frame |
|---|---|---|---|
| blink score threshold | 0.968 | 0.917 | MediaPipe pass (CPU) |
| EAR threshold | 0.973 | 0.924 | same pass |
| CLIP zero-shot ("open eye" / "closed eye") | 0.698 | 0.377 | 3.5 ms (GPU) |
| CLIP features + linear classifier | 0.972 | 0.923 | 3.5 ms (GPU) |
| DINOv2-small features + linear classifier | 0.966 | 0.926 | 4.5 ms (GPU) |

Frozen foundation features match the cheap signals but do not beat them; zero-shot CLIP is weak at a fine detail like eyelids.
Details and caveats: `details/FOUNDATION.md`.

## E5. Hosted vision-language models (VLMs) on synthetic frames

Free NVIDIA build.nvidia.com API, zero-shot, 142 frames. DMD is never sent (licence). Details: `details/VLM.md`.

| Model | Eyes closed (bal. acc.) | Looking away (bal. acc.) | Seconds per frame |
|---|---|---|---|
| Llama 3.2 11B Vision | 0.59 | 0.46 | 2.7 |
| Nemotron VL 120B (reasoning) | 0.59 | 0.63 | 8.6 (29 of 142 unparseable answers) |

For comparison, the real-time pipeline scores ~0.97 per frame on eye state (E4) at ~30 ms. Zero-shot VLMs are near chance on
eyelids and 100-300x too slow for a 25 Hz warning; their use is offline (labelling help, describing an event afterwards).
Caveat: 142 frames, synthetic faces, closures behind sunglasses count as "closed" in the labels.

Listed but not callable on the free tier (HTTP 404 "Function not found for account"): Cosmos Reason 2, Gemma 3, VILA, Phi-3 Vision.

## E6. CARLA: why driver state matters for the car

**Collision warning sweep** (`scripts/carla_fcw.py`, stationary car ahead; reaction 1.0 s attentive / 1.6 s inattentive, braking 8 m/s²):

| Speed | Attentive (warning at TTC 2.0 s) | Inattentive, fixed warning (TTC 2.0 s) | Inattentive, DMS-gated warning (TTC 3.0 s) |
|---|---|---|---|
| 30 km/h | stops | impact at 14 km/h | stops |
| 50 km/h | stops | impact at 37 km/h | stops |
| 80 km/h | impact at 45 km/h | impact at 70 km/h | impact at 15 km/h |

At 80 km/h a 2 s warning is too late even for an attentive driver: the sweep shows the limit of warning-only systems.

**Live escalation** (`scripts/live_demo.py`, simulated driver, eyes closed from 2 s): warning 3.5 s, slowing 10 s, minimal-risk stop 15 s,
standstill 22.5 s (`results/live_escalation.png`). The DMS state is also published as ROS 2 topic `/dms/state` and recorded to a rosbag (`details/ROS2.md`).

## E7. Speed

| Part | Speed |
|---|---|
| default pipeline (MediaPipe, CPU) | ~35 fps Windows / ~58 fps WSL |
| with MobileGaze | ~12-17 fps (below the 25 Hz SD-202 video rate) |
| DINOv2 / CLIP on two eye crops (GPU) | 3.5-4.5 ms per frame |
| hosted VLM | seconds per frame: offline use only |

## Considered, not done

| Idea | Why not |
|---|---|
| NVIDIA Cosmos Transfer (photoreal video from CARLA) | needs 65-80 GB GPU; hosted endpoint returns 404 on the free tier. Control videos are prepared (`details/COSMOS.md`). |
| SAM 3 / Grounding DINO in the live loop | gives masks / boxes, not eyelid state, and too slow on 6 GB. Fits occupant monitoring (phone, child seat) or offline labelling. |
| Small temporal model (GRU) on per-frame signals | per-frame accuracy is already ~97 %; the misses come from label timing, which a model trained on the same labels would copy. |
| Paid cloud GPUs | project rule: free tier only. Kaggle / Colab free T4 would work for bigger models, but DMD must not be uploaded. |
| Own phone clips | protocol ready (`details/RECORDING.md`), not recorded yet. |
