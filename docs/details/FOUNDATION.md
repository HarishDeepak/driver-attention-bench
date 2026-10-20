# Foundation models for eye state: what we tried

Short version: we ran frozen foundation models (DINOv2, CLIP) on the eye crops of the DMD drowsiness clips and compared them with our
existing MediaPipe signals (eyeBlink blendshape, EAR). On this frame-level test they are about as good as the cheap signals, and
they are not faster. We did not put segmentation or vision-language models into the real-time loop.

## What the models are

- **DINOv2 (Meta, 2023)**: a Vision Transformer trained on images with no labels (self-supervised). We use the small version and
  take the CLS token (a 384-number summary of the image) as the feature. You train a small classifier on top; the big model stays frozen.
- **CLIP (OpenAI, 2021)**: an image encoder and a text encoder trained together so that a photo and its caption get similar vectors.
  Zero-shot means: embed the image, embed prompts such as "a photo of a closed eye", pick the closer one. We use ViT-B/32 (512 numbers).
- **SAM 3 (Meta, 2025 [verify])**: "segment anything" with text or example prompts. Output is a pixel mask of an object, not a label
  such as "eyelid closed".
- **Grounding DINO (IDEA Research, 2023 [verify])**: open-vocabulary detector. You type "eye" and get boxes. It does not say whether
  the eye is open.
- **Vision-language models (VLMs, e.g. LLaVA- or Qwen-VL-type [verify])**: a language model that looks at an image and answers in text
  ("is the driver's phone visible?"). Flexible, but slow on a small GPU.

## What we tested

Script: `scripts/fm_eye_probe.py`. Data: 7 drowsiness clips from 6 subjects (gA_1, gA_5, gB_6, gB_7, gB_9, gB_10). We kept every
5th frame that has a clear `eyes_state` open or close label (transitions and "undefined" dropped), 5665 frames in total, 870 closed.
Both eyes are cropped with MediaPipe landmarks and fed to each model. Everything is leave-one-subject-out (LOSO).

| Method | Balanced acc | F1 (closed) | ms per frame | Device |
|---|---|---|---|---|
| (a) blendshape eyeBlink threshold | 0.968 | 0.917 | 46.7 (MediaPipe pass) | CPU |
| (b) EAR threshold | 0.973 | 0.924 | 46.7 (same pass) | CPU |
| (c) CLIP zero-shot (no training) | 0.698 | 0.377 | 3.5 | GPU |
| (d) CLIP linear probe | 0.972 | 0.923 | 3.5 | GPU |
| (e) DINOv2-small linear probe | 0.966 | 0.926 | 4.5 | GPU |

Timing: GPU is an RTX 4050 Laptop (6 GB), fp16, batch 64, model forward only, two eye crops per frame. The MediaPipe number is the
CPU time of the landmarker call per frame; it was measured while the MediaPipe batch job ran on the same machine, so it is inflated [verify on an idle machine].

## Caveats (read these before quoting numbers)

- **Frame-level, not event-level.** We score single frames. We do not measure alert timing, PERCLOS or the SD-202 rules. The
  event-level results are in `README.md` and `docs/EXPERIMENTS.md`.
- **Easy frames only.** Blink/transition frames are dropped, so the task is easier than real drowsiness detection.
- **Few subjects.** Six people and 870 closed frames. A difference of 0.01 in balanced accuracy means nothing here.
- **Thresholds are learned on training subjects only.** For (a) and (b) the cut-off is the one that maximises balanced accuracy on the
  other five subjects. The held-out subject never touches it.
- **Crop geometry is a guess.** Square crop, 1.8x the eye landmark box, not tuned. DINOv2's processor centre-crops a little more.
- **Only the small models.** Larger DINOv2 or CLIP variants may do better; we did not test them [verify].
- **Dataset licence.** DMD is CC BY-NC-ND. No images were written into the repo. Only features and per-frame scores are cached in
  `data/dmd/fm_cache/` (gitignored).

## Speed against the 25 Hz requirement

Euro NCAP SD-202 asks for at least 25 frames per second (SD-202 v1.2). That is 40 ms per frame for everything.
- The DINOv2 and CLIP forward passes take about 4 ms per frame on this GPU. They fit the budget.
- The face-finding pass (MediaPipe) took about 47 ms per frame on CPU in this run, under load. That alone is over 40 ms.
- A foundation model does not replace the detector: it needs the eye crops from the detector. So on this hardware the detector
  is the bottleneck, not the model [verify on an idle machine].

## Why SAM 3, Grounding DINO and VLMs are not in the real-time loop

- **Masks are not eyelid state.** SAM 3 tells you where the eye is, not whether it is shut. Grounding DINO gives boxes. Either
  would still need a classifier on top, which is what the probes above already do.
- **Too slow for 6 GB.** Segmentation and VLM models run at a few frames per second or less on this GPU, far below 25 Hz [verify].
- **Where they do fit:** occupant monitoring that does not need 25 Hz, such as "phone in hand" or "child seat occupied", at a few fps
  [verify]. Or offline: auto-labelling new clips, which is what we would do with DMD-style data next.

## Real infrared (MRL Eye)

DMD is daylight colour video. A car cabin at night is lit by near-infrared light, so we also tested on the MRL Eye dataset
(Czech VSB group, 2018). It holds about 85k grey infrared images of eyes from 37 people, cropped tight around the eye, with
labels for eye state (open or closed), glasses, reflections, lighting (bad or good) and the camera that took the picture.
Three cameras are used: RealSense SR300 (640x480), IDS Imaging (1280x1024) and Aptina (752x480). Data: `data/mrl/` (not in git).

**Why MediaPipe blink and EAR cannot run on it.** Both need a full face to find the eye landmarks. MRL only has the eye
crop, so there is no face and no landmarks to compute EAR or the blink score from. Only the image-based models could be tested.

**Method.** Script `scripts/ir_eye_probe.py`. Up to 300 images per person per eye state (seeded), 16,920 images in total,
8,492 of them closed. Grey images are copied to three identical channels. Same DINOv2-small and CLIP ViT-B/32 features and
the same logistic-regression probe as `fm_eye_probe.py`. Two tests:
- (b) train on all DMD single-eye features (daylight RGB), test on all MRL images (day-to-infrared transfer);
- (c) train on MRL people, test on the held-out person (leave-one-subject-out).

| Method | Subset | n | Balanced acc | F1 (closed) | ms per image (GPU) |
|---|---|---|---|---|---|
| (a) CLIP zero-shot | all | 16,920 | 0.549 | 0.688 | |
| (b) DINOv2-small, DMD-RGB trained | all | 16,920 | 0.685 | 0.584 | |
| (b) CLIP ViT-B/32, DMD-RGB trained | all | 16,920 | 0.760 | 0.765 | |
| (c) DINOv2-small, MRL LOSO | all | 16,920 | 0.934 | 0.934 | |
| (c) CLIP ViT-B/32, MRL LOSO | all | 16,920 | 0.939 | 0.940 | |
| (d) CLIP, MRL LOSO (best of b/c) | lighting bad | 10,670 | 0.934 | 0.941 | |
| | lighting good | 6,250 | 0.944 | 0.937 | |
| | glasses yes | 3,901 | 0.880 | 0.886 | |
| | glasses no | 13,019 | 0.957 | 0.957 | |
| | reflections none | 14,217 | 0.948 | 0.950 | |
| | reflections small | 927 | 0.943 | 0.939 | |
| | reflections big | 1,776 | 0.864 | 0.828 | |
| | sensor 01 (RealSense) | 14,851 | 0.938 | 0.943 | |
| | sensor 02 (IDS) | 1,885 | 0.913 | 0.890 | |
| | sensor 03 (Aptina) | 184 | 1.000 | 1.000 | |
| (e) forward pass, DINOv2-small | RTX 4050 Laptop, fp16, batch 64 | | | | 1.34 |
| (e) forward pass, CLIP ViT-B/32 | RTX 4050 Laptop, fp16, batch 64 | | | | 0.73 |

Full numbers: `out/fm/ir_eye_probe.csv`.

**What it says.**
- Training on daylight colour images does not transfer well to infrared. The CLIP zero-shot prompts are close to chance (0.55).
  The DMD-trained probe gets 0.69 (DINOv2) and 0.76 (CLIP).
- Training on infrared eyes works well: about 0.93 to 0.94 on people the model has never seen. So the signal is there; the
  daylight-trained probe is missing it.
- The hardest cases are big reflections (0.86) and glasses (0.88). Lighting barely matters.

**Caveats (read before quoting).**
- Cropped eyes, not a face. Real driver monitoring needs to find the face and eyes first, which is the step that is slow here.
- Different cameras. Three small sensors, not a car DMS camera with its own optics, mounting angle and IR illuminator.
- Subsampled: at most 300 images per person per eye state, so 16,920 of the roughly 85k images. Per-person counts vary, so some people contribute fewer images.
- Frame-level, single images. No temporal information, so no blink duration or PERCLOS.
- Labels are the dataset's own (eye state from the dataset annotation, not re-checked by us). The DMD probe was trained on 11,330 single-eye rows, only 1,740 of them closed, so its numbers may understate what a balanced daylight set would give.
- Thresholds are not tuned. Sensor 03 has only 184 images, so its 1.000 means nothing.
- Dataset licence: MRL is for research use. No images were copied into the repo; only features and labels were cached in `data/mrl/fm_cache.npz` (gitignored).
