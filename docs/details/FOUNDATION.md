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
  event-level results are in `README.md` and `docs/LEARN.md`.
- **Easy frames only.** Blink/transition frames are dropped, so the task is easier than real drowsiness detection.
- **Few subjects.** Six people and 870 closed frames. A difference of 0.01 in balanced accuracy means nothing here.
- **Thresholds are learned on training subjects only.** For (a) and (b) the cut-off is the one that maximises balanced accuracy on the
  other five subjects. The held-out subject never touches it.
- **Crop geometry is a guess.** Square crop, 1.8x the eye landmark box, not tuned. DINOv2's processor centre-crops a little more.
- **Only the small models.** Larger DINOv2 or CLIP variants may do better; we did not test them [verify].
- **Dataset licence.** DMD is CC BY-NC-ND. No images were written into the repo. Only features and per-frame scores are cached in
  `data/dmd/fm_cache/` (gitignored).

## Speed against the 25 Hz requirement

Euro NCAP SD-202 asks for at least 25 frames per second [see HANDOFF.md, "Verified facts"]. That is 40 ms per frame for everything.
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
