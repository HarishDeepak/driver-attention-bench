# VLM arm: asking hosted vision-language models about driver frames

## What a VLM is
A vision-language model (VLM) takes an image plus a text prompt and answers in text. You can ask it
"are the driver's eyes closed?" and get a sentence or JSON back. No training on our data is needed: this is
*zero-shot*, the model only sees the question.

## What this arm tests
`scripts/vlm_arm.py` takes frames from the six Blender synthetic driver clips (`data/synthetic/`), sends each
frame to three hosted models, and asks for strict JSON:

    {"eyes": "open | closed | unknown", "looking_at_road": true | false}

The answers are compared with our labels (`data/synthetic/labels.csv`):
- **closed**: the frame time falls inside a labelled eyes-closed interval.
- **away**: the frame time falls inside a labelled gaze-off-road interval.

Metrics per model:
- **Balanced accuracy**: the average of "how many closed frames were found" and "how many open frames were
  correctly not flagged". It is fair when closed frames are rare.
- **F1**: the balance of false alarms and misses for the positive class.
- An answer of `unknown`, or a reply we cannot parse, counts as wrong.
- **Median latency** and **effective fps**: how long one answer takes, and how many frames per second the model
  can judge. Our real-time loop needs 25 Hz, so this shows how far a hosted model is from that.

## Why DMD is not used
The DMD dataset (`data/dmd/`) has a licence that forbids sending it to third parties. Sending its frames to a
hosted API counts as that. The script refuses any path containing "dmd", so this cannot happen by accident.

## How to run it
Get a key from NVIDIA, set it in your shell only (never in a file), then run from the project folder:

    $env:NVIDIA_API_KEY="nvapi-..."
    .venv\Scripts\python.exe scripts/vlm_arm.py

Useful flags:
- `--step 1.0`: seconds between sampled frames (default 1 s).
- `--limit 3`: only 3 frames per clip, for a quick test.
- `--models a b`: pick other models.
- `--dry-run`: build one request, print its size, send nothing (no key needed).
- `--selfcheck`: run the built-in asserts on the reply parser and the metrics (no key needed).

Outputs go to `out/vlm/frames.csv` (one row per frame and model) and `out/vlm/summary.csv` (one row per model).

## Expected role
This arm is for **offline auto-labelling and event review**: a slow model reads a recorded clip and proposes
labels or flags moments worth a human look. It is **not** the real-time loop. A hosted call takes about a second
or more, far slower than the 25 Hz that the on-device DMS loop needs (`src/dms_core.py`, `src/signals.py`).

## Results
pending
