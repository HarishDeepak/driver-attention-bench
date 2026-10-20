"""Frame-level eye-state probe on DMD drowsiness clips: MediaPipe baselines (eyeBlink, EAR) vs frozen foundation models
(DINOv2-small and CLIP ViT-B/32 on both eye crops; linear probe, plus CLIP zero-shot), all leave-one-subject-out.
Eye crops stay in memory; only per-frame scores and features are cached, in data/dmd/fm_cache/ (never images).
Usage: wsl ... ~/dmsvenv/bin/python scripts/fm_eye_probe.py [--clip SUBSTR] [--selfcheck]
"""
import argparse
import csv
import glob
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")  # the DMD MediaPipe job shares the CPU: stay at <= 4 workers

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import balanced_accuracy_score, f1_score  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from dmd_to_labels import runs  # noqa: E402
from signals import EYES, FaceSignals  # noqa: E402

FACE, ANN = ROOT / "data/dmd/face", ROOT / "data/dmd/ann"
CACHE, OUT = ROOT / "data/dmd/fm_cache", ROOT / "out/fm"
STRIDE, CAP, BATCH = 5, 3000, 64
PROMPTS = ["a photo of an open eye", "a photo of a closed eye"]


def drowsy_clips(name=""):
    """(clip stem, drowsiness annotation path) for every drowsiness file whose name contains `name`."""
    for j in sorted(glob.glob(str(ANN / "*_ann_drowsiness.json"))):
        stem = Path(j).name.split("_rgb_ann")[0]
        if name in stem:
            yield stem, j


def frame_labels(vcd):
    """frame -> 1 (eyes closed) / 0 (eyes open). opening, closing and undefined are dropped (no clear label)."""
    lab = {}
    for want, y in (("eyes_state/open", 0), ("eyes_state/close", 1)):
        for a, b in runs(vcd, lambda t: t == want):
            lab.update((f, y) for f in range(a, b + 1))
    return lab


def eye_crop(bgr, pts):
    """Square crop, 1.8x the larger side of the eye landmarks, centred on them (clipped at the frame border)."""
    (x0, y0), (x1, y1) = pts.min(0), pts.max(0)
    cx, cy, s = (x0 + x1) / 2, (y0 + y1) / 2, 1.8 * max(x1 - x0, y1 - y0)
    return bgr[max(int(cy - s / 2), 0):int(cy + s / 2), max(int(cx - s / 2), 0):int(cx + s / 2)]


def as_tensor(o):
    """transformers 5 returns an output object from get_*_features; older versions return the tensor."""
    import torch
    return o if torch.is_tensor(o) else o.pooler_output


def load_models():
    import torch
    import torch.nn.functional as F
    from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dt = torch.float16 if dev == "cuda" else torch.float32
    dino_id, clip_id = "facebook/dinov2-small", "openai/clip-vit-base-patch32"
    clip, cp = CLIPModel.from_pretrained(clip_id).to(dev, dt).eval(), CLIPProcessor.from_pretrained(clip_id)
    with torch.no_grad():
        txt = as_tensor(clip.get_text_features(**cp(text=PROMPTS, return_tensors="pt", padding=True).to(dev)))
    return dict(dev=dev, dt=dt, dino=AutoModel.from_pretrained(dino_id).to(dev, dt).eval(),
                dino_proc=AutoImageProcessor.from_pretrained(dino_id), clip=clip, clip_proc=cp,
                txt=F.normalize(txt.float(), dim=-1))


def sync(dev):
    import torch
    if dev == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter()


def embed(imgs, M):
    """imgs: PIL crops (2 per frame, left then right). Returns per-crop DINOv2 CLS, CLIP image embed, CLIP closed-prob, and forward seconds."""
    import torch
    import torch.nn.functional as F
    dino, clip, zs, sec = [], [], [], {"dino": 0.0, "clip": 0.0}
    with torch.no_grad():
        for i in range(0, len(imgs), BATCH):
            b = imgs[i:i + BATCH]
            x = M["dino_proc"](images=b, return_tensors="pt")["pixel_values"].to(M["dev"], M["dt"])
            t = sync(M["dev"])
            d = M["dino"](pixel_values=x).pooler_output
            sec["dino"] += sync(M["dev"]) - t
            x = M["clip_proc"](images=b, return_tensors="pt")["pixel_values"].to(M["dev"], M["dt"])
            t = sync(M["dev"])
            e = F.normalize(as_tensor(M["clip"].get_image_features(pixel_values=x)).float(), dim=-1)
            sec["clip"] += sync(M["dev"]) - t
            dino.append(d.float().cpu().numpy())
            clip.append(e.cpu().numpy())
            zs.append((100 * e @ M["txt"].T).softmax(-1)[:, 1].cpu().numpy())
    return np.concatenate(dino), np.concatenate(clip), np.concatenate(zs), sec


def extract(stem, ann, M):
    """One clip -> cached dict: per kept frame (labelled, face found) blink, EAR, features of both eyes, CLIP zero-shot prob."""
    cache = CACHE / f"{stem}.npz"
    if cache.exists():
        return dict(np.load(cache))
    lab = frame_labels(json.load(open(ann, encoding="utf-8")))
    want = sorted(f for f in lab if f % STRIDE == 0)
    if not want:
        return None
    if len(want) > CAP:
        want = want[::math.ceil(len(want) / CAP)]
    wanted, cap = set(want), cv2.VideoCapture(str(FACE / f"{stem}_rgb_face.mp4"))
    fps, fs = cap.get(cv2.CAP_PROP_FPS), FaceSignals(yunet=False)
    rows, crops, mp_s = [], [], 0.0
    for f in range(want[-1] + 1):
        if f not in wanted:
            cap.grab()
            continue
        ok, bgr = cap.read()
        if not ok:
            break
        t = time.perf_counter()
        sig = fs(bgr, f / fps * 1000)
        mp_s += time.perf_counter() - t
        if sig is None:  # no face: MediaPipe fails on it, so no method can score it
            continue
        h, w = bgr.shape[:2]
        px = np.array(sig["landmarks"]) * (w, h)
        eyes = [cv2.cvtColor(eye_crop(bgr, px[i]), cv2.COLOR_BGR2RGB) for i in EYES]
        if any(e.size == 0 for e in eyes):
            continue
        rows.append((f, lab[f], sig["blink"], sig["ear"]))
        crops += [Image.fromarray(e) for e in eyes]
    if not rows:
        return None
    dino, clip, zs, sec = embed(crops, M)
    n = len(rows)
    r = np.array(rows, float)
    out = dict(frame=r[:, 0].astype(int), y=r[:, 1].astype(int), blink=r[:, 2], ear=r[:, 3],
               subj=np.array(["_".join(stem.split("_")[:2])] * n), dino=dino.reshape(n, -1), clip=clip.reshape(n, -1),
               zs=zs.reshape(n, 2).mean(1), ms_mp=np.array(1000 * mp_s / n), ms_dino=np.array(1000 * sec["dino"] / (2 * n)),
               ms_clip=np.array(1000 * sec["clip"] / (2 * n)))
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **out)
    return out


def best_threshold(s, y):
    """Threshold on a closedness score (closed when s > thr) that maximises balanced accuracy on the given (training) frames."""
    grid = np.quantile(s, np.linspace(0, 1, 201))
    return grid[int(np.argmax([balanced_accuracy_score(y, s > t) for t in grid]))]


def loso_threshold(s, y, subj):
    pred = np.zeros_like(y)
    for g in np.unique(subj):
        te = subj == g
        pred[te] = s[te] > best_threshold(s[~te], y[~te])
    return pred


def loso_probe(X, y, subj):
    pred = np.zeros_like(y)
    for g in np.unique(subj):
        te = subj == g
        clf = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000)).fit(X[~te], y[~te])
        pred[te] = clf.predict(X[te])
    return pred


def metrics(y, pred):
    return dict(balanced_acc=balanced_accuracy_score(y, pred), f1_closed=f1_score(y, pred, pos_label=1, zero_division=0))


def selfcheck():
    rng = np.random.default_rng(0)
    n = 240
    subj, y = np.repeat(np.array(["s1", "s2", "s3", "s4"]), n // 4), rng.integers(0, 2, n)
    X, s = rng.normal(size=(n, 8)) + y[:, None] * 0.5, rng.normal(size=n) + y
    preds = {"probe": loso_probe(X, y, subj), "threshold": loso_threshold(s, y, subj), "random": (rng.random(n) > 0.5).astype(int)}
    for name, p in preds.items():
        m = metrics(y, p)
        assert all(0 <= v <= 1 for v in m.values()), (name, m)
    assert metrics(y, loso_threshold(s, y, subj))["balanced_acc"] > 0.55
    sep = np.array([0., 0., 1., 1.])
    assert metrics(np.array([0, 0, 1, 1]), sep > best_threshold(sep, np.array([0, 0, 1, 1])))["balanced_acc"] == 1
    print("fm_eye_probe selfcheck OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default="", help="only clips whose name contains this")
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    if a.selfcheck:
        return selfcheck()
    M = load_models()
    print(f"device: {M['dev']}")
    clips = []
    for stem, ann in drowsy_clips(a.clip):
        d = extract(stem, ann, M)
        print(f"{stem}: " + ("skipped (no labelled face frames)" if d is None else f"{len(d['y'])} frames, closed {int(d['y'].sum())}"))
        if d is not None:
            clips.append(d)
    keys = [k for k in clips[0] if not k.startswith("ms_")]
    D = {k: np.concatenate([c[k] for c in clips]) for k in keys}
    y, subj = D["y"], D["subj"]
    assert len(set(subj)) > 1, "leave-one-subject-out needs at least two subjects (drop --clip)"
    ms ={k: float(np.mean([c[k] for c in clips])) for k in ("ms_mp", "ms_dino", "ms_clip")}
    rows = [
        ("(a) blendshape eyeBlink threshold", metrics(y, loso_threshold(D["blink"], y, subj)), ms["ms_mp"], "CPU (MediaPipe)"),
        ("(b) EAR threshold", metrics(y, loso_threshold(-D["ear"], y, subj)), ms["ms_mp"], "CPU (MediaPipe)"),
        ("(c) CLIP zero-shot", metrics(y, (D["zs"] > 0.5).astype(int)), ms["ms_clip"], M["dev"]),
        ("(d) CLIP linear probe", metrics(y, loso_probe(D["clip"], y, subj)), ms["ms_clip"], M["dev"]),
        ("(e) DINOv2-small linear probe", metrics(y, loso_probe(D["dino"], y, subj)), ms["ms_dino"], M["dev"]),
    ]
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "eye_probe.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "balanced_acc", "f1_closed", "n_frames", "n_closed", "ms_per_frame", "device"])
        for name, m, t, dev in rows:
            w.writerow([name, f"{m['balanced_acc']:.3f}", f"{m['f1_closed']:.3f}", len(y), int(y.sum()), f"{t:.2f}", dev])
    print(f"\n{'method':34s} {'bal_acc':>7s} {'F1_closed':>9s} {'frames':>6s} {'ms/frame':>8s}  device")
    for name, m, t, dev in rows:
        print(f"{name:34s} {m['balanced_acc']:7.3f} {m['f1_closed']:9.3f} {len(y):6d} {t:8.2f}  {dev}")
    print(f"subjects: {sorted(set(subj))}  -> {OUT / 'eye_probe.csv'}")


if __name__ == "__main__":
    main()
