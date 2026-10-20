"""Real infrared eye-state test on the MRL Eye dataset (cropped IR eyes, night-time proxy). Reuses the backbones and probes of
fm_eye_probe.py: (a) CLIP zero-shot, (b) linear probe trained on DMD RGB single-eye features and tested on all MRL images,
(c) linear probe trained on MRL with leave-one-subject-out. DINOv2-small and CLIP ViT-B/32. Breakdown (d) for the best of (b)/(c).
Only features are cached, in data/mrl/fm_cache.npz (never images). data/ is gitignored.
Usage: wsl ... ~/dmsvenv/bin/python scripts/ir_eye_probe.py [--selfcheck]
"""
import argparse
import csv
import glob
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "scripts")]
from fm_eye_probe import BATCH, embed, load_models, loso_probe, metrics  # noqa: E402

RAW, CACHE, OUT = ROOT / "data/mrl/raw", ROOT / "data/mrl/fm_cache.npz", ROOT / "out/fm"
PER, CHUNK, SEED = 300, 1024, 0
DMD_DIM = {"dino": 384, "clip": 512}  # per eye: DINOv2-small CLS and CLIP image embed
# MRL name: subjectID_imageID_gender_glasses_eyestate_reflections_lighting_sensorID.png (eyestate 0 closed, 1 open)
NAME = re.compile(r"s(\d{4})_(\d+)_([01])_([01])_([01])_([012])_([01])_(\d{2})\.png")
SUBSETS = {"lighting=bad": ("lighting", 0), "lighting=good": ("lighting", 1), "glasses=yes": ("glasses", 1),
           "glasses=no": ("glasses", 0), "reflections=none": ("reflections", 0), "reflections=small": ("reflections", 1),
           "reflections=big": ("reflections", 2), "sensor=01": ("sensor", 1), "sensor=02": ("sensor", 2)}


def parse(name):
    """MRL file name -> metadata. y is 1 for closed (same convention as fm_eye_probe.py)."""
    m = NAME.fullmatch(name)
    if not m:
        raise ValueError(f"unexpected MRL file name: {name}")
    subj, _, _, glasses, state, refl, light, sensor = m.groups()
    return dict(subj=int(subj), y=1 - int(state), glasses=int(glasses), reflections=int(refl), lighting=int(light), sensor=int(sensor))


def sample(paths, per=PER, seed=SEED):
    """Up to `per` images per (subject, eye state), seeded. Returns the chosen paths in a fixed order."""
    rng, groups = np.random.default_rng(seed), {}
    for p in paths:
        m = parse(p.name)
        groups.setdefault((m["subj"], m["y"]), []).append(p)
    out = []
    for k in sorted(groups):
        ps = sorted(groups[k])
        if len(ps) > per:
            ps = [ps[i] for i in sorted(rng.choice(len(ps), per, replace=False))]
        out += ps
    return out


def extract(M):
    """All sampled MRL images -> cached per-image metadata, features, CLIP zero-shot prob and mean GPU ms per image."""
    if CACHE.exists():
        return dict(np.load(CACHE))
    files = sample(sorted(RAW.rglob("*.png")))
    meta = [parse(p.name) for p in files]
    embed([Image.open(files[0]).convert("RGB")] * BATCH, M)  # warm-up, not timed
    feats, sec = {"dino": [], "clip": [], "zs": []}, {"dino": 0.0, "clip": 0.0}
    for i in range(0, len(files), CHUNK):
        imgs = [Image.open(p).convert("RGB") for p in files[i:i + CHUNK]]  # grey IR -> 3 identical channels
        d, c, z, s = embed(imgs, M)
        feats["dino"].append(d)
        feats["clip"].append(c)
        feats["zs"].append(z)
        sec["dino"] += s["dino"]
        sec["clip"] += s["clip"]
        print(f"  embedded {min(i + CHUNK, len(files))}/{len(files)}", flush=True)
    n = len(files)
    out = {k: np.array([m[k] for m in meta]) for k in ("subj", "y", "glasses", "reflections", "lighting", "sensor")}
    out.update(dino=np.concatenate(feats["dino"]), clip=np.concatenate(feats["clip"]), zs=np.concatenate(feats["zs"]),
               ms_dino=np.array(1000 * sec["dino"] / n), ms_clip=np.array(1000 * sec["clip"] / n))
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, **out)
    return out


def dmd_single_eye():
    """All DMD cached frames split into single-eye samples: features concatenated [left, right] -> two rows each."""
    X, Y = {k: [] for k in DMD_DIM}, []
    for f in sorted(glob.glob(str(ROOT / "data/dmd/fm_cache/*.npz"))):
        d = np.load(f)
        for k, dim in DMD_DIM.items():
            assert d[k].shape[1] == 2 * dim, (f, k, d[k].shape)
            X[k] += [d[k][:, :dim], d[k][:, dim:]]
        Y += [d["y"], d["y"]]
    return {k: np.concatenate(v) for k, v in X.items()}, np.concatenate(Y)


def fit_probe(X, y):
    return make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000)).fit(X, y)


def selfcheck():
    p = parse("s0001_00002_1_0_1_2_0_02.png")
    assert p == dict(subj=1, y=0, glasses=0, reflections=2, lighting=0, sensor=2), p
    assert parse("s0037_00010_0_1_0_0_1_01.png")["y"] == 1
    try:
        parse("bad.png")
        raise AssertionError("parse accepted a bad name")
    except ValueError:
        pass
    names = [Path(f"s{s:04d}_{i:05d}_0_0_{st}_0_0_01.png") for s in range(3) for st in (0, 1) for i in range(10)]
    got = sample(names, per=4)
    assert len(got) == 3 * 2 * 4, len(got)
    m = metrics(np.array([0, 0, 1, 1]), np.array([0, 0, 1, 1]))
    assert m["balanced_acc"] == 1 and m["f1_closed"] == 1, m
    m = metrics(np.array([0, 0, 1, 1]), np.array([1, 1, 1, 1]))
    assert abs(m["balanced_acc"] - 0.5) < 1e-9, m
    print("ir_eye_probe selfcheck OK")


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    if a.selfcheck:
        return selfcheck()
    M = load_models()
    gpu = torch.cuda.get_device_name(0) if M["dev"] == "cuda" else "CPU"
    D = extract(M)
    y, subj, n = D["y"], D["subj"], len(D["y"])
    print(f"device: {gpu} | MRL images {n} (closed {int(y.sum())}) | subjects {len(set(subj))}")
    Xd, yd = dmd_single_eye()
    print(f"DMD single-eye training rows {len(yd)} (closed {int(yd.sum())})")

    preds = {"(a) CLIP zero-shot": (D["zs"] > 0.5).astype(int)}
    for k, lab in (("dino", "DINOv2-small"), ("clip", "CLIP ViT-B/32")):
        preds[f"(b) {lab}, DMD-RGB trained"] = fit_probe(Xd[k], yd).predict(D[k])
        preds[f"(c) {lab}, MRL LOSO"] = loso_probe(D[k], y, subj)
    res = {k: metrics(y, p) for k, p in preds.items()}
    best = max((k for k in preds if not k.startswith("(a)")), key=lambda k: res[k]["balanced_acc"])
    rows = [(k, "all", n, res[k], None) for k in preds]
    for sub, (field, val) in SUBSETS.items():
        m = D[field] == val
        if m.any():
            rows.append((f"(d) {best}", sub, int(m.sum()), metrics(y[m], preds[best][m]), None))
    rows += [("(e) forward pass, DINOv2-small", gpu + ", fp16, batch 64", n, None, float(D["ms_dino"])),
             ("(e) forward pass, CLIP ViT-B/32", gpu + ", fp16, batch 64", n, None, float(D["ms_clip"]))]

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "ir_eye_probe.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "subset", "n", "bal_acc", "f1_closed", "ms_per_image"])
        for name, sub, cnt, m, ms in rows:
            w.writerow([name, sub, cnt, f"{m['balanced_acc']:.3f}" if m else "", f"{m['f1_closed']:.3f}" if m else "",
                        f"{ms:.2f}" if ms is not None else ""])
    print(f"\n{'method':38s} {'subset':28s} {'n':>6s} {'bal_acc':>7s} {'F1_closed':>9s} {'ms/img':>7s}")
    for name, sub, cnt, m, ms in rows:
        b = f"{m['balanced_acc']:7.3f} {m['f1_closed']:9.3f}" if m else f"{'':7s} {'':9s}"
        t = f"{ms:7.2f}" if ms is not None else ""
        print(f"{name:38s} {sub[:28]:28s} {cnt:6d} {b} {t}")
    print(f"\nbest of (b)/(c) for breakdown: {best}\n-> {OUT / 'ir_eye_probe.csv'}")


if __name__ == "__main__":
    main()
