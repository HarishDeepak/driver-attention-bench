"""DINOv2-small eye-state arm: frozen foundation backbone on both eye crops + a linear probe -> P(eyes closed), live on the GPU.
Probe weights (models/dino_eye_probe.npz) are trained on DMD features (CC BY-NC-ND): they stay local, never in git.
Train: python src/dino_eye.py train   (needs data/dmd/fm_cache from scripts/fm_eye_probe.py)      Self-check: python src/dino_eye.py"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PROBE = ROOT / "models" / "dino_eye_probe.npz"
DINO_ID = "facebook/dinov2-small"


def eye_crop(bgr, pts):
    """Square crop, 1.8x the larger side of the eye landmarks, centred on them (clipped at the frame border)."""
    (x0, y0), (x1, y1) = pts.min(0), pts.max(0)
    cx, cy, s = (x0 + x1) / 2, (y0 + y1) / 2, 1.8 * max(x1 - x0, y1 - y0)
    return bgr[max(int(cy - s / 2), 0):int(cy + s / 2), max(int(cx - s / 2), 0):int(cx + s / 2)]


def probe_prob(feats, P):
    """feats: (n, 384) DINOv2 CLS features -> P(closed) per row, from the saved scaler + logistic regression."""
    z = ((feats - P["mean"]) / P["scale"]) @ P["coef"] + P["intercept"]
    return 1 / (1 + np.exp(-z))


class DinoEye:
    """__call__(bgr, px, eyes) -> mean P(closed) of the two eye crops, or None if a crop is empty."""

    def __init__(self, probe=PROBE):
        import torch
        from transformers import AutoImageProcessor, AutoModel
        assert Path(probe).exists(), f"missing {probe}: run python src/dino_eye.py train"
        self.torch, self.P = torch, dict(np.load(probe))
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.dt = torch.float16 if self.dev == "cuda" else torch.float32
        self.proc = AutoImageProcessor.from_pretrained(DINO_ID)
        self.net = AutoModel.from_pretrained(DINO_ID).to(self.dev, self.dt).eval()

    def __call__(self, bgr, px, eyes):
        import cv2
        crops = [cv2.cvtColor(eye_crop(bgr, px[i]), cv2.COLOR_BGR2RGB) for i in eyes]
        if any(c.size == 0 for c in crops):
            return None
        x = self.proc(images=crops, return_tensors="pt")["pixel_values"].to(self.dev, self.dt)
        with self.torch.no_grad():
            f = self.net(pixel_values=x).pooler_output.float().cpu().numpy()
        return float(probe_prob(f, self.P).mean())


def train():
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    X, y = [], []
    for f in sorted((ROOT / "data/dmd/fm_cache").glob("*.npz")):
        d = np.load(f)
        n = len(d["y"])
        X.append(d["dino"].reshape(2 * n, -1))  # each frame = left + right eye -> two single-eye samples
        y.append(np.repeat(d["y"], 2))
    X, y = np.concatenate(X), np.concatenate(y)
    sc = StandardScaler().fit(X)
    lr = LogisticRegression(class_weight="balanced", max_iter=3000).fit(sc.transform(X), y)
    PROBE.parent.mkdir(exist_ok=True)
    np.savez(PROBE, mean=sc.mean_, scale=sc.scale_, coef=lr.coef_[0], intercept=lr.intercept_)
    print(f"trained on {len(y)} eye crops ({int(y.sum())} closed) -> {PROBE}")


if __name__ == "__main__":
    if sys.argv[1:] == ["train"]:
        train()
        sys.exit()
    P = dict(mean=np.zeros(3), scale=np.ones(3), coef=np.array([2.0, 0, 0]), intercept=np.array(0.0))
    p = probe_prob(np.array([[3.0, 9, 9], [-3.0, 9, 9], [0.0, 1, 1]]), P)
    assert p[0] > 0.99 and p[1] < 0.01 and abs(p[2] - 0.5) < 1e-9, p
    pts = np.array([[100, 50], [120, 45], [140, 50], [120, 55]], float)
    assert eye_crop(np.zeros((200, 300, 3)), pts).shape[:2] == (72, 72)
    print("dino_eye OK")
