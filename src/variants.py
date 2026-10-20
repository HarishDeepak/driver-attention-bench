"""Digital scenario variants (THEORY 1.5), applied per frame. fps25 = frame dropping, done in run_clip. Synthetic: domain gap to real optics."""
import cv2
import numpy as np

RNG = np.random.default_rng(0)  # seeded: same noise sequence every run
L_EYE = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]
R_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
MASK = [234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 454, 329, 197, 100]


def _noise(img, sigma):
    return np.clip(img + RNG.normal(0, sigma, img.shape), 0, 255).astype(np.uint8)


def _pts(lm, idx, shape):
    h, w = shape[:2]
    return np.array([(lm[i][0] * w, lm[i][1] * h) for i in idx], np.float32)


def ir(bgr):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    h, w = g.shape
    y, x = np.ogrid[:h, :w]
    vig = 1 - 0.6 * (((x - w / 2) / (w / 2)) ** 2 + ((y - h / 2) / (h / 2)) ** 2) / 2
    return cv2.cvtColor(_noise(255 * g ** 0.7 * vig, 8), cv2.COLOR_GRAY2BGR)


def glare(bgr):
    h, w = bgr.shape[:2]
    y, x = np.ogrid[:h, :w]
    blob = np.exp(-(((x - 0.4 * w) ** 2 + (y - 0.35 * h) ** 2) / (2 * (0.2 * w) ** 2)))
    return np.clip(bgr + 220 * blob[..., None], 0, 255).astype(np.uint8)


def dark(bgr):
    return _noise(bgr * 0.2, 6)


def sunglasses(bgr, lm):
    out = bgr.copy()
    for idx in (L_EYE, R_EYE):
        p = _pts(lm, idx, bgr.shape)
        c = p.mean(0)
        cv2.fillConvexPoly(out, cv2.convexHull(((p - c) * 1.5 + c).astype(np.int32)), (15, 15, 15))
    return out


def mask(bgr, lm):
    out = bgr.copy()
    cv2.fillPoly(out, [_pts(lm, MASK, bgr.shape).astype(np.int32)], (200, 190, 180))
    return out


def apply(name, bgr, landmarks=None):
    if name in ("clean", "fps25"):
        return bgr
    if name in ("sunglasses", "mask"):
        return bgr if landmarks is None else {"sunglasses": sunglasses, "mask": mask}[name](bgr, landmarks)
    return {"ir": ir, "glare": glare, "dark": dark}[name](bgr)


if __name__ == "__main__":
    img = RNG.integers(60, 200, (120, 160, 3), dtype=np.uint8)
    lm = [(0.5, 0.5)] * 478  # degenerate placeholder, then spread the eye / mask points over the image
    for k, i in enumerate(L_EYE): lm[i] = (0.6 + 0.01 * (k % 4), 0.35 + 0.01 * (k // 4))
    for k, i in enumerate(R_EYE): lm[i] = (0.3 + 0.01 * (k % 4), 0.35 + 0.01 * (k // 4))
    for k, i in enumerate(MASK): lm[i] = (0.3 + 0.4 * (k % 6) / 5, 0.55 + 0.35 * (k // 6) / 3)
    for n in ("ir", "glare", "dark", "sunglasses", "mask"):
        out = apply(n, img, lm)
        assert out.shape == img.shape and out.dtype == img.dtype and (out != img).any(), n
    for n in ("clean", "fps25", "sunglasses"):  # unchanged: no-op variants, no landmarks
        assert (apply(n, img) == img).all(), n
    assert (apply("clean", img, lm) == img).all()
    assert apply("dark", img).mean() < 0.4 * img.mean() and apply("glare", img).mean() > img.mean()
    assert (apply("sunglasses", img, lm)[44, 98] < 40).all()  # inside the left lens
    print("variants OK")
