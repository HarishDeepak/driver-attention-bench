"""Camera frame -> driver signals (MediaPipe Face Landmarker; optional MobileGaze ONNX, YuNet fallback) -> per-frame (eyes_closed|None, gaze_off_road) for dms_core."""
import math
from collections import deque
from pathlib import Path
from statistics import median

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mpt
from mediapipe.tasks.python import vision

MODELS = Path(__file__).resolve().parent.parent / "models"
MODEL, YUNET, GAZE = MODELS / "face_landmarker.task", MODELS / "yunet.onnx", MODELS / "mobileone_s0_gaze.onnx"
EYES = ([362, 385, 387, 263, 373, 380], [33, 160, 158, 133, 153, 144])  # per eye: outer, 2 upper, inner, 2 lower lid points (EAR)
FOREHEAD = (109, 338, 10, 9)  # left, right, top, brow landmarks: reference skin patch for the dark-lens cue


def clahe(bgr):
    """Contrast-limited histogram equalisation on the L channel (helps dark / IR-like frames)."""
    l, a, b = cv2.split(cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB))
    return cv2.cvtColor(cv2.merge([cv2.createCLAHE(2.0, (8, 8)).apply(l), a, b]), cv2.COLOR_LAB2BGR)


def ear(pts):
    """Eye aspect ratio from 6 points (outer, up1, up2, inner, low2, low1): (|p2-p6| + |p3-p5|) / (2 |p1-p4|)."""
    p = np.asarray(pts, float)
    return (np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4])) / (2 * np.linalg.norm(p[0] - p[3]) + 1e-9)


def box_mean(gray, x0, y0, x1, y1):
    h, w = gray.shape
    x0, y0, x1, y1 = max(int(x0), 0), max(int(y0), 0), min(int(x1), w), min(int(y1), h)
    return float(gray[y0:y1, x0:x1].mean()) if x1 > x0 and y1 > y0 else float("nan")


def eye_mean(gray, p):
    """Mean grey level in the box around one eye's landmarks, padded vertically by 10 % of the eye width (a closed eye has a thin box)."""
    (x0, y0), (x1, y1) = p.min(0), p.max(0)
    return box_mean(gray, x0, y0 - 0.1 * (x1 - x0), x1, y1 + 0.1 * (x1 - x0))


class MobileGaze:
    """MobileGaze MobileOne-S0 (yakhyo/gaze-estimation, MIT; Gaze360 MAE 12.6 deg per its README) ONNX: 448x448 face crop -> (yaw, pitch) in degrees (camera frame, Gaze360 convention)."""

    def __init__(self, path=GAZE):
        import onnxruntime as ort  # lazy: only the mobilegaze arm needs it
        assert Path(path).exists(), f"missing {path}: github.com/yakhyo/gaze-estimation/releases/tag/weights"
        self.ses = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self.bins = np.arange(90, dtype=np.float32)

    def __call__(self, bgr, box):
        x0, y0, x1, y1 = box
        pad = 0.1 * max(x1 - x0, y1 - y0)  # ponytail: fixed 10 % pad, not the RetinaFace box the model was trained on
        h, w = bgr.shape[:2]
        crop = bgr[max(int(y0 - pad), 0):min(int(y1 + pad), h), max(int(x0 - pad), 0):min(int(x1 + pad), w)]
        if crop.size == 0:
            return None
        x = (cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), (448, 448)).astype(np.float32) / 255 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        yaw, pitch = self.ses.run(None, {self.ses.get_inputs()[0].name: x.transpose(2, 0, 1)[None].astype(np.float32)})
        soft = lambda z: np.exp(z - z.max()) / np.exp(z - z.max()).sum()
        return tuple(float((soft(z[0]) * self.bins).sum() * 4 - 180) for z in (yaw, pitch))


class FaceSignals:
    """__call__(bgr, t_ms) -> None (MediaPipe lost the face) or dict(blink, ear, eye_ratio, yaw, pitch, eye_h, eye_v, landmarks[, gaze_yaw, gaze_pitch]).

    After each call `self.fb` is None, or (MediaPipe lost the face but YuNet still sees one) a dict with
    the face box and, in the mobilegaze arm, gaze_yaw / gaze_pitch."""

    def __init__(self, model=MODEL, use_clahe=False, gaze=False, yunet=True, dino=False):
        opts = vision.FaceLandmarkerOptions(
            base_options=mpt.BaseOptions(model_asset_path=str(model)), running_mode=vision.RunningMode.VIDEO, num_faces=1,
            output_face_blendshapes=True, output_facial_transformation_matrixes=True)
        self.lm, self.use_clahe, self.last_ms, self.fb = vision.FaceLandmarker.create_from_options(opts), use_clahe, -1, None
        self.gaze = MobileGaze() if gaze else None
        if dino:
            from dino_eye import DinoEye  # lazy: only the dino arm needs torch
            self.dino = DinoEye()
        else:
            self.dino = None
        self.yunet = None
        if yunet:
            if YUNET.exists():
                lvl = cv2.utils.logging.getLogLevel()
                cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)  # opencv 5 prints a harmless "Targets are not supported" warning here
                self.yunet = cv2.FaceDetectorYN.create(str(YUNET), "", (320, 320), 0.7)
                cv2.utils.logging.setLogLevel(lvl)
            else:
                print(f"note: {YUNET} missing, no YuNet face fallback (opencv_zoo face_detection_yunet_2023mar.onnx)")

    def __call__(self, bgr, t_ms):
        self.last_ms = max(int(t_ms), self.last_ms + 1)  # detect_for_video needs strictly increasing ints
        self.fb = None
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)  # lens-darkness cue is measured before CLAHE (CLAHE would lift dark lenses)
        if self.use_clahe:
            bgr = clahe(bgr)
        h, w = bgr.shape[:2]
        rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        r = self.lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), self.last_ms)
        if not r.face_landmarks:
            if self.yunet is not None:
                self.yunet.setInputSize((w, h))
                _, f = self.yunet.detect(bgr)
                if f is not None:
                    x, y, bw, bh = f[0][:4]
                    self.fb = dict(box=(x, y, x + bw, y + bh))
                    if self.gaze and (g := self.gaze(bgr, self.fb["box"])):
                        self.fb.update(gaze_yaw=g[0], gaze_pitch=g[1])
            return None
        lm = [(p.x, p.y) for p in r.face_landmarks[0]]
        px = np.array(lm) * (w, h)
        bs = {c.category_name: c.score for c in r.face_blendshapes[0]}
        R = np.array(r.facial_transformation_matrixes[0])[:3, :3]
        R = R / np.linalg.norm(R, axis=0)  # drop any scale
        fx0, fx1, ft, fbr = px[FOREHEAD[0]][0], px[FOREHEAD[1]][0], px[FOREHEAD[2]][1], px[FOREHEAD[3]][1]
        sig = dict(
            blink=(bs["eyeBlinkLeft"] + bs["eyeBlinkRight"]) / 2,
            ear=float(np.mean([ear(px[i]) for i in EYES])),  # ponytail: 2D EAR, shrinks a bit with head pitch; no 3D correction
            # eye region vs forehead brightness: ~0.45 for open eyes of the test face, low behind dark lenses (scale-free: survives dark / IR frames)
            eye_ratio=float(np.nanmean([eye_mean(gray, px[i]) for i in EYES])) / (box_mean(gray, fx0, ft + 0.35 * (fbr - ft), fx1, ft + 0.8 * (fbr - ft)) + 1e-9),
            yaw=math.degrees(math.atan2(R[0, 2], R[2, 2])),
            pitch=math.degrees(math.atan2(-R[1, 2], math.hypot(R[0, 2], R[2, 2]))),
            # +eye_h = looking to the subject's left, +eye_v = up
            eye_h=(bs["eyeLookOutLeft"] + bs["eyeLookInRight"] - bs["eyeLookInLeft"] - bs["eyeLookOutRight"]) / 2,
            eye_v=(bs["eyeLookUpLeft"] + bs["eyeLookUpRight"] - bs["eyeLookDownLeft"] - bs["eyeLookDownRight"]) / 2,
            landmarks=lm)
        if self.gaze and (g := self.gaze(bgr, (*px.min(0), *px.max(0)))):
            sig.update(gaze_yaw=g[0], gaze_pitch=g[1])
        if self.dino and (d := self.dino(bgr, px, EYES)) is not None:
            sig["dino"] = d  # P(eyes closed) from DINOv2 + linear probe
        return sig


class Classifier:
    """step(t, signals|None, fb=None) -> None while calibrating, else (eyes_closed True|False|None=unknown, gaze_off_road).

    eye='blend': closed when the median blink score > halfway between the open baseline and 1.0.
    eye='ear': closed when the median EAR < halfway between the open baseline and 0.
    gaze='heuristic': away when head yaw/pitch or the eyeLook blendshapes leave the calibrated cone.
    gaze='mobilegaze': away when MobileGaze yaw/pitch deviate from the calibrated baseline by more than yaw_deg / pitch_deg
      (same 25 / 20 deg cone as the heuristic: about 2x the model's ~12.6 deg Gaze360 error, so only a clear glance counts).
    Eyes are unknown (None) when the eye region is much darker than the forehead (dark lenses) or when only YuNet still sees the face;
    while unknown, away uses head pose only. After each step `perclos` = fraction of the last perclos_s seconds (unknown time
    excluded) with eyes >= 80 % closed, None until measurable."""

    # ponytail: thresholds are calibration knobs, tune per camera; a fixed 25 deg / 20 deg cone ignores seat position and camera mounting angle
    # ponytail: unk_ratio 0.3 sits between one still face (open eyes ~0.45) and the synthetic sunglasses variant (~0.2) only, needs real clips behind real lenses
    def __init__(self, calib_s=2.0, hold_s=0.5, yaw_deg=25, pitch_deg=20, eye_dev=0.35, smooth=3, eye="blend", gaze="heuristic",
                 unk_ratio=0.3, perclos_s=60.0, closed_frac=0.5, fixed_eye_thr=None, robust=True):
        self.calib_s, self.hold_s, self.yaw_deg, self.pitch_deg, self.eye_dev = calib_s, hold_s, yaw_deg, pitch_deg, eye_dev
        self.closed_frac = closed_frac  # how far from the open baseline towards fully closed counts as closed
        self.fixed_eye_thr = fixed_eye_thr  # if set: one absolute eye threshold for everyone (blink > thr / EAR < thr), no per-driver eye calibration
        # robust: "eyes open" = the most-open 20 % of the calibration window (blink 20th / EAR 80th percentile), capped at plausible
        # open values, instead of the median: blinks, drowsy lids or a glance down in that window no longer shift the baseline
        # (night clips: median baselines of 0.53-0.56 pushed the threshold to 0.78 and hid real closures)
        self.robust = robust
        self.eye, self.gaze, self.unk_ratio, self.perclos_s = eye, gaze, unk_ratio, perclos_s
        self.buf, self.t0, self.base = [], None, None
        self.vals, self.ratios, self.closure = deque(maxlen=smooth), deque(maxlen=smooth), deque()
        self.last, self.last_t, self.perclos = (False, True), None, None

    def _dev(self, s):
        return abs(s["gaze_yaw"] - self.base["gaze_yaw"]) > self.yaw_deg or abs(s["gaze_pitch"] - self.base["gaze_pitch"]) > self.pitch_deg

    def _perclos(self, t, c):
        """c: closure 0..1 or None (unknown). O(window) per frame: fine at 25 Hz x 60 s."""
        self.closure.append((t, c))
        while self.closure[0][0] < t - self.perclos_s:
            self.closure.popleft()
        iv = [(b[0] - a[0], a[1]) for a, b in zip(self.closure, list(self.closure)[1:]) if a[1] is not None]
        tot = sum(d for d, _ in iv)
        self.perclos = sum(d for d, c in iv if c >= 0.8) / tot if tot else None

    def step(self, t, s, fb=None):
        if self.base is None:  # calibration: driver looks ahead, eyes open
            if s is not None:
                self.t0 = t if self.t0 is None else self.t0
                self.buf.append(s)
            if self.t0 is None or t - self.t0 < self.calib_s:
                return None
            self.base = {k: median(x[k] for x in self.buf) for k in self.buf[0] if k != "landmarks" and all(k in x for x in self.buf)}
            if self.robust:
                self.base["blink"] = min(float(np.percentile([x["blink"] for x in self.buf], 20)), 0.3)
                self.base["ear"] = max(float(np.percentile([x["ear"] for x in self.buf], 80)), 0.2)
                # gaze / head baseline only from open-eye frames: drooping lids raise the eyeLookDown scores, so a drowsy
                # calibration window learned "looking down" as "ahead" and then flagged every alert glance as away (night clip)
                opn = [x for x in self.buf if x["ear"] >= 0.75 * self.base["ear"] and x["blink"] <= self.base["blink"] + 0.25]
                if len(opn) >= max(5, len(self.buf) // 5):
                    self.base.update({k: median(x[k] for x in opn) for k in ("yaw", "pitch", "eye_h", "eye_v") if all(k in x for x in opn)})
            # ponytail: halfway between open baseline and fully closed (blink 1.0 / EAR 0); no per-driver closed sample
            self.closed_thr = self.base["blink"] + self.closed_frac * (1 - self.base["blink"])
            self.ear_thr = (1 - self.closed_frac) * self.base["ear"]
            if self.fixed_eye_thr is not None:
                self.closed_thr = self.ear_thr = self.fixed_eye_thr
            self.last_t = t
        if s is None:  # face lost: hold briefly, then we cannot confirm attention
            if t - self.last_t <= self.hold_s:
                return self.last
            self._perclos(t, None)
            if fb is None:
                return False, True
            # YuNet still sees a face: face present, landmarks lost -> eyes unobservable; gaze from the crop if we have it, else away as before
            return None, (self._dev(fb) if self.gaze == "mobilegaze" and "gaze_yaw" in fb else True)
        b = self.base
        self.vals.append(s["blink"] if self.eye == "blend" else s["ear"] if self.eye == "ear" else s.get("dino", self.vals[-1] if self.vals else 0.0))
        self.ratios.append(s.get("eye_ratio", 1.0))
        unk = median(self.ratios) < self.unk_ratio
        # dino: the probe outputs a probability learned across drivers, so no per-driver threshold: closed when P(closed) > 0.5
        closed = None if unk else (median(self.vals) > self.closed_thr if self.eye == "blend" else median(self.vals) < self.ear_thr
                                   if self.eye == "ear" else median(self.vals) > 0.5)
        away = abs(s["yaw"] - b["yaw"]) > self.yaw_deg or abs(s["pitch"] - b["pitch"]) > self.pitch_deg
        eyes_seen = not unk and not closed  # dark lenses or closed lids: eye direction is meaningless -> head pose only
        if self.gaze == "mobilegaze" and eyes_seen and "gaze_yaw" in s:
            away = self._dev(s)
        elif eyes_seen:
            away = away or math.hypot(s["eye_h"] - b["eye_h"], s["eye_v"] - b["eye_v"]) > self.eye_dev
        clo = ((s["blink"] - b["blink"]) / (1 - b["blink"]) if self.eye == "blend" else 1 - s["ear"] / b["ear"] if self.eye == "ear"
               else self.vals[-1])
        self._perclos(t, None if unk else min(max(clo, 0), 1))
        self.last, self.last_t = (closed, away), t
        return self.last


if __name__ == "__main__":
    def sig(blink=0.1, yaw=0.0, pitch=0.0, eye_h=0.0, eye_v=0.0, ear=0.3, eye_ratio=0.8, **kw):
        return dict(blink=blink, yaw=yaw, pitch=pitch, eye_h=eye_h, eye_v=eye_v, ear=ear, eye_ratio=eye_ratio, **kw)

    def calibrated(c, **kw):
        [c.step(i / hz, sig(**kw)) for i in range(2 * hz + 1)]
        return c

    # EAR geometry: open eye 20 wide x 6 tall -> 0.3, flat lid -> 0
    open_, flat = [(0, 0), (6, -3), (14, -3), (20, 0), (14, 3), (6, 3)], [(0, 0), (6, 0), (14, 0), (20, 0), (14, 0), (6, 0)]
    assert abs(ear(open_) - 0.3) < 1e-9 and ear(flat) == 0
    hz = 25
    c = Classifier()
    assert [c.step(i / hz, sig()) for i in range(int(2 * hz) - 1)] == [None] * (2 * hz - 1)  # still calibrating
    assert c.step(2.0, sig()) == (False, False) and abs(c.closed_thr - 0.55) < 1e-9 and abs(c.ear_thr - 0.15) < 1e-9
    t = 2.04
    assert c.step(t, sig(blink=0.9)) == (False, False)  # one closed sample: median of 3 smooths it out
    assert c.step(t + 0.04, sig(blink=0.9)) == (True, False)
    assert c.step(t + 0.08, sig(blink=0.1)) == (True, False) and c.step(t + 0.12, sig(blink=0.1)) == (False, False)
    assert c.step(t + 0.16, sig(yaw=30))[1] and c.step(t + 0.2, sig(pitch=-25))[1] and c.step(t + 0.24, sig(eye_h=0.4))[1]
    assert not c.step(t + 0.28, sig(yaw=10, pitch=10, eye_h=0.1, eye_v=0.1))[1]
    # face lost: hold last decision 0.5 s, then (False, True); with a YuNet face the eyes become unknown instead
    c.step(5.0, sig(blink=0.9)); c.step(5.04, sig(blink=0.9))
    assert c.step(5.3, None) == (True, False) and c.step(5.6, None) == (False, True)
    assert c.step(5.7, None, dict(box=(0, 0, 1, 1))) == (None, True)
    # calibration ignores no-face frames and uses medians (outlier ignored)
    c = Classifier(calib_s=1.0)
    assert c.step(0.0, None) is None and c.step(0.5, sig(blink=0.2, yaw=5)) is None
    c.step(0.7, sig(blink=0.2, yaw=5)); c.step(0.9, sig(blink=0.9, yaw=50))
    assert c.step(1.5, sig(blink=0.2, yaw=5)) == (False, False) and c.base["yaw"] == 5
    # long calibration (DMD): clip starts 6 s off-road with eyes closed, then 14 s ahead -> the median over 20 s still is "ahead, open"
    c = Classifier(calib_s=20.0)
    assert all(c.step(i / hz, sig(blink=0.9, yaw=40) if i < 6 * hz else sig(blink=0.1, yaw=3)) is None for i in range(20 * hz))
    assert c.step(20.0, sig(yaw=3)) == (False, False) and c.base["yaw"] == 3 and c.base["blink"] == 0.1
    assert c.step(20.04, sig(yaw=40))[1]  # the same head turn now counts as away
    # EAR arm: closed when EAR < half the open baseline; the blink score is ignored
    c = calibrated(Classifier(eye="ear"))
    assert c.step(2.2, sig(blink=0.9)) == (False, False)
    assert [c.step(2.24 + i / hz, sig(ear=0.05)) for i in range(2)][-1] == (True, False)
    # unknown eyes: dark lenses (low eye/forehead ratio) -> None, head pose still works, eyeLook blendshapes ignored
    c = calibrated(Classifier())
    st = [c.step(2.1 + i / hz, sig(blink=0.9, eye_ratio=0.1, eye_h=0.5)) for i in range(4)]
    assert st[-1] == (None, False) and c.step(3.0, sig(blink=0.9, eye_ratio=0.1, yaw=40)) == (None, True)
    # PERCLOS: 25 s open then 5 s closed -> 5/30 (+- one frame)
    c = calibrated(Classifier())
    for i in range(30 * hz):
        c.step(2.04 + i / hz, sig(blink=0.9 if i >= 25 * hz else 0.1))
    assert abs(c.perclos - 5 / 30) < 0.01, c.perclos
    # unknown time is excluded: 5 s unknown + 5 s closed in a 10 s window -> PERCLOS ~1, not 0.5
    c = calibrated(Classifier(perclos_s=10))
    for i in range(10 * hz):
        c.step(2.04 + i / hz, sig(blink=0.9, eye_ratio=0.1 if i < 5 * hz else 0.8))
    assert c.perclos > 0.99, c.perclos
    # MobileGaze arm: gaze angles vs the calibrated baseline (heuristic cues ignored); YuNet-crop gaze when landmarks are lost
    c = calibrated(Classifier(gaze="mobilegaze"), gaze_yaw=5, gaze_pitch=-3)
    assert c.step(2.1, sig(yaw=60, gaze_yaw=10, gaze_pitch=0)) == (False, False)
    assert c.step(2.2, sig(gaze_yaw=40, gaze_pitch=-3))[1] and c.step(2.3, sig(gaze_yaw=5, gaze_pitch=25))[1]
    assert c.step(4.0, None, dict(gaze_yaw=5, gaze_pitch=-3)) == (None, False) and c.step(4.1, None, dict(gaze_yaw=60, gaze_pitch=0)) == (None, True)
    # robust calibration: a window that is 40 % closed eyes still gives an open baseline (median would land between open and closed)
    sigs = [sig(blink=0.9 if i % 5 < 2 else 0.1, ear=0.05 if i % 5 < 2 else 0.3) for i in range(2 * hz + 1)]
    c, m = Classifier(robust=True), Classifier(robust=False)
    [(c.step(i / hz, x), m.step(i / hz, x)) for i, x in enumerate(sigs)]
    assert abs(c.base["blink"] - 0.1) < 1e-9 and abs(c.base["ear"] - 0.3) < 1e-9 and m.base["blink"] == 0.1
    sigs = [sig(blink=0.9 if i % 5 < 3 else 0.1) for i in range(2 * hz + 1)]  # 60 % closed: median breaks, robust holds
    c, m = Classifier(robust=True), Classifier(robust=False)
    [(c.step(i / hz, x), m.step(i / hz, x)) for i, x in enumerate(sigs)]
    assert abs(c.base["blink"] - 0.1) < 1e-9 and m.base["blink"] == 0.9
    # robust gaze baseline: drowsy frames in the window (lids down, eyes "looking down") do not set the "ahead" direction
    sigs = [sig(blink=0.8, ear=0.08, eye_v=-0.7) if i % 5 < 2 else sig(eye_v=0.0) for i in range(2 * hz + 1)]
    c = Classifier()
    [c.step(i / hz, x) for i, x in enumerate(sigs)]
    assert c.base["eye_v"] == 0.0 and c.step(2.2, sig(eye_v=0.0)) == (False, False)
    # DINOv2 arm: closed when the probe's P(closed) median > 0.5, independent of the blink score
    c = calibrated(Classifier(eye="dino"), dino=0.1)
    assert [c.step(2.1 + i / hz, sig(blink=0.0, dino=0.9))[0] for i in range(3)] == [False, True, True]
    print("signals OK")
