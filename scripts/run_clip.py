"""Run one clip (video file or camera index) through variant -> FaceSignals -> Classifier -> dms_core.Monitor; write frame + warning CSVs.
Arms (--eye, --gaze) get a run-name suffix only when non-default, so evaluate/report compare them as separate variants."""
import argparse
import csv
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import variants  # noqa: E402
from dms_core import Monitor  # noqa: E402
from signals import Classifier, FaceSignals  # noqa: E402

V = ["clean", "ir", "glare", "dark", "sunglasses", "mask", "fps25"]
ap = argparse.ArgumentParser()
ap.add_argument("source", help="video path or camera index")
ap.add_argument("--variant", choices=V, default="clean")
ap.add_argument("--clahe", action="store_true")
ap.add_argument("--eye", choices=["blend", "ear"], default="blend", help="eye-closure method: blink blendshape or EAR")
ap.add_argument("--gaze", choices=["heuristic", "mobilegaze"], default="heuristic", help="gaze arm: head pose + eyeLook, or MobileGaze ONNX")
ap.add_argument("--calib-s", type=float, default=2.0, help="calibration window (s, median over it; warnings impossible meanwhile): 2 for live/own clips, 20-30 for clips that start off-road")
ap.add_argument("--show", action="store_true")
ap.add_argument("--record", help="also save the raw input video here (.mp4), e.g. a live camera session to score and label later")
ap.add_argument("--out", default=str(ROOT / "out" / "runs"))
a = ap.parse_args()

live = a.source.isdigit()
cap = cv2.VideoCapture(int(a.source) if live else a.source)
assert cap.isOpened(), f"cannot open {a.source}"
main = FaceSignals(use_clahe=a.clahe, gaze=a.gaze == "mobilegaze")
helper = FaceSignals(yunet=False) if a.variant in ("sunglasses", "mask") else None
clf, mon = Classifier(calib_s=a.calib_s, eye=a.eye, gaze=a.gaze), Monitor()
stem = ("camera" if live else Path(a.source).stem) + f"__{a.variant}" + ("_clahe" if a.clahe else "") + ("_ear" if a.eye == "ear" else "") + ("_mobilegaze" if a.gaze == "mobilegaze" else "")
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
t_start, last_kept, rows, warns = time.monotonic(), None, [], []
n_fb, rec = 0, None

while True:
    ok, frame = cap.read()
    if not ok:
        break
    # file: container timestamp (phone video may be variable frame rate); camera: wall clock
    t = time.monotonic() - t_start if live else cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
    if a.variant == "fps25" and last_kept is not None and t - last_kept < 1 / 25 - 1e-3:
        continue
    last_kept = t
    if a.record:  # ponytail: fixed 30 fps container; a live camera's real rate varies, the frames CSV keeps the true timestamps
        rec = rec or cv2.VideoWriter(a.record, cv2.VideoWriter_fourcc(*"mp4v"), 30, (frame.shape[1], frame.shape[0]))
        rec.write(frame)
    lm = None
    if helper:
        h = helper(frame, t * 1000)
        lm = h and h["landmarks"]
    s = main(variants.apply(a.variant, frame, lm), t * 1000)
    st = clf.step(t, s, main.fb)
    n_fb += main.fb is not None
    w = mon.step(t, *st) if st else []
    warns += [(t, k) for k in w]
    f = lambda k: round(s[k], 4) if s else ""
    g = s if s and "gaze_yaw" in s else main.fb or {}  # gaze from landmarks' crop, else from the YuNet crop
    rows.append([round(t, 3), int(s is not None), f("blink"), f("yaw"), f("pitch"), f("eye_h"), f("eye_v")]
                + ([int(st[0]) if st[0] is not None else "", int(st[1])] if st else ["", ""])
                + [f("ear"), "" if clf.perclos is None else round(clf.perclos, 4), "" if st is None else int(st[0] is None), int(main.fb is not None),
                   round(g["gaze_yaw"], 2) if "gaze_yaw" in g else "", round(g["gaze_pitch"], 2) if "gaze_pitch" in g else ""])
    if a.show:
        img = variants.apply(a.variant, frame, lm)
        txt = "calibrating" if st is None else f"closed={st[0]} (None=unknown) away={st[1]}"
        cv2.putText(img, f"t={t:.2f} face={s is not None} {txt}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        for i, (tw, k) in enumerate(warns[-3:]):
            cv2.putText(img, f"WARN {k} @{tw:.2f}", (10, 55 + 25 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("run_clip", img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
if rec:
    rec.release()
cv2.destroyAllWindows()
with open(out / f"{stem}_frames.csv", "w", newline="") as fh:
    cw = csv.writer(fh)
    cw.writerow("t,face,blink,yaw,pitch,eye_h,eye_v,closed,away,ear,perclos,eyes_unknown,face_fallback,gaze_yaw,gaze_pitch".split(","))
    cw.writerows(rows)
with open(out / f"{stem}_warnings.csv", "w", newline="") as fh:
    cw = csv.writer(fh)
    cw.writerow(["t", "kind"])
    cw.writerows((round(t, 3), k) for t, k in warns)
n, wall = len(rows), time.monotonic() - t_start
unk = sum(r[11] == 1 for r in rows)
print(f"{stem}: {n} frames, face {sum(r[1] for r in rows)}/{n}, fallback {n_fb}, eyes_unknown {unk}, calibrated={clf.base is not None}"
      + (f" blink0={clf.base['blink']:.3f} ear0={clf.base['ear']:.3f}" if clf.base else "") + f", warnings={[k for _, k in warns]}, fps={n / wall:.1f}")
