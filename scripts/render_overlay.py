"""Burn a run's per-frame decisions, the ground truth and the warnings into the video -> out/videos/<run>.mp4 (local only).
Usage: python scripts/render_overlay.py <video> <runs_dir>/<run>_frames.csv [--labels labels.csv] [--width 960]
DMD output videos must stay local (CC BY-NC-ND): out/ is gitignored."""
import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from evaluate import read  # noqa: E402

GREEN, RED, YELLOW, WHITE = (0, 220, 0), (0, 0, 255), (0, 220, 255), (255, 255, 255)


def draw(img, r, t, warns, truth=None):
    """img: BGR frame (already resized); r: frames.csv row; warns: [(t, kind)] so far; truth: label text or None."""
    h, w = img.shape[:2]
    img[:90] = (img[:90] * 0.3).astype(img.dtype)
    put = lambda s, y, c, x=10: cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 1, cv2.LINE_AA)
    if r["closed"] == "" and r["away"] == "":
        put(f"t={t:5.1f}s  calibrating (learning this driver's normal face)", 22, WHITE)
    else:
        eyes = "unknown" if r["eyes_unknown"] == "1" else "CLOSED" if r["closed"] == "1" else "open"
        put(f"t={t:5.1f}s  system: eyes {eyes}", 22, RED if eyes == "CLOSED" else GREEN)
        put(f"looking {'AWAY' if r['away'] == '1' else 'at road'}", 22, RED if r["away"] == "1" else GREEN, x=330)
    if truth is not None:
        put(f"truth: {truth}", 46, YELLOW)
    if r["face"] == "1" and r["blink"] != "":
        b = float(r["blink"])
        cv2.rectangle(img, (w - 170, 12), (w - 20, 26), WHITE, 1)
        cv2.rectangle(img, (w - 170, 12), (w - 170 + int(150 * b), 26), YELLOW, -1)
        put("blink score", 42, WHITE, x=w - 170)
    else:
        put("no face", 42, RED, x=w - 170)
    for j, (tw, k) in enumerate([x for x in warns if x[0] <= t][-1:]):
        if t - tw < 3:  # show each warning for 3 s
            cv2.rectangle(img, (0, h - 50), (w, h), (0, 0, 160), -1)
            put(f"WARNING: {k.replace('_', ' ')} (at {tw:.1f}s)", h - 18, WHITE)
    return img


def truth_at(events, t):
    k = [e[0] for e in events if e[1] <= t < e[2]]
    return ("eyes closed" if "closed" in k else "") + (" / " if len(k) > 1 else "") + ("looking away" if "away" in k else "") or "eyes open, road"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("frames")
    ap.add_argument("--labels")
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--out", default=str(ROOT / "out" / "videos"))
    a = ap.parse_args()
    fp = Path(a.frames)
    run = fp.name[: -len("_frames.csv")]
    fr, warns = read(fp), [(float(x["t"]), x["kind"]) for x in read(fp.with_name(f"{run}_warnings.csv"))]
    clip = run.rsplit("__", 1)[0]
    events = [(x["kind"], float(x["t_start"]), float(x["t_end"])) for x in read(a.labels) if x["clip"] == clip] if a.labels else None
    cap = cv2.VideoCapture(a.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    dst, vw = Path(a.out) / f"{run}.mp4", None
    for r in fr:
        ok, img = cap.read()
        if not ok:
            break
        img = cv2.resize(img, (a.width, round(img.shape[0] * a.width / img.shape[1])))
        t = float(r["t"])
        img = draw(img, r, t, warns, truth_at(events, t) if events is not None else None)
        if vw is None:
            vw = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps, (img.shape[1], img.shape[0]))
        vw.write(img)
    vw.release()
    print(f"{dst}  ({len(fr)} frames, {len(warns)} warnings)")


if __name__ == "__main__":
    main()
