"""DMD (Vicomtech Driver Monitoring Dataset) OpenLABEL/VCD annotations -> our labels.csv (clip,kind,t_start,t_end).

Layout: <videos>/<name>_rgb_face.mp4 (face camera) + <ann>/<name>_rgb_ann_drowsiness.json and ..._ann_gaze.json.
Label choices (DMD config_drowsiness.json / config_gaze.json):
  closed = eyes_state/close OR closing (2 opening and 4 undefined are NOT closed); touching/overlapping frames merge into one run.
  away   = gaze_zone/* except front and front_right (mirrors, infotainment, steering_wheel, left, right all count as away);
           not_valid is treated as unknown: neither away nor front, so it ends a run.
Scope (data/dmd/scope.csv, clip,kind,t_start,t_end): where a kind has ground truth. closed = frames with any eyes_state/* except undefined
  (clips without a drowsiness file: none); away = frames with any gaze_zone/* except not_valid (clips without a gaze file: none).
  evaluate.py --scope ignores warnings and events outside it, so e.g. a microsleep warning in a gaze-only clip is not a false alert.
Frame f covers [f/fps, (f+1)/fps). fps comes from the video file (cv2), else --fps. No run is dropped: the evaluator ignores short ones.
"""
import argparse, csv, glob, json, os
from pathlib import Path

CLOSED = {"eyes_state/close", "eyes_state/closing"}
NOT_AWAY = {"gaze_zone/front", "gaze_zone/front_right", "gaze_zone/not_valid"}


def runs(vcd, want):
    """vcd: parsed OpenLABEL dict; want: predicate on action type -> sorted [(first_frame, last_frame)] of merged runs."""
    frames = set()
    for a in (vcd.get("openlabel") or vcd.get("vcd") or vcd).get("actions", {}).values():
        if want(a.get("type", "")):
            for fi in a.get("frame_intervals", []):
                frames.update(range(fi["frame_start"], fi["frame_end"] + 1))
    out = []
    for f in sorted(frames):
        if out and f == out[-1][1] + 1:
            out[-1][1] = f
        else:
            out.append([f, f])
    return [tuple(r) for r in out]


def events(drowsiness, gaze, fps):
    r = [("closed",) + x for x in runs(drowsiness, lambda t: t in CLOSED)]
    r += [("away",) + x for x in runs(gaze, lambda t: t.startswith("gaze_zone/") and t not in NOT_AWAY)]
    return [(k, f0 / fps, (f1 + 1) / fps) for k, f0, f1 in r]


def scope(drowsiness, gaze, fps):
    r = [("closed",) + x for x in runs(drowsiness, lambda t: t.startswith("eyes_state/") and t != "eyes_state/undefined")]
    r += [("away",) + x for x in runs(gaze, lambda t: t.startswith("gaze_zone/") and t != "gaze_zone/not_valid")]
    return [(k, f0 / fps, (f1 + 1) / fps) for k, f0, f1 in r]


def video_fps(path, default):
    import cv2
    fps = cv2.VideoCapture(str(path)).get(cv2.CAP_PROP_FPS)
    return fps if fps and fps > 1 else default


def main(videos, ann, out, default_fps):
    rows, sc = [], []
    for v in sorted(glob.glob(os.path.join(videos, "*.mp4"))):
        stem, prefix = Path(v).stem, Path(v).stem.split("_rgb")[0]
        js = {k: glob.glob(os.path.join(ann, f"{prefix}*ann_{k}.json")) for k in ("drowsiness", "gaze")}
        load = lambda k: json.load(open(js[k][0], encoding="utf-8")) if js[k] else {}
        fps = video_fps(v, default_fps)
        ev = events(load("drowsiness"), load("gaze"), fps)
        print(f"{stem}: fps={fps:g} closed={sum(e[0] == 'closed' for e in ev)} away={sum(e[0] == 'away' for e in ev)} (ann: {', '.join(k for k in js if js[k]) or 'NONE'})")
        rows += [(stem, k, round(a, 3), round(b, 3)) for k, a, b in ev]
        sc += [(stem, k, round(a, 3), round(b, 3)) for k, a, b in scope(load("drowsiness"), load("gaze"), fps)]
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["clip", "kind", "t_start", "t_end"])
        w.writerows(rows)
    with open(Path(out).with_name("scope.csv"), "w", newline="") as f:
        csv.writer(f).writerows([["clip", "kind", "t_start", "t_end"]] + sc)
    for k in ("closed", "away"):
        d = [b - a for _, kk, a, b in rows if kk == k]
        print(f"{k}: {len(d)} events, >=1.5 s: {sum(x >= 1.5 for x in d)}, >=3 s: {sum(x >= 3 for x in d)}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:  # self-check on a tiny VCD-like dict, 10 fps
        act = lambda t, *iv: {"type": t, "frame_intervals": [{"frame_start": a, "frame_end": b} for a, b in iv]}
        drow = {"openlabel": {"actions": {"0": act("eyes_state/open", (0, 9)), "1": act("eyes_state/closing", (10, 14)),
                                           "2": act("eyes_state/close", (15, 34), (50, 51)), "3": act("eyes_state/opening", (35, 39))}}}
        gz = {"openlabel": {"actions": {"0": act("gaze_zone/front", (0, 19)), "1": act("gaze_zone/left", (20, 29)),
                                         "2": act("gaze_zone/left_mirror", (30, 39)), "3": act("gaze_zone/front_right", (40, 49)),
                                         "4": act("gaze_zone/not_valid", (50, 59)), "5": act("gaze_zone/infotainment", (60, 64))}}}
        got = events(drow, gz, 10)
        assert got == [("closed", 1.0, 3.5), ("closed", 5.0, 5.2), ("away", 2.0, 4.0), ("away", 6.0, 6.5)], got
        assert scope(drow, gz, 10) == [("closed", 0.0, 4.0), ("closed", 5.0, 5.2), ("away", 0.0, 5.0), ("away", 6.0, 6.5)], scope(drow, gz, 10)
        print("dmd_to_labels OK")
        sys.exit()
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default="data/dmd/face")
    ap.add_argument("--ann", default="data/dmd/ann")
    ap.add_argument("--out", default="data/dmd/labels.csv")
    ap.add_argument("--fps", type=float, default=30.0, help="fallback if the video fps cannot be read")
    ap.add_argument("--close-only", action="store_true", help="closed = eyes_state/close only (closing excluded): label-convention check")
    a = ap.parse_args()
    if a.close_only:
        CLOSED = {"eyes_state/close"}
    main(a.videos, a.ann, a.out, a.fps)
