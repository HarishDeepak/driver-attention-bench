"""E12: replay saved per-frame signals (run_clip *_frames.csv) through Classifier -> Monitor -> Drowsiness; per clip: minutes scored,
microsleep warnings, peak PERCLOS, highest drowsiness level and when each level was first reached. No MediaPipe rerun.
Usage: python scripts/drowsy_eval.py out/nitymed/runs/*_frames.csv --eye ear --calib-s 20 [--csv out/e12_nitymed.csv]"""
import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from dms_core import Monitor  # noqa: E402
from drowsiness import PERCLOS_WARMUP_S, Drowsiness  # noqa: E402
from evaluate import read  # noqa: E402
from replay import KEYS  # noqa: E402
from signals import Classifier  # noqa: E402


def run(frames, eye, calib_s):
    clf, mon, dr = Classifier(eye=eye, calib_s=calib_s), Monitor(), Drowsiness()
    first, n_ms, pmax, t0, t_prev, at = {}, 0, 0.0, None, None, [0.0] * 4  # at: seconds spent at each level
    for r in frames:
        t = float(r["t"])
        s = None if r["face"] != "1" else {**{k: float(r[k]) for k in KEYS}, "eye_ratio": 0.0 if r["eyes_unknown"] == "1" else 1.0}
        st = clf.step(t, s, None)
        if st is None:
            continue
        t0 = t if t0 is None else t0
        w = mon.step(t, *st)
        n_ms += "microsleep" in w
        lv = dr.step(t, st[0], w, clf.perclos, st[1])
        pmax = max(pmax, clf.perclos or 0.0) if t - t0 >= PERCLOS_WARMUP_S else pmax
        for k in range(1, lv + 1):
            first.setdefault(k, round(t, 1))
        at[lv] += t - t_prev if t_prev is not None else 0.0
        t_prev = t
    mins = (float(frames[-1]["t"]) - t0) / 60 if t0 is not None else 0.0
    tot = sum(at) or 1.0
    return dict(minutes=round(mins, 2), microsleeps=n_ms, perclos_max=round(pmax, 3), level_max=max(first, default=0),
                share_l1plus=round(sum(at[1:]) / tot, 3), share_l2plus=round(sum(at[2:]) / tot, 3),
                **{f"t_level{k}": first.get(k, "") for k in (1, 2, 3)}), t0


def truth(events, t0, t1, hz=25):
    """Same Monitor + Drowsiness on hand-labelled closures (no PERCLOS: labels are 0.5 s steps) -> {level: first time}"""
    mon, dr, first = Monitor(), Drowsiness(), {}
    for i in range(int(t0 * hz), int(t1 * hz)):
        t = i / hz
        c = any(a <= t < b for a, b in events)
        for k in range(1, dr.step(t, c, mon.step(t, c, False), None) + 1):
            first.setdefault(k, round(t, 1))
    return first


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("frames", nargs="+")
    ap.add_argument("--eye", choices=["blend", "ear"], default="blend")
    ap.add_argument("--calib-s", type=float, default=20.0)
    ap.add_argument("--csv")
    ap.add_argument("--labels", help="clip,kind,t_start,t_end: also report the level times the hand-labelled closures give")
    a = ap.parse_args()
    ev = {}
    for r in read(a.labels) if a.labels else []:
        if r["kind"] == "closed":
            ev.setdefault(r["clip"], []).append((float(r["t_start"]), float(r["t_end"])))
    rows = []
    for f in sorted(a.frames):
        clip, fr = Path(f).name.split("__")[0], read(f)
        res, t0 = run(fr, a.eye, a.calib_s)
        if clip in ev and t0 is not None:
            tr = truth(ev[clip], t0, float(fr[-1]["t"]))
            res.update({f"truth_level{k}": tr.get(k, "") for k in (1, 2, 3)})
        rows.append(dict(clip=clip, **res))
    for r in rows:
        print(" ".join(f"{k}={v}" for k, v in r.items()))
    hist = {k: sum(r["level_max"] == k for r in rows) for k in range(4)}
    print(f"== {len(rows)} clips, {sum(r['minutes'] for r in rows):.1f} min; highest level reached: {hist}")
    if a.csv:
        with open(a.csv, "w", newline="") as fh:
            cw = csv.DictWriter(fh, list(dict.fromkeys(k for r in rows for k in r)))
            cw.writeheader()
            cw.writerows(rows)
