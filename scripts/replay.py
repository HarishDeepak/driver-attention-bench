"""Replay saved per-frame signals (run_clip *_frames.csv) through Classifier + Monitor with other knobs: seconds instead of a MediaPipe rerun.
--closed-frac 0.3 0.4 0.5 ...: score each value; --loso also picks the value on the other subjects (DMD clip prefix gA_1, gB_6, ...) and
scores it on the held-out subject, so the tuned number is not tuned on the data it is reported on.
Usage: python scripts/replay.py --runs out/dmd/runs --labels data/dmd/labels.csv --scope data/dmd/scope.csv --closed-frac 0.3 0.4 0.5 --loso"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from dms_core import Monitor  # noqa: E402
from evaluate import apply_scope, calib_end, read, score_run, wilson  # noqa: E402
from signals import Classifier  # noqa: E402

KEYS = ("blink", "ear", "yaw", "pitch", "eye_h", "eye_v")


def replay(frames, eye, gaze, **kw):
    """frames: rows of a *_frames.csv -> (warnings [(t, kind)], calibration end)."""
    clf, mon, warns, ce = Classifier(eye=eye, gaze=gaze, **kw), Monitor(), [], None
    for r in frames:
        t = float(r["t"])
        g = {k: float(r[k]) for k in ("gaze_yaw", "gaze_pitch") if r.get(k, "") != ""} if gaze == "mobilegaze" else {}
        s = None if r["face"] != "1" else {**{k: float(r[k]) for k in KEYS}, "eye_ratio": 0.0 if r["eyes_unknown"] == "1" else 1.0, **g}
        st = clf.step(t, s, (g if r["face_fallback"] == "1" else None))
        if st is not None:
            ce = t if ce is None else ce
            warns += [(t, k) for k in mon.step(t, *st)]
    return warns, ce


def score(frames, warns, ce, events, scope, tol=None):
    """-> (hits, expected, false alerts, scored hours) for one run, same rules as evaluate.py (scope + calibration)."""
    ev = [e for e in events if ce is not None and e[1] >= ce]
    if scope:
        ev, warns = apply_scope(ev, warns, scope)
    rows = score_run(ev, warns, tol)
    hours = sum(b - max(a, ce) for rg in scope.values() for a, b in rg if b > ce) / 3600 if scope and ce is not None else float(frames[-1]["t"]) / 3600
    return sum(r[5] == "hit" for r in rows), sum(r[5] != "false_alert" for r in rows), sum(r[5] == "false_alert" for r in rows), hours


def line(name, h, n, fa, hrs):
    lo, hi = wilson(h, n)
    return f"{name:<24} hits {h:>3}/{n:<3} = {h / n if n else 0:.3f} [{lo:.3f}, {hi:.3f}]  false alerts {fa:>3} = {fa / hrs if hrs else 0:6.1f}/h"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--scope")
    ap.add_argument("--calib-s", type=float, default=20.0)
    ap.add_argument("--closed-frac", type=float, nargs="+", default=[0.5])
    ap.add_argument("--fixed-thr", type=float, nargs="*", default=[], help="also try one absolute eye threshold for everyone (no per-driver eye calibration); applied to the arm's own signal (blink or EAR)")
    ap.add_argument("--tol", type=float, help="lenient labels (e.g. hand-labelled at 0.5 s steps): hit = warning from required moment - tol until event end + tol")
    ap.add_argument("--robust", action="store_true", help="also try the robust calibration (most-open 20 % of the window)")
    ap.add_argument("--loso", action="store_true", help="leave-one-subject-out choice of the setting (objective: hits - false alerts)")
    a = ap.parse_args()
    events, sc = defaultdict(list), defaultdict(lambda: defaultdict(list))
    for r in read(a.labels):
        events[r["clip"]].append((r["kind"], float(r["t_start"]), float(r["t_end"])))
    for r in read(a.scope) if a.scope else []:
        sc[r["clip"]][r["kind"]].append((float(r["t_start"]), float(r["t_end"])))
    settings = [("closed_frac", f) for f in a.closed_frac] + [("fixed_eye_thr", v) for v in a.fixed_thr] + ([("robust", True)] if a.robust else [])
    res = defaultdict(dict)  # (arm, clip) -> {setting: (h, n, fa, hours)}
    for ff in sorted(Path(a.runs).glob("*_frames.csv")):
        clip, variant = ff.name[: -len("_frames.csv")].rsplit("__", 1)
        eye, gaze = ("ear" if "_ear" in variant else "blend"), ("mobilegaze" if "mobilegaze" in variant else "heuristic")
        fr = read(ff)
        for f in settings:
            w, ce = replay(fr, eye, gaze, calib_s=a.calib_s, **dict([f]))
            res[(variant, clip)][f] = score(fr, w, ce, events.get(clip, []), sc[clip] if a.scope else None, a.tol)
    tot = lambda cells: [sum(c[i] for c in cells) for i in range(4)]
    for arm in sorted({k[0] for k in res}):
        clips = [k[1] for k in res if k[0] == arm]
        print(f"== {arm}")
        for f in settings:
            print(line(f"{f[0]} {f[1]}", *tot([res[(arm, c)][f] for c in clips])))
        if a.loso:
            subj = lambda c: c.split("_s")[0]
            pick, cells = {}, []
            for s in sorted({subj(c) for c in clips}):
                train = [c for c in clips if subj(c) != s]
                pick[s] = max(settings, key=lambda f: (lambda t: t[0] - t[2])(tot([res[(arm, c)][f] for c in train])))
                cells += [res[(arm, c)][pick[s]] for c in clips if subj(c) == s]
            print(line("LOSO-picked", *tot(cells)), " picks:", pick)


if __name__ == "__main__":
    if len(sys.argv) == 1:  # self-check: eyes closed 4 s after 20 s calibration -> microsleep + sleep warnings; a high closed_frac misses them
        hz, rows = 30, []
        for i in range(30 * hz):
            t = i / hz
            closed = 22 <= t < 26
            rows.append(dict(t=str(t), face="1", blink=str(0.6 if closed else 0.1), ear=str(0.15 if closed else 0.3), yaw="0", pitch="0",
                             eye_h="0", eye_v="0", eyes_unknown="0", face_fallback="0"))
        w, ce = replay(rows, "blend", "heuristic", calib_s=20, closed_frac=0.3)
        assert [k for _, k in w] == ["microsleep", "sleep"] and abs(ce - 20) < 0.1, w
        assert replay(rows, "blend", "heuristic", calib_s=20, closed_frac=0.6)[0] == []  # 0.6 < 0.1 + 0.6 * 0.9 = 0.64
        assert [k for _, k in replay(rows, "ear", "heuristic", calib_s=20, closed_frac=0.4)[0]] == ["microsleep", "sleep"]  # 0.15 < 0.6 * 0.3
        h, n, fa, _ = score(rows, w, ce, [("closed", 22.0, 26.0)], None)
        assert (n, fa) == (2, 0), (h, n, fa)
        print("replay OK")
        sys.exit()
    main()
