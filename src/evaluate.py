"""Score warnings vs hand labels: per-event outcome, per-variant and per-arm hit rate with Wilson CI, false alerts per hour, unknown-eye fraction.
--scope clip,kind,t_start,t_end (kind closed|away): only score where that kind has ground truth; warnings and events outside are ignored.
Events starting before the run's calibration ended (warnings are impossible then) are always dropped; counts are printed.
Run names are clip__variant[_clahe][_ear][_mobilegaze]: the optional _ear / _mobilegaze suffix is the arm, everything else the base variant."""
import argparse, csv, math, statistics
from collections import defaultdict
from pathlib import Path

from dms_core import MICROSLEEP_S, SLEEP_S, LONG_DISTRACTION_S, timing_error, within_tol

REQ = {"closed": {"microsleep": MICROSLEEP_S, "sleep": SLEEP_S}, "away": {"long_distraction": LONG_DISTRACTION_S}}
KIND_OF = {wk: ek for ek, d in REQ.items() for wk in d}  # warning kind -> label kind (scope key)
GRACE_S = 1.0  # a warning may fall up to this long after the event ends and still belong to it


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p, d = k / n, 1 + z * z / n
    c, h = p + z * z / (2 * n), z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d


def score_run(events, warns):
    """events: [(kind, t_start, t_end)], warns: [(t, kind)] -> rows (event, kind, t_expected, t_warn, err, outcome)."""
    rows, used = [], set()
    for i, (ek, t0, t1) in enumerate(events):
        for wk, req in REQ[ek].items():
            if t1 - t0 < req:
                continue
            te = t0 + req
            cand = [j for j, (t, k) in enumerate(warns) if k == wk and j not in used and t0 <= t <= t1 + GRACE_S]
            if not cand:
                rows.append((f"{ek}@{t0:g}", wk, te, None, None, "miss"))
                continue
            j = min(cand, key=lambda j: abs(warns[j][0] - te))
            used.add(j)
            err = timing_error(warns[j][0], t0, req)
            rows.append((f"{ek}@{t0:g}", wk, te, warns[j][0], err, "hit" if within_tol(err) else "late" if err > 0 else "early"))
    rows += [("", k, None, t, None, "false_alert") for j, (t, k) in enumerate(warns) if j not in used]
    return rows


def in_scope(ranges, t):
    return any(a <= t <= b for a, b in ranges)


def apply_scope(events, warns, scope):
    """scope: {kind: [(t0, t1)]} of one clip -> events starting inside their kind's scope, warnings inside theirs."""
    return ([e for e in events if in_scope(scope.get(e[0], []), e[1])], [w for w in warns if in_scope(scope.get(KIND_OF[w[1]], []), w[0])])


def calib_end(fr):
    """Time of the first classified frame; 0 if the frames CSV has no decisions column, inf if the run never calibrated."""
    return next((float(r["t"]) for r in fr if r["away"] != ""), math.inf) if fr and "away" in fr[0] else 0.0


def arm_of(variant):
    """'clean_clahe_ear' -> ('clean_clahe', 'ear'); no arm suffix -> arm 'default'."""
    t = variant.split("_")
    arm = [x for x in t if x in ("ear", "mobilegaze")]
    return "_".join(x for x in t if x not in arm), "_".join(arm) or "default"


def count(rows, col):
    """(rows with col == 1, rows where col is set); blank = calibrating / column absent."""
    v = [r[col] for r in rows if r.get(col, "") != ""]
    return sum(x == "1" for x in v), len(v)


def pooled(rr, key):
    """Pooled fraction over run dicts (frame-weighted); None if no frames."""
    k, n = sum(r[key][0] for r in rr), sum(r[key][1] for r in rr)
    return k / n if n else None


def read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def stats(rv, rr):
    """rv: result rows, rr: run dicts of the same group -> hits, expected, rate, ci_lo, ci_hi, median |err|, false alerts, hours, false alerts/h, unknown frac, fallback frac."""
    n, k = sum(r[8] != "false_alert" for r in rv), sum(r[8] == "hit" for r in rv)
    lo, hi = wilson(k, n)
    ae = [abs(r[7]) for r in rv if r[7] is not None]
    fa, h = sum(r[8] == "false_alert" for r in rv), sum(r["hours"] for r in rr)
    return [k, n, k / n if n else float("nan"), lo, hi, statistics.median(ae) if ae else None, fa, h, fa / h if h else float("nan"),
            pooled(rr, "unk"), pooled(rr, "fb")]


def main(labels, runs_dir, out, scope=None):
    events, sc = defaultdict(list), defaultdict(lambda: defaultdict(list))
    for r in read(labels):
        events[r["clip"]].append((r["kind"], float(r["t_start"]), float(r["t_end"])))
    for r in read(scope) if scope else []:
        sc[r["clip"]][r["kind"]].append((float(r["t_start"]), float(r["t_end"])))
    res, runs, drop = [], [], defaultdict(int)
    for wf in sorted(Path(runs_dir).glob("*_warnings.csv")):
        run = wf.name[: -len("_warnings.csv")]
        clip, variant = run.rsplit("__", 1)
        warns = sorted((float(r["t"]), r["kind"]) for r in read(wf))
        ff = Path(runs_dir) / f"{run}_frames.csv"
        fr = read(ff) if ff.exists() else []
        ev, ce = events.get(clip, []), calib_end(fr)
        drop["calibration"] += sum(e[1] < ce for e in ev)
        ev = [e for e in ev if e[1] >= ce]
        hours = (float(fr[-1]["t"]) if fr else 0) / 3600
        if scope:  # false alerts/h over the scored time only: seconds with ground truth for some kind
            n, w = len(ev), len(warns)
            ev, warns = apply_scope(ev, warns, sc[clip])
            drop["scope events"] += n - len(ev)
            drop["scope warnings"] += w - len(warns)
            hours = sum(b - max(a, ce) for rg in sc[clip].values() for a, b in rg if b > ce) / 3600
        runs.append(dict(run=run, clip=clip, variant=variant, base=arm_of(variant)[0], arm=arm_of(variant)[1], frames=len(fr),
                         hours=hours, unk=count(fr, "eyes_unknown"), fb=count(fr, "face_fallback")))
        res += [(run, clip, variant) + r for r in score_run(ev, warns)]
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cell = lambda v: "" if v is None else f"{v:.3f}" if isinstance(v, float) else v
    wr = lambda name, head, rows: csv.writer(open(out / name, "w", newline="")).writerows([head.split(",")] + [[cell(v) for v in r] for r in rows])
    wr("results.csv", "run,clip,variant,event,kind,t_expected,t_warn,err,outcome", res)
    wr("runs.csv", "run,clip,variant,arm,frames,hours,eyes_unknown_frac,face_fallback_frac",
       [[r["run"], r["clip"], r["variant"], r["arm"], r["frames"], r["hours"], pooled([r], "unk"), pooled([r], "fb")] for r in runs])
    pick = lambda g: ([x for x in res if x[2] == g[0]], [r for r in runs if r["variant"] == g[0]])
    summ = []
    for v in sorted({r["variant"] for r in runs}):
        rv, rr = pick((v,))
        st = stats(rv, rr)
        summ.append([v] + st[:5] + st[5:6] + st[6:])
    head = "variant,hits,expected,rate,ci_lo,ci_hi,median_abs_err,false_alerts,hours,false_alerts_per_h,eyes_unknown_frac,face_fallback_frac"
    wr("summary.csv", head, summ)
    # arms: pooled over all base variants, but only over (clip, base) pairs that every arm was run on, so arms see the same clips
    keys = {a: {(r["clip"], r["base"]) for r in runs if r["arm"] == a} for a in {r["arm"] for r in runs}}
    common = set.intersection(*keys.values()) if keys else set()
    arms = []
    for a in sorted(keys, key=lambda a: (a != "default", a)):
        rr = [r for r in runs if r["arm"] == a and (r["clip"], r["base"]) in common]
        names = {r["run"] for r in rr}
        arms.append([a] + stats([x for x in res if x[0] in names], rr) + [len(rr)])
    wr("arms.csv", "arm,hits,expected,rate,ci_lo,ci_hi,median_abs_err,false_alerts,hours,false_alerts_per_h,eyes_unknown_frac,face_fallback_frac,runs", arms)
    print(head.replace(",", "\t"))
    for s_ in summ:
        print("\t".join(map(str, [cell(v) for v in s_])))
    print("ignored (all runs):", dict(drop))
    if len(arms) > 1:
        print("arm comparison (same clips x variants):")
        for a in arms:
            print("\t".join(map(str, [cell(v) for v in a])))


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:  # self-check
        lo, hi = wilson(17, 20)
        assert abs(lo - 0.640) < 0.01 and abs(hi - 0.948) < 0.01, (lo, hi)
        ev = [("closed", 10.0, 15.0), ("away", 20.0, 26.0), ("closed", 30.0, 31.0), ("away", 40.0, 45.0)]
        wr = [(11.55, "microsleep"), (13.4, "sleep"), (23.1, "long_distraction"), (50.0, "microsleep")]
        got = {(r[0], r[1]): r for r in score_run(ev, wr)}
        assert got[("closed@10", "microsleep")][5] == "hit"
        assert got[("closed@10", "sleep")][5] == "late" and abs(got[("closed@10", "sleep")][4] - 0.4) < 1e-9
        assert got[("away@20", "long_distraction")][5] == "hit"
        assert got[("away@40", "long_distraction")][5] == "miss"
        assert [r[5] for r in score_run(ev, wr) if r[5] == "false_alert"] == ["false_alert"]
        assert len(score_run(ev, wr)) == 5  # 1.0 s closure expects nothing
        assert arm_of("clean") == ("clean", "default") and arm_of("clean_clahe_ear") == ("clean_clahe", "ear")
        assert arm_of("sunglasses_mobilegaze") == ("sunglasses", "mobilegaze") and arm_of("fps25_ear_mobilegaze") == ("fps25", "ear_mobilegaze")
        assert count([{"eyes_unknown": ""}, {"eyes_unknown": "0"}, {"eyes_unknown": "1"}, {}], "eyes_unknown") == (1, 2)
        assert pooled([dict(unk=(1, 2)), dict(unk=(0, 2))], "unk") == 0.25 and pooled([dict(unk=(0, 0))], "unk") is None
        # end to end on synthetic runs: arm suffixes become separate variants, arms pooled over common (clip, base) only
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "runs").mkdir()
            (d / "labels.csv").write_text("clip,kind,t_start,t_end\nc,closed,10,15\n")
            for run, warn, unk in [("c__clean", "microsleep", 0), ("c__clean_ear", "microsleep", 0), ("c__ir_ear", None, 1)]:
                (d / "runs" / f"{run}_warnings.csv").write_text("t,kind\n" + (f"11.5,{warn}\n" if warn else ""))
                (d / "runs" / f"{run}_frames.csv").write_text("t,eyes_unknown,face_fallback\n0,,0\n36,%d,0\n72,%d,1\n" % (unk, unk))
            main(str(d / "labels.csv"), str(d / "runs"), str(d))
            sm = {r["variant"]: r for r in read(d / "summary.csv")}
            assert set(sm) == {"clean", "clean_ear", "ir_ear"} and sm["ir_ear"]["eyes_unknown_frac"] == "1.000" and sm["clean"]["eyes_unknown_frac"] == "0.000"
            ar = {r["arm"]: r for r in read(d / "arms.csv")}
            assert ar["default"]["runs"] == "1" and ar["ear"]["runs"] == "1" and ar["ear"]["hits"] == "1" and ar["ear"]["face_fallback_frac"] == "0.333", ar  # ir_ear has no default twin
        # scope + calibration: closed@1 starts during calibration (dropped); closed@10 is in scope; the 3rd closure is out of scope; away has no scope
        assert apply_scope([("closed", 10, 15), ("closed", 50, 55), ("away", 10, 15)], [(11.5, "microsleep"), (60, "sleep"), (12, "long_distraction")],
                           {"closed": [(0, 20)]}) == ([("closed", 10, 15)], [(11.5, "microsleep")])
        assert calib_end([{"t": "0", "away": ""}, {"t": "2.04", "away": "0"}]) == 2.04 and calib_end([{"t": "0", "away": ""}]) == math.inf and calib_end([{"t": "0"}]) == 0
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "runs").mkdir()
            (d / "labels.csv").write_text("clip,kind,t_start,t_end\nc,closed,1,5\nc,closed,10,15\nc,closed,50,55\n")
            (d / "scope.csv").write_text("clip,kind,t_start,t_end\nc,closed,0,20\n")
            (d / "runs" / "c__clean_warnings.csv").write_text("t,kind\n11.5,microsleep\n60,microsleep\n")
            (d / "runs" / "c__clean_frames.csv").write_text("t,away\n0,\n2,0\n3600,0\n")
            main(str(d / "labels.csv"), str(d / "runs"), str(d), str(d / "scope.csv"))
            r = read(d / "summary.csv")[0]
            assert (r["hits"], r["expected"], r["false_alerts"]) == ("1", "2", "0") and abs(float(r["hours"]) - 18 / 3600) < 1e-3, r  # microsleep hit, sleep miss; 18 s scored (calibration 2 s excluded)
        print("evaluate OK")
        sys.exit()
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/clips/labels.csv")
    ap.add_argument("--runs", default="out/runs")
    ap.add_argument("--out", default="out")
    ap.add_argument("--scope", help="CSV clip,kind,t_start,t_end: score only where ground truth exists")
    a = ap.parse_args()
    main(a.labels, a.runs, a.out, a.scope)
