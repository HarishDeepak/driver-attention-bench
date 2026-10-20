"""From measured DMS accuracy to crash outcomes: Monte Carlo over driver reaction times for a distracted driver approaching a
stopped car (same kinematics and thresholds as scripts/carla_fcw.py). The collision warning fires at TTC 2.0 s, or 3.0 s when the
DMS reports the driver inattentive; the DMS catches the inattention with probability p (its measured event-level hit rate).
Output: out/dms_safety_mc.csv + results/dms_safety_mc.png. ALL DRIVER NUMBERS ARE ASSUMPTIONS (see RT_MEDIAN, RT_SIGMA)."""
import csv
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
THR_FIXED, THR_GATED, A_BRAKE = 2.0, 3.0, 8.0  # as carla_fcw.py
RT_MEDIAN, RT_SIGMA = 1.6, 0.3  # distracted driver's reaction after the warning: lognormal, median 1.6 s (assumption, not measured)
N = 20000
# p = share of inattention episodes the DMS catches. 0.61: DMD event-level hit rate (EAR arm, lids-shut labels, E3) as a conservative
# proxy; 1.0: ideal (the live take caught 4/4, one take only)
CASES = {"no DMS (fixed warning)": 0.0, "DMS, measured on DMD (p = 0.61)": 0.61, "ideal DMS (p = 1.0)": 1.0}


def impact_kmh(v_kmh, thr, rt):
    """Stopped target, warning at TTC thr, driver brakes at A_BRAKE after rt -> impact speed in km/h (0 = stops in time)."""
    v = v_kmh / 3.6
    gap = v * thr - v * rt  # gap left when braking starts
    if gap <= 0:
        return v_kmh
    return math.sqrt(max(v * v - 2 * A_BRAKE * gap, 0.0)) * 3.6


def simulate(v_kmh, p, rng):
    rt = rng.lognormal(math.log(RT_MEDIAN), RT_SIGMA, N)
    thr = np.where(rng.random(N) < p, THR_GATED, THR_FIXED)
    imp = np.array([impact_kmh(v_kmh, t, r) for t, r in zip(thr, rt)])
    return (imp > 0).mean(), imp.mean()


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rng, speeds, rows = np.random.default_rng(0), list(range(30, 110, 10)), []
    for name, p in CASES.items():
        for v in speeds:
            crash, imp = simulate(v, p, rng)
            rows.append((name, p, v, round(crash, 4), round(imp, 2)))
    (ROOT / "out").mkdir(exist_ok=True)
    with open(ROOT / "out" / "dms_safety_mc.csv", "w", newline="") as f:
        csv.writer(f).writerows([("case", "p_detect", "speed_kmh", "crash_prob", "mean_impact_kmh")] + rows)
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.4))
    for name in CASES:
        r = [x for x in rows if x[0] == name]
        a.plot([x[2] for x in r], [x[3] for x in r], marker="o", label=name)
        b.plot([x[2] for x in r], [x[4] for x in r], marker="o", label=name)
    a.set(xlabel="speed (km/h)", ylabel="crash probability", ylim=(0, 1.02), title="Distracted driver, stopped car ahead")
    b.set(xlabel="speed (km/h)", ylabel="mean impact speed (km/h)", title="Mean impact speed (0 = stopped in time)")
    a.legend(fontsize=7)
    for ax in (a, b):
        ax.title.set_fontsize(9)
    fig.suptitle(f"Monte Carlo, {N} drivers per point: reaction lognormal (median {RT_MEDIAN} s), braking {A_BRAKE} m/s², "
                 "warning TTC 2.0 s or 3.0 s if the DMS flags inattention", fontsize=8)
    fig.tight_layout()
    (ROOT / "results").mkdir(exist_ok=True)
    fig.savefig(ROOT / "results" / "dms_safety_mc.png", dpi=130)
    for r in rows:
        print(f"{r[0]:<34} {r[2]:>4} km/h  crash {r[3]:.2f}  mean impact {r[4]:5.1f} km/h")


if __name__ == "__main__":
    if len(sys.argv) == 1 or sys.argv[1] != "--selfcheck":
        main()
        sys.exit()
    # same case as the CARLA sweep: 30 km/h, inattentive, fixed warning -> CARLA impact 14.2 km/h
    assert abs(impact_kmh(30, 2.0, 1.6) - 14.2) < 0.5, impact_kmh(30, 2.0, 1.6)
    assert impact_kmh(30, 3.0, 1.6) == 0 and impact_kmh(50, 2.0, 2.5) == 50
    c0, _ = simulate(50, 0.0, np.random.default_rng(1))
    c1, _ = simulate(50, 1.0, np.random.default_rng(1))
    assert c1 < c0, (c0, c1)
    print("dms_safety_mc OK")
