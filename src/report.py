"""PDF test report from out/results.csv, out/summary.csv and the per-run CSVs. Usage: report.py --out out"""
import argparse, csv, textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from dms_core import TOL_S
from evaluate import arm_of

A4 = (8.27, 11.69)
COL = {"closed": "tab:blue", "away": "tab:orange"}
LIMITS = """Self-recorded tiny dataset: one subject, a phone camera, RGB (not IR). The number of events per variant is small, so the
Wilson intervals are wide; a hit rate here is a functional check of the pipeline, not an estimate for other people or cars.

The variants (ir, glare, dark, sunglasses, mask, fps25) are image effects applied to the same clips, not newly rendered or
newly recorded people. Real IR sunglasses, real mask fit and real night cabins will differ (domain gap).

Labels are made by hand by stepping through video frames, so label error is about one frame (0.04 s at 25 Hz). The tolerance of
%.1f s is therefore only slightly larger than the label and frame uncertainty.

Thresholds (microsleep 1.5 s, sleep 3.0 s, long distraction 3.0 s) are the values chosen in this project inside the SD-202
ranges; they are not those of any product. Euro NCAP scoring, other scenarios (owl/lizard glances, unresponsive driver) and
vehicle intervention are not covered.

This is a test bench built for learning and demonstration. It is not a validation of a product and makes no product claim.""" % TOL_S


def read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def num(x):
    return float(x) if x not in ("", None) else float("nan")


def text_page(pdf, title, body, size=9.5):
    fig = plt.figure(figsize=A4)
    fig.text(0.08, 0.94, title, fontsize=16, weight="bold", va="top")
    paras = [textwrap.fill(" ".join(p.split()), 95) for p in body.split("\n\n")]
    fig.text(0.08, 0.89, "\n\n".join(paras), fontsize=size, va="top", linespacing=1.4)
    pdf.savefig(fig)
    plt.close(fig)


def timeline(pdf, run, frames, warns, events):
    t = [num(r["t"]) for r in frames]
    fig, ax = plt.subplots(2, 1, figsize=A4, sharex=True)
    fig.suptitle(f"Timeline: {run}", fontsize=12, weight="bold")
    ax[0].plot(t, [num(r["blink"]) for r in frames], lw=1, color="k")
    ax[0].set_ylabel("blink score")
    ax[1].plot(t, [num(r["away"]) for r in frames], lw=1.2, color="k", drawstyle="steps-post")
    ax[1].set_ylabel("away (0/1)")
    ax[1].set_xlabel("video time (s)")
    for kind, t0, t1 in events:
        for a in ax:
            a.axvspan(t0, t1, color=COL[kind], alpha=0.2)
    for tw, kind in warns:
        for a in ax:
            a.axvline(tw, color="tab:red", lw=1.2, ls="--")
        ax[0].text(tw, ax[0].get_ylim()[1], kind, rotation=90, va="top", ha="right", fontsize=7, color="tab:red")
    ax[0].legend(handles=[plt.Rectangle((0, 0), 1, 1, color=COL[k], alpha=0.3) for k in COL] + [plt.Line2D([], [], color="tab:red", ls="--")],
                 labels=["labeled closed", "labeled away", "warning"], fontsize=7, loc="upper right")
    pdf.savefig(fig)
    plt.close(fig)


def main(out, labels):
    out = Path(out)
    summ, res = read(out / "summary.csv"), read(out / "results.csv")
    ev = {}
    for r in read(labels) if Path(labels).exists() else []:
        ev.setdefault(r["clip"], []).append((r["kind"], float(r["t_start"]), float(r["t_end"])))
    with PdfPages(out / "report.pdf") as pdf:
        # page 1: title, method, table
        fig = plt.figure(figsize=A4)
        fig.text(0.08, 0.94, "In-cabin DMS test bench: SD-202-style warning timing", fontsize=16, weight="bold", va="top")
        method = ("Method: MediaPipe Face Landmarker blendshapes (eye blink, head pose) with per-driver calibration give per-frame "
                  "'eyes closed' and 'gaze off road'. SD-202 timers (microsleep 1.5 s, sleep 3.0 s, long distraction 3.0 s) raise warnings. "
                  f"Each warning is compared with hand-labeled events; a warning is a hit if it is within {TOL_S:g} s of the required time. "
                  "Intervals are Wilson 95%. False alerts are warnings that match no expected warning. Eyes are 'unknown' (not counted as closed) when the eye region "
                  "is much darker than the forehead (dark lenses) or MediaPipe lost the face while YuNet still sees it. Arms (_ear = EAR instead of the blink blendshape, "
                  "_mobilegaze = MobileGaze instead of head pose + eyeLook) are separate rows and compared on the arm comparison page.")
        fig.text(0.08, 0.90, textwrap.fill(method, 100), fontsize=9, va="top", linespacing=1.4)
        ax = fig.add_axes([0.06, 0.45, 0.88, 0.35])
        ax.axis("off")
        f = lambda x, d=2: "-" if x == "" else f"{float(x):.{d}f}"
        rows = [[s["variant"], f"{s['hits']}/{s['expected']}", f"{f(s['rate'])} [{f(s['ci_lo'])}, {f(s['ci_hi'])}]", f(s["median_abs_err"]),
                 s["false_alerts"], f(s["false_alerts_per_h"], 1), f(s.get("eyes_unknown_frac", ""))] for s in summ]
        tb = ax.table(cellText=rows or [["no runs"] + [""] * 6], colLabels=["variant", "hits/expected", "hit rate [95% CI]", "median |err| s", "false alerts", "false alerts/h", "eyes unknown"],
                      loc="upper center", cellLoc="center")
        tb.auto_set_font_size(False)
        tb.set_fontsize(8)
        tb.scale(1, 1.5)
        pdf.savefig(fig)
        plt.close(fig)
        # page 2: hit rate with Wilson bars + false alerts per hour
        fig, ax = plt.subplots(2, 1, figsize=A4)
        x = list(range(len(summ)))
        rate, lo, hi = ([num(s[k]) for s in summ] for k in ("rate", "ci_lo", "ci_hi"))
        ax[0].bar(x, rate, color="tab:blue", alpha=0.7, yerr=[[r - l for r, l in zip(rate, lo)], [h - r for r, h in zip(rate, hi)]], capsize=4)
        ax[0].set(ylim=(0, 1.05), ylabel="hit rate (Wilson 95% CI)", title="Warnings within tolerance per variant")
        ax[1].bar(x, [num(s["false_alerts_per_h"]) for s in summ], color="tab:red", alpha=0.7)
        ax[1].set(ylabel="false alerts per hour", title="False alerts (clip time is short: the rate is very noisy)")
        for a in ax:
            a.set_xticks(x, [s["variant"] for s in summ], rotation=30, ha="right")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
        # page 3: arms side by side (pooled over the same clips x variants)
        arms = read(out / "arms.csv") if (out / "arms.csv").exists() else []
        if len(arms) > 1:
            fig, ax = plt.subplots(3, 1, figsize=A4)
            x = list(range(len(arms)))
            rate, lo, hi = ([num(a[k]) for a in arms] for k in ("rate", "ci_lo", "ci_hi"))
            ax[0].bar(x, rate, color="tab:blue", alpha=0.7, yerr=[[r - l for r, l in zip(rate, lo)], [h - r for r, h in zip(rate, hi)]], capsize=4)
            ax[0].set(ylim=(0, 1.05), ylabel="hit rate (Wilson 95% CI)", title="Arms on the same clips and variants: warnings within tolerance")
            ax[1].bar(x, [num(a["false_alerts_per_h"]) for a in arms], color="tab:red", alpha=0.7)
            ax[1].set(ylabel="false alerts per hour", title="False alerts (very noisy on short clips)")
            ax[2].bar(x, [num(a["eyes_unknown_frac"]) for a in arms], color="tab:gray", alpha=0.7)
            ax[2].set(ylim=(0, 1.05), ylabel="fraction of frames", title="Eyes unknown")
            for a_ in ax:
                a_.set_xticks(x, [f"{a['arm']} (n={a['hits']}/{a['expected']})" for a in arms], rotation=20, ha="right")
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
        # one timeline per clean run (each arm separately)
        for run in sorted({r["run"] for r in res if arm_of(r["variant"])[0] == "clean"}):
            clip = run.rsplit("__", 1)[0]
            ff, wf = out / "runs" / f"{run}_frames.csv", out / "runs" / f"{run}_warnings.csv"
            if ff.exists() and wf.exists():
                timeline(pdf, run, read(ff), [(float(r["t"]), r["kind"]) for r in read(wf)], ev.get(clip, []))
        text_page(pdf, "Feasibility and limitations", LIMITS)
    print(out / "report.pdf")


if __name__ == "__main__":
    import sys
    if "--selfcheck" in sys.argv:  # synthetic runs for two arms -> evaluate -> report.pdf with the arm page (table, variants, arms, 2 timelines, limits)
        import tempfile
        from evaluate import main as evaluate
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "runs").mkdir()
            (d / "labels.csv").write_text("clip,kind,t_start,t_end" + chr(10) + "c,closed,2,6" + chr(10))
            for run, unk in [("c__clean", 0), ("c__clean_ear", 1)]:
                (d / "runs" / f"{run}_warnings.csv").write_text("t,kind" + chr(10) + "3.5,microsleep" + chr(10))
                (d / "runs" / f"{run}_frames.csv").write_text("t,blink,away,eyes_unknown,face_fallback" + chr(10) + "".join(f"{i / 25},0.1,0,{unk},0" + chr(10) for i in range(250)))
            evaluate(str(d / "labels.csv"), str(d / "runs"), str(d))
            main(str(d), str(d / "labels.csv"))
            assert (d / "report.pdf").stat().st_size > 5000 and len(read(d / "arms.csv")) == 2
        print("report OK")
        sys.exit()
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out")
    ap.add_argument("--labels", default="data/clips/labels.csv")
    a = ap.parse_args()
    main(a.out, a.labels)
