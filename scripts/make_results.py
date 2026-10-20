"""Collect publishable results from out/ into results/: summary CSVs, figures and a demo GIF.
Only aggregates and synthetic/simulator material: no DMD frames or per-event DMD labels (CC BY-NC-ND)."""
import csv
import shutil
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from evaluate import read, wilson  # noqa: E402

OUT, RES = ROOT / "out", ROOT / "results"
COPY = {"carla_fcw.csv": "carla_fcw.csv", "carla_fcw.png": "carla_fcw.png", "synthetic/summary.csv": "synthetic_summary.csv",
        "synthetic/report.pdf": "synthetic_report.pdf", "dmd/summary.csv": "dmd_summary.csv", "dmd/arms.csv": "dmd_arms.csv",
        "fm/eye_probe.csv": "foundation_eye_probe.csv", "vlm/summary.csv": "vlm_summary.csv"}
# DMD measurement steps, default arm: runs in out/dmd_baseline, out/dmd_scope, out/dmd; step 4 = scripts/replay.py on close-only labels
DMD_STEPS = [("first run", 7, 66, 107), ("+ scoring scope", 7, 62, 18.9), ("+ 20 s calibration", 10, 58, 9.7), ("+ lids-shut labels", 22, 49, 19.4)]


def dmd_steps(path):
    fig, ax = plt.subplots(figsize=(7, 3.2))
    for i, (name, k, n, fa) in enumerate(DMD_STEPS):
        lo, hi = wilson(k, n)
        ax.bar(i, k / n, color="#4C72B0")
        ax.errorbar(i, k / n, yerr=[[k / n - lo], [hi - k / n]], color="k", capsize=4)
        ax.text(i, hi + 0.02, f"{k}/{n}\n{fa:g} FA/h", ha="center", fontsize=8)
    ax.set_xticks(range(len(DMD_STEPS)), [s[0] for s in DMD_STEPS], fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("warnings on time (±0.2 s)")
    ax.set_title("DMD real drivers: fixing the measurement, not the model (95 % Wilson CI)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def live_escalation(src, path):
    r = read(src)
    t, v = [float(x["t"]) for x in r], [float(x["speed_kmh"]) for x in r]
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(t, v, color="#4C72B0")
    prev = None
    for x in r:
        if x["level"] != prev:
            ax.axvline(float(x["t"]), color="grey", ls=":", lw=0.8)
            ax.text(float(x["t"]) + 0.2, 97, f"{x['level']} {float(x['t']):g}s", rotation=90, fontsize=7, va="top")
            prev = x["level"]
    ax.axvspan(2, t[-1], color="#DD8452", alpha=0.12, label="eyes closed from 2 s (simulated driver)")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("CARLA ego speed (km/h)")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=7, loc="lower left")
    ax.set_title("Escalation in CARLA: warning -> slowing -> minimal-risk stop", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def demo_gif(video, frames_csv, warns_csv, path, fps=8, width=360):
    """Synthetic clip with the pipeline's per-frame decision and warnings burnt in."""
    fr, warns = read(frames_csv), [(float(w["t"]), w["kind"]) for w in read(warns_csv)]
    cap, imgs, i, step = cv2.VideoCapture(str(video)), [], 0, None
    step = max(1, round(cap.get(cv2.CAP_PROP_FPS) / fps))
    while True:
        ok, img = cap.read()
        if not ok or i >= len(fr):
            break
        if i % step == 0:
            r, t = fr[i], float(fr[i]["t"])
            img = cv2.resize(img, (width, round(img.shape[0] * width / img.shape[1])))
            img[:66] = (img[:66] * 0.35).astype(img.dtype)  # dark band so the text stays readable
            state = "calibrating" if r["closed"] == "" else f"closed={'yes' if r['closed'] == '1' else 'no'}  away={'yes' if r['away'] == '1' else 'no'}"
            cv2.putText(img, f"t={t:4.1f}s  {state}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            for j, (tw, k) in enumerate([w for w in warns if w[0] <= t][-2:]):
                cv2.putText(img, f"WARNING {k} @ {tw:.1f}s", (8, 44 + 20 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            imgs.append(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).quantize(64))
        i += 1
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=1000 // fps, loop=0, optimize=True)
    return len(imgs)


def cosmos_panel(d, path, sec=2.5):
    """One frame of each CARLA control video (RGB, depth, segmentation) side by side."""
    tiles = []
    for name in ("rgb", "depth", "semantic_segmentation"):
        cap = cv2.VideoCapture(str(d / f"{name}.mp4"))
        cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
        ok, img = cap.read()
        if not ok:
            return
        img = cv2.resize(img, (426, 240))
        cv2.putText(img, name.replace("_", " "), (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        tiles.append(img)
    cv2.imwrite(str(path), cv2.hconcat(tiles))


def live_gif(video, frames_csv, labels_csv, path, t0, t1, fps=6, width=600):
    """Overlay drawn at GIF size (sharp text), [t0, t1] s of a recorded run, adaptive 256-colour palette per frame."""
    from render_overlay import draw, truth_at
    fr = read(frames_csv)
    warns = [(float(w["t"]), w["kind"]) for w in read(Path(frames_csv).with_name(Path(frames_csv).name.replace("_frames", "_warnings")))]
    events = [(x["kind"], float(x["t_start"]), float(x["t_end"])) for x in read(labels_csv)]
    cap, imgs, last = cv2.VideoCapture(str(video)), [], -1.0
    for r in fr:
        ok, img = cap.read()
        t = float(r["t"])
        if not ok or t > t1:
            break
        if t >= t0 and t - last >= 1 / fps:
            last = t
            img = cv2.resize(img, (width, round(img.shape[0] * width / img.shape[1])), interpolation=cv2.INTER_AREA)
            img = draw(img, r, t, warns, truth_at(events, t))
            imgs.append(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
    pal = imgs[len(imgs) // 2].quantize(192)  # one shared palette: frames differ little, so GIF frame deltas stay small
    imgs = [im.quantize(palette=pal, dither=Image.Dither.NONE) for im in imgs]
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=1000 // fps, loop=0, optimize=True)


def main():
    RES.mkdir(exist_ok=True)
    for src, dst in COPY.items():
        if (OUT / src).exists():
            shutil.copy(OUT / src, RES / dst)
    dmd_steps(RES / "dmd_steps.png")
    live = sorted(OUT.glob("live_*.csv"))
    if live:
        live_escalation(live[-1], RES / "live_escalation.png")
    cosmos_panel(OUT / "cosmos", RES / "carla_control_videos.png")
    live = ROOT / "data/clips/clip05_live.mp4"  # own face, phone camera, voice-guided take, hand-labelled truth (publishable, unlike DMD)
    if live.exists():
        live_gif(live, OUT / "live_runs5/camera__clean_frames.csv", OUT / "live_runs5/labels_hand.csv", RES / "demo_live_phone.gif", 21, 51)
    clip = "synth02_sleep"
    n = demo_gif(ROOT / "data/synthetic" / f"{clip}.mp4", OUT / "synthetic/runs" / f"{clip}__clean_frames.csv",
                 OUT / "synthetic/runs" / f"{clip}__clean_warnings.csv", RES / "demo_synthetic_sleep.gif")
    for p in sorted(RES.iterdir()):
        print(f"{p.name:32} {p.stat().st_size / 1024:8.0f} KiB")
    print(f"gif frames: {n}")


if __name__ == "__main__":
    main()
