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
        "fm/eye_probe.csv": "foundation_eye_probe.csv", "fm/ir_eye_probe.csv": "infrared_eye_probe.csv", "vlm/summary.csv": "vlm_summary.csv"}
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
            img = draw(img, r, t, warns, truth_at(events, t), "ear" if "_ear" in Path(frames_csv).name else "blink")
            imgs.append(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
    pal = imgs[len(imgs) // 2].quantize(192)  # one shared palette: frames differ little, so GIF frame deltas stay small
    imgs = [im.quantize(palette=pal, dither=Image.Dither.NONE) for im in imgs]
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=1000 // fps, loop=0, optimize=True)


def ir_chart(src, path):
    """Two panels from out/fm/ir_eye_probe.csv: day-RGB vs IR training, and the IR-trained probe by condition."""
    r = read(src)
    ba = lambda m, s="all": float(next(x["bal_acc"] for x in r if x["method"] == m and x["subset"] == s))
    left = [("CLIP\nzero-shot", ba("(a) CLIP zero-shot"), "#999999"),
            ("DINOv2\ntrained on\nday RGB", ba("(b) DINOv2-small, DMD-RGB trained"), "#DD8452"),
            ("CLIP\ntrained on\nday RGB", ba("(b) CLIP ViT-B/32, DMD-RGB trained"), "#DD8452"),
            ("DINOv2\ntrained on\nIR", ba("(c) DINOv2-small, MRL LOSO"), "#4C72B0"),
            ("CLIP\ntrained on\nIR", ba("(c) CLIP ViT-B/32, MRL LOSO"), "#4C72B0")]
    m = "(d) (c) CLIP ViT-B/32, MRL LOSO"
    right = [(f"{k.split('=')[0]}\n{k.split('=')[1]}", ba(m, k)) for k in
             ("lighting=good", "lighting=bad", "glasses=no", "glasses=yes", "reflections=none", "reflections=big")]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.4), gridspec_kw=dict(width_ratios=[5, 6]))
    for i, (name, v, c) in enumerate(left):
        a.bar(i, v, color=c)
        a.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    a.set_xticks(range(len(left)), [x[0] for x in left], fontsize=7)
    a.set_title("Real infrared eyes, people never seen in training", fontsize=9)
    for i, (name, v) in enumerate(right):
        b.bar(i, v, color="#4C72B0" if i % 2 == 0 else "#55A868")
        b.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    b.set_xticks(range(len(right)), [x[0] for x in right], fontsize=7)
    b.set_title("IR-trained CLIP probe by condition", fontsize=9)
    for ax in (a, b):
        ax.set_ylim(0.4, 1.02)
        ax.axhline(0.5, color="grey", lw=0.6, ls=":")
    a.set_ylabel("balanced accuracy (0.5 = guessing)")
    fig.suptitle("MRL Eye dataset: 16,920 infrared eye images, 37 people, leave-one-person-out", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def clip_gif(video, path, t0, t1, fps=6, width=760):
    """[t0, t1] s of an already-rendered video (e.g. the face + CARLA side-by-side) -> GIF with one shared palette."""
    cap = cv2.VideoCapture(str(video))
    src, imgs, i = cap.get(cv2.CAP_PROP_FPS), [], 0
    while True:
        ok, img = cap.read()
        if not ok or i / src > t1:
            break
        if i / src >= t0 and i % max(1, round(src / fps)) == 0:
            img = cv2.resize(img, (width, round(img.shape[0] * width / img.shape[1])), interpolation=cv2.INTER_AREA)
            imgs.append(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
        i += 1
    pal = imgs[len(imgs) // 2].quantize(192)
    imgs = [im.quantize(palette=pal, dither=Image.Dither.NONE) for im in imgs]
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=1000 // fps, loop=0, optimize=True)


FULL = {  # full-length videos next to the short README GIFs (all publishable: own face, synthetic, simulator, NITYMED CC BY 4.0)
    "videos/clip05_live_overlay.mp4": "live_phone_take5.mp4",
    "videos/carla_real_face_take5.mp4": "face_to_carla_take5.mp4",
    "videos/carla_escalation_sim.mp4": "carla_minimal_risk_manoeuvre.mp4",
    "videos/male_P1042793_720__clean_ear.mp4": "night_nitymed_P1042793.mp4",
    "videos/synth02_sleep__clean.mp4": "synthetic_sleep.mp4",
}


def full_videos():
    """Re-encode to H.264 (browser-playable, small) with ffmpeg into results/videos/; DMD videos are never included."""
    import shutil, subprocess
    ff = shutil.which("ffmpeg")
    if not ff:
        print("ffmpeg not found: full videos skipped")
        return
    (RES / "videos").mkdir(exist_ok=True)
    for src, dst in FULL.items():
        if (OUT / src).exists():
            subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(OUT / src), "-vf", "scale='min(960,iw)':-2", "-c:v", "libx264",
                            "-crf", "30", "-preset", "slow", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(RES / "videos" / dst)], check=True)


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
    sbs = OUT / "videos/carla_real_face_take5.mp4"  # take 5 (own face) driving the CARLA ego: live_demo.py <clip> --video
    if sbs.exists():
        clip_gif(sbs, RES / "demo_face_to_carla.gif", 22, 40, fps=5)
    night = ROOT / "data/nitymed/microsleep/male_P1042793_720.mp4"  # NITYMED (CC BY 4.0, Petrellis et al.): EAR arm, robust calibration
    if night.exists():
        live_gif(night, OUT / "nitymed/robust/male_P1042793_720__clean_ear_frames.csv", OUT / "nitymed/labels_hand.csv",
                 RES / "demo_night_nitymed.gif", 19, 39)
    stop = OUT / "videos/carla_escalation_sim.mp4"  # simulated sleeping driver: live_demo.py sim:2-40 --video
    if stop.exists():
        clip_gif(stop, RES / "demo_carla_safe_stop.gif", 12, 27, fps=5, width=760)
    if (OUT / "fm/ir_eye_probe.csv").exists():
        ir_chart(OUT / "fm/ir_eye_probe.csv", RES / "infrared_eye_state.png")
    live = ROOT / "data/clips/clip05_live.mp4"  # own face, phone camera, voice-guided take, hand-labelled truth (publishable, unlike DMD)
    if live.exists():
        live_gif(live, OUT / "live_runs5/camera__clean_frames.csv", OUT / "live_runs5/labels_hand.csv", RES / "demo_live_phone.gif", 21, 51)
    clip = "synth02_sleep"
    n = demo_gif(ROOT / "data/synthetic" / f"{clip}.mp4", OUT / "synthetic/runs" / f"{clip}__clean_frames.csv",
                 OUT / "synthetic/runs" / f"{clip}__clean_warnings.csv", RES / "demo_synthetic_sleep.gif")
    full_videos()
    for p in sorted(RES.rglob("*.*")):
        print(f"{p.name:32} {p.stat().st_size / 1024:8.0f} KiB")
    print(f"gif frames: {n}")


if __name__ == "__main__":
    main()
