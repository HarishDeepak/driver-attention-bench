"""VLM arm: zero-shot hosted vision-language models judge driver frames (eyes open/closed, looking at road), scored against labels.
Frames come from data/synthetic only (DMD is never sent to third parties; any path containing "dmd" is refused).
Usage: $env:NVIDIA_API_KEY=...; .venv\\Scripts\\python.exe scripts/vlm_arm.py [--step 1.0] [--limit N] [--models m1 m2]
       python scripts/vlm_arm.py --dry-run      # build one request, print its size, send nothing (no key needed)
       python scripts/vlm_arm.py --selfcheck    # asserts on the JSON parser and the metrics (no key needed)
Outputs: out/vlm/frames.csv (one row per frame and model), out/vlm/summary.csv (one row per model)."""
import argparse
import base64
import csv
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "synthetic"
OUT = ROOT / "out" / "vlm"
URL = "https://integrate.api.nvidia.com/v1/chat/completions"
DEFAULT_MODELS = ["nvidia/cosmos-reason2-8b", "meta/llama-3.2-11b-vision-instruct", "google/gemma-3-12b-it"]
PROMPT = ('This is one frame from an in-cabin driver monitoring camera. Reply with strict JSON only: '
          '{"eyes":"open|closed|unknown","looking_at_road":true|false}. '
          '"eyes" = both eyes open, closed, or cannot be determined. "looking_at_road" = gaze directed at the road ahead.')
MAX_SIDE, MAX_RETRIES, SKIP_AFTER = 512, 3, 3
FIELDS = ["model", "clip", "t", "gt_closed", "gt_away", "pred_eyes", "pred_road", "latency_s", "error"]


def refuse(p):
    if "dmd" in str(p).lower():
        sys.exit(f"refused: {p} is DMD data; its licence forbids sending it to third parties")


def load_labels(path):
    out = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            out.setdefault((r["clip"], r["kind"]), []).append((float(r["t_start"]), float(r["t_end"])))
    return out


def inside(intervals, t):
    return any(s <= t <= e for s, e in intervals)


def jpeg_b64(img):
    h, w = img.shape[:2]
    s = min(1.0, MAX_SIDE / max(h, w))
    if s < 1:
        img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    assert ok, "jpeg encode failed"
    return base64.b64encode(buf.tobytes()).decode()


def sample(clip, step, limit):
    """(t, base64 jpeg) every `step` seconds, at most `limit` frames."""
    cap = cv2.VideoCapture(str(clip))
    assert cap.isOpened(), f"cannot open {clip}"
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    out, idx, nxt = [], 0, 0.0
    while limit is None or len(out) < limit:
        ok, img = cap.read()
        if not ok:
            break
        t = idx / fps
        idx += 1
        if t + 1e-9 >= nxt:
            nxt += step
            out.append((t, jpeg_b64(img)))
    cap.release()
    return out


def request_body(model, b64):
    return {"model": model, "temperature": 0, "max_tokens": 1024,  # room for reasoning models' thinking text
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64}}]}]}


def parse_reply(text):
    """First JSON object in the reply that has an 'eyes' key (models may wrap it in reasoning text)."""
    dec = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _ = dec.raw_decode(text, i)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "eyes" in obj:
            break
    else:
        raise ValueError("no JSON object with 'eyes' in reply")
    eyes, road = obj.get("eyes"), obj.get("looking_at_road")
    if eyes not in ("open", "closed", "unknown") or not isinstance(road, bool):
        raise ValueError(f"bad JSON fields: {obj}")
    return eyes, road


def post(req_obj, key):
    req = urllib.request.Request(URL, json.dumps(req_obj).encode(),
                                 {"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    for attempt in range(MAX_RETRIES + 1):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                m = json.load(r)["choices"][0]["message"]  # reasoning models may leave content empty
                return m.get("content") or m.get("reasoning_content") or ""
        except urllib.error.HTTPError as e:
            if (e.code == 429 or e.code >= 500) and attempt < MAX_RETRIES:
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(f"HTTP {e.code}: {e.read()[:200].decode(errors='replace')}")


def metrics(pairs):
    """pairs: (gt bool, pred bool or None). None (unknown/error) is never correct. Returns (balanced accuracy, F1)."""
    pos = [pr for g, pr in pairs if g]
    neg = [pr for g, pr in pairs if not g]
    sens = sum(pr is True for pr in pos) / len(pos) if pos else float("nan")
    spec = sum(pr is False for pr in neg) / len(neg) if neg else float("nan")
    tp = sum(g and pr is True for g, pr in pairs)
    fp = sum((not g) and pr is True for g, pr in pairs)
    fn = sum(g and pr is not True for g, pr in pairs)
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    return (sens + spec) / 2, f1


def run_model(model, frames, key):
    rows, reason = [], ""
    for f in frames:
        t0 = time.monotonic()
        try:
            eyes, road = parse_reply(post(request_body(model, f["b64"]), key))
            err = ""
        except Exception as e:  # any failure is one error row; the run keeps going
            eyes, road, err = "", None, str(e).replace(key, "<key>")[:300]
            err = f"{type(e).__name__}: {err}"
        rows.append(dict(model=model, clip=f["clip"], t=f["t"], gt_closed=f["gt_closed"], gt_away=f["gt_away"],
                         pred_eyes=eyes, pred_road=road, latency_s=round(time.monotonic() - t0, 3), error=err))
        if len(rows) == SKIP_AFTER and all(r["error"] for r in rows):
            reason = rows[-1]["error"]
            break
    return rows, reason


def summarise(model, rows, reason):
    lat = [r["latency_s"] for r in rows]
    ba_c, f1_c = metrics([(r["gt_closed"], None if not r["pred_eyes"] or r["pred_eyes"] == "unknown" else r["pred_eyes"] == "closed") for r in rows])
    ba_a, f1_a = metrics([(r["gt_away"], None if r["pred_road"] is None else not r["pred_road"]) for r in rows])
    return dict(model=model, status=f"skipped: {reason}" if reason else "ok", n=len(rows),
                n_errors=sum(bool(r["error"]) for r in rows), ba_closed=round(ba_c, 3), f1_closed=round(f1_c, 3),
                ba_away=round(ba_a, 3), f1_away=round(f1_a, 3),
                median_latency_s=round(statistics.median(lat), 3) if lat else "",
                eff_fps=round(len(rows) / sum(lat), 2) if sum(lat) else "")


def selfcheck():
    assert parse_reply('{"eyes":"closed","looking_at_road":true}') == ("closed", True)
    assert parse_reply('Let me think {about} the eyes.\nAnswer: {"eyes":"open","looking_at_road":false} done.') == ("open", False)
    assert parse_reply('Note {x} then ```json\n{"eyes":"unknown","looking_at_road":true}\n```') == ("unknown", True)
    for bad in ["no json here", '{"eyes":"maybe","looking_at_road":true}', '{"eyes":"open","looking_at_road":"yes"}']:
        try:
            parse_reply(bad)
        except ValueError:
            continue
        raise AssertionError(f"parser accepted {bad!r}")
    ba, f1 = metrics([(True, True), (True, None), (False, False), (False, True)])  # sens .5, spec .5, tp1 fp1 fn1
    assert abs(ba - 0.5) < 1e-9 and abs(f1 - 0.5) < 1e-9, (ba, f1)
    ba, f1 = metrics([(True, True), (False, False)])
    assert ba == 1 and f1 == 1
    try:
        refuse(DATA / "dmd" / "x.mp4")
    except SystemExit:
        pass
    else:
        raise AssertionError("DMD path not refused")
    body = request_body("m", "AAAA")
    assert body["messages"][0]["content"][1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    print("selfcheck OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=float, default=1.0, help="seconds between sampled frames")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--limit", type=int, help="frames per clip (testing)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    if a.selfcheck:
        return selfcheck()
    refuse(DATA)
    clips = sorted(DATA.glob("synth*.mp4"))
    assert clips, f"no synth*.mp4 in {DATA}"
    labels = load_labels(DATA / "labels.csv")
    frames = []
    for c in clips:
        refuse(c)
        for t, b64 in sample(c, a.step, a.limit):
            frames.append(dict(clip=c.stem, t=t, b64=b64,
                               gt_closed=inside(labels.get((c.stem, "closed"), []), t),
                               gt_away=inside(labels.get((c.stem, "away"), []), t)))
    if a.dry_run:
        size = len(json.dumps(request_body(a.models[0], frames[0]["b64"])))
        print(f"dry run: {len(frames)} frames x {len(a.models)} models = {len(frames) * len(a.models)} calls planned")
        print(f"one request ({a.models[0]}): {size / 1024:.1f} KiB JSON, image {len(frames[0]['b64']) / 1024:.1f} KiB base64; nothing sent")
        return
    key = os.environ.get("NVIDIA_API_KEY")
    if not key:
        sys.exit("NVIDIA_API_KEY is not set (set it in this shell, never in a file)")
    all_rows, summ = [], []
    for m in a.models:
        rows, reason = run_model(m, frames, key)
        all_rows += rows
        summ.append(summarise(m, rows, reason))
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "frames.csv", "w", newline="") as f:
        w = csv.DictWriter(f, FIELDS)
        w.writeheader()
        w.writerows(all_rows)
    with open(OUT / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, list(summ[0]))
        w.writeheader()
        w.writerows(summ)
    for s in summ:
        print(" | ".join(f"{k}={v}" for k, v in s.items()))


if __name__ == "__main__":
    main()
