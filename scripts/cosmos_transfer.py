"""Send CARLA control videos (RGB + depth + semantic seg) and a prompt TOML to a Cosmos Transfer endpoint, save the result mp4.
UNTESTED: no API key / endpoint was available when this was written; the request body follows the NVIDIA NIM for Cosmos Transfer2.5
/v1/infer example (https://docs.nvidia.com/nim/cosmos/latest/api-reference.html): prompt, video, resolution, <control>: {control, control_weight}.
Only the `edge` control is shown in that doc; depth/seg/vis are ASSUMED to use the same shape. A hosted build.nvidia.com endpoint
for custom videos is NOT documented (forum report: 404 'Function not found for account'), so the URL is yours to supply (docs/details/COSMOS.md).
Usage: NVIDIA_API_KEY=... python scripts/cosmos_transfer.py --url https://HOST/v1/infer [scripts/cosmos_prompt.toml] [--out out/cosmos/result.mp4]
       python scripts/cosmos_transfer.py --dry-run     # build the request, print sizes, send nothing (no key needed)"""
import argparse
import base64
import json
import os
import sys
import tomllib
import urllib.request
from pathlib import Path

CONTROLS = ("edge", "depth", "vis", "seg")  # documented controls; each request needs at least one


def b64(path):
    return base64.b64encode(Path(path).read_bytes()).decode()  # ponytail: plain base64 ('URL or base64-encoded video' in the doc); data-URI form untested


def build(cfg):
    body = {"prompt": cfg["prompt"], "video": b64(cfg["video"]), "resolution": str(cfg.get("resolution", "720"))}
    for c in CONTROLS:
        if c in cfg:
            body[c] = {"control_weight": cfg[c]["control_weight"], **({"control": b64(cfg[c]["control"])} if "control" in cfg[c] else {})}
    assert any(c in body for c in CONTROLS), "need at least one control (edge/depth/vis/seg) in the TOML"
    return body


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("toml", nargs="?", default=str(Path(__file__).with_name("cosmos_prompt.toml")))
    ap.add_argument("--url", default=os.environ.get("COSMOS_URL"), help="inference URL, e.g. http://HOST:8000/v1/infer (or env COSMOS_URL)")
    ap.add_argument("--out", default="out/cosmos/result.mp4")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    key = os.environ.get("NVIDIA_API_KEY")
    if not a.dry_run and not key:
        sys.exit("NVIDIA_API_KEY is not set. Create a key yourself at https://build.nvidia.com (account needed), then: set NVIDIA_API_KEY=nvapi-...  (see docs/details/COSMOS.md)")
    if not a.dry_run and not a.url:
        sys.exit("no endpoint: pass --url or set COSMOS_URL (a hosted URL for custom control videos is not documented, see docs/details/COSMOS.md)")
    body = build(tomllib.load(open(a.toml, "rb")))
    show = lambda v: f"<{len(v) // 1024} KiB base64>" if isinstance(v, str) and v.startswith("AAAA") else {k: show(x) for k, x in v.items()} if isinstance(v, dict) else v
    print(show(body))
    if a.dry_run:
        return
    req = urllib.request.Request(a.url, json.dumps(body).encode(), {"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    try:
        r = json.load(urllib.request.urlopen(req, timeout=a.timeout))
    except urllib.error.HTTPError as e:
        sys.exit(f"HTTP {e.code}: {e.read()[:500].decode(errors='replace')}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_bytes(base64.b64decode(r["b64_video"]))  # response shape per doc: {"b64_video": ..., "seed": ...}
    print(f"saved {a.out} (seed {r.get('seed')})")


if __name__ == "__main__":
    main()
