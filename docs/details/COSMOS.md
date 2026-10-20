# Cosmos Transfer prep (CARLA control videos -> photoreal variants)

Tags: **[verified]** = run here 2026-10-20, **[doc]** = read in a source, **[U]** = unverified. No NVIDIA account or key exists; nothing was sent anywhere.

## What it does

CARLA replays a recorder log and renders three aligned control videos from the ego camera: RGB, depth, semantic segmentation.
NVIDIA Cosmos Transfer takes those controls plus a text prompt and generates a photoreal video with the same scene structure
(rain, dusk, snow, ...). Use here: weather/lighting variants of the FCW scenario as "digital scenario" material. Output is
generated imagery of a driving scene, not a driver face, so it does not feed the DMS metrics directly (the DMS clips stay real/image-variant).

## Requirements

- **GPU: not local.** The Transfer2.5-2B NIM needs "Hopper or later and at least 80 GB" ([support matrix](https://docs.nvidia.com/nim/cosmos/latest/support-matrix.html)); CARLA's docs name H100s ([CARLA x Cosmos Transfer](https://carla.readthedocs.io/en/latest/nvidia_cosmos_transfer/)); earlier research here noted ~65 GB for Transfer1. The RTX 4050 (6 GB) is far off.
- **Hosting.** Documented ways to get a server:
  1. CARLA's server image (Brev 1-click launchable or `make_docker.sh`, both in the CARLA docs above; needs a funded Brev account) + CARLA's own `cosmos_client.py` (Gradio client; `PythonAPI/examples/nvidia/cosmos/client`). This is the path NVIDIA/CARLA documented end to end.
  2. A self-hosted NIM (`POST /v1/infer`, [API reference](https://docs.nvidia.com/nim/cosmos/latest/api-reference.html)): needs the big GPU too.
  3. **build.nvidia.com hosted endpoint:** the Transfer1-7B and Transfer2.5-2B pages are labelled "Free Endpoint", but I found **no documented API request for custom videos**, and a developer forum thread reports `404 Function not found for account` when calling it ([forum, 2026-09-02](https://forums.developer.nvidia.com/t/cosmos-transfer-2-5-free-endpoint-listed-but-custom-api-generation-is-unavailable-and-returns-404/382121); no NVIDIA reply seen). The web "Experience" tab may only offer preset examples. Treat hosted-with-custom-videos as **[U]**.
- Input rules [doc, [model card](https://build.nvidia.com/nvidia/cosmos-transfer1-7b)]: mp4 control videos derived from the same source with identical size/length; lengths that are multiples of 120 frames work best; prompt under 300 words. Our clips: 1280x720, 120 frames at 24 fps (5 s).

## Get a key (you must do this yourself)

Sign up at <https://build.nvidia.com> (NVIDIA Developer account), generate an API key (`nvapi-...`), set it for the session only:
`$env:NVIDIA_API_KEY = "nvapi-..."`. Do not commit it. Cost/credits: **[U]** I could not find credit amounts for the free endpoint; the NIM docs mention
the free NVIDIA Developer Program (self-hosting on up to 16 GPUs) and a 90-day AI Enterprise trial (business email). Brev bills GPU hours (price not checked).

## Step 1: control videos (done, [verified])

Source: the existing recorder log `out/carla_fcw.log`, run 12 = 80 km/h, inattentive driver, ungated FCW (ego actor id 353, 135.75 s to end).
CARLA's own generator was used as is, with a 3-sensor rig `scripts/cosmos_aov.yaml` (720p instead of 1080p for 6 GB VRAM):

```powershell
# server up (Town04_Opt, ground layer only: full layers crashed the GPU). Load the map first so the replay does not reload it with all layers:
.venv\Scripts\python.exe -c "import carla; c=carla.Client('127.0.0.1',2000); c.set_timeout(120); c.load_world('Town04_Opt', map_layers=carla.MapLayer.Ground)"
.venv\Scripts\python.exe C:\carla\CARLA_0.9.16\PythonAPI\examples\nvidia\cosmos\client\carla_cosmos_gen.py `
  -f out\carla_fcw.log --sensors scripts\cosmos_aov.yaml -c 353 -s 135.75 -d 6.6 --num-post-workers 2 -o out\cosmos
```

Needs `pip install pyyaml` in `.venv` (done) and `ffmpeg` on PATH (winget build present). Result in `out/cosmos/`:
`rgb.mp4`, `depth.mp4`, `semantic_segmentation.mp4`, each 1280x720, 120 frames at 24 fps (5.0 s; ffmpeg resamples the 20 Hz log to 24 fps).
Previews: `out/cosmos/preview/*.png`.

Known upstream quirks: (a) on Windows the script's `--class-filter-config` is lost in its worker processes (spawn), so `rgb_edges.mp4` / `rgb_masked.mp4`
come out black; I deleted them. No edge control video exists; derive one with Canny from `rgb.mp4` if you want it, or use depth+seg only.
(b) The clip covers the approach only (ego still ~40 m behind the stopped car at the end); the impact at ~6.6 s is cut off. [U] why ~100 of ~132 frames are written.
(c) The ego hood is visible at the bottom; the camera is at z=1.8 on the ego.

## Step 2: send to Cosmos (`scripts/cosmos_transfer.py`, **UNTESTED**)

```powershell
.venv\Scripts\python.exe scripts\cosmos_transfer.py --dry-run                      # builds the request, prints sizes, sends nothing [verified]
$env:NVIDIA_API_KEY = "nvapi-..."; $env:COSMOS_URL = "http://HOST:8000/v1/infer"      # URL of YOUR server (see hosting above)
.venv\Scripts\python.exe scripts\cosmos_transfer.py scripts\cosmos_prompt.toml --out out\cosmos\result.mp4
```

- Exits with a clear message if `NVIDIA_API_KEY` or the URL is missing (verified). Key goes in `Authorization: Bearer` (the scheme used by NVIDIA's hosted cosmos3-nano example [doc]; for a NIM the docs describe no auth).
- Request body (from the NIM Transfer2.5 `/v1/infer` doc example): `prompt`, `video`, `resolution` ("720"), and per control `{control_weight, control}`; videos are sent as plain base64
  (doc says "URL or base64-encoded video"; data-URI form untested). Response read as `{"b64_video": ..., "seed": ...}` and written to the mp4.
- **Assumed, not documented:** the `depth` and `seg` blocks use the same `{control, control_weight}` shape as the documented `edge` example. `seed`/`guidance` are not in the NIM doc, so they are not sent.
- CARLA's own `cosmos_client.py` uses a different protocol (Gradio `/upload_file` + `/generate_video`, TOML with `sigma_max`, `num_steps`, `guidance`, `[seg]`/`[depth]`/`[edge]`);
  use it against a Brev/Docker server from CARLA's docs. Example prompts: `...\nvidia\cosmos\client\example_data\prompts\*.toml`.

## Files

`scripts/cosmos_aov.yaml`, `scripts/cosmos_prompt.toml`, `scripts/cosmos_transfer.py`, `out/cosmos/{rgb,depth,semantic_segmentation}.mp4`, `out/cosmos/gen.log`.
