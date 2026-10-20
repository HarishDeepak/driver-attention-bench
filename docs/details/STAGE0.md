# Stage 0 — de-risk results (2026-10-20)

## CARLA 0.9.16 on RTX 4050 Laptop (6 GB VRAM)
- Extraction verified: 31,417 files, same count as the zip; `CarlaUE4.exe` present.
- Server: `CarlaUE4.exe -quality-level=Low -windowed -ResX=800 -ResY=600` (on-screen rendering, no `-RenderOffScreen` needed).
- Smoke test (`scripts/carla_smoke.py`, Town01, Lincoln MKZ on autopilot, 1 RGB camera 640x480, async mode, 15 s):

| Metric | Result |
|---|---|
| Frames received | 1201 in 15 s = **80.1 fps** |
| VRAM, map loaded | 2034 MiB of 6141 MiB |
| VRAM, under load | **2263 MiB** of 6141 MiB |
| Result | `SMOKE_OK`, no crash / OOM |

**Decision: GPU test passed, CARLA stage stays in scope.** ~3.8 GB VRAM headroom (a second camera or Town with more assets is fine).
Caveat: async mode, 1 camera, low quality. Final sweep will use synchronous mode with a fixed time step, so fps here is only a capacity indicator.
