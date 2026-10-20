# Recording script (your own test clips)

Camera: Xiaomi 14 front camera, recorded with the normal camera app (1080p, 30 fps), then copied to `data/clips/`.
Hold the phone fixed (stand, books, or dashboard of a PARKED car), about 50-70 cm from the face, slightly below eye level and
a little off to the side, like a real DMS camera. Never record while driving.

Every clip starts the same way: **look straight at the "road" (past the camera), eyes open, still, for 3 s.**
This is the per-driver calibration window. Count seconds in your head; exact timing does not matter, the labels do.

| File name | Actions after the 3 s calibration |
|---|---|
| `clip01_microsleep.mp4` | blink normally 3 times - close eyes 2 s - open 4 s - close eyes 2 s - open 3 s |
| `clip02_sleep.mp4` | close eyes 5 s - open 4 s - close eyes 4 s - open 3 s |
| `clip03_owl.mp4` | turn head to the right (mirror) 4 s - back 4 s - head down to the left (console) 4 s - back 3 s - quick 1 s glance right - back 3 s |
| `clip04_lizard.mp4` | head still, eyes only to the right 4 s - back 4 s - eyes only down 4 s - back 3 s |
| `clip05_sunglasses.mp4` | wear sunglasses: close eyes 2 s - open 4 s - close eyes 5 s - open 4 s - turn head right 4 s - back 3 s |
| `clip06_mask.mp4` | wear a face mask (or a cap): same actions as clip05 |

## Labels (ground truth)

After recording, open each clip in a player that can step frame by frame (VLC: `e` = next frame; time shown in ms with the
"precise time" setting, or use any editor). Note the time where the eyes are fully closed / the head (or eyes) have left the road,
and the time they are back. Write one row per event to `data/clips/labels.csv`:

```
clip,kind,t_start,t_end
clip01_microsleep,closed,6.40,8.45
clip03_owl,away,4.10,8.20
```

- `clip` = file name without `.mp4`; `kind` = `closed` or `away`; times in seconds, 2 decimals.
- Do NOT label normal blinks or the 1 s glance: the system must not warn on those, so they count only if it fires (false alert).

Then run the suite: `wsl bash -lc 'cd /mnt/c/<path-to>/driver-attention-bench && bash run_suite.sh'`.
