"""Replay a live_demo CARLA recorder log with a chase camera and burn in the escalation state -> out/videos/<log>.mp4.
Usage (CARLA server running): python scripts/carla_video.py [out/live_<stamp>.log]  (uses the matching live_<stamp>.csv for the overlay)"""
import csv
import queue
import sys
from pathlib import Path

import carla
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MAP, DT, W, H = "Town04_Opt", 0.05, 960, 540
COL = {"ok": (0, 200, 0), "warning": (0, 200, 255), "slowing": (0, 140, 255), "mrm": (0, 0, 255), "stopped": (0, 0, 180)}


def main():
    logs = sorted((ROOT / "out").glob("live_*.log"))
    log = Path(sys.argv[1]) if len(sys.argv) > 1 else logs[-1]
    # optional 2nd arg: the face video live_demo was fed (e.g. its render_overlay version) -> side-by-side, frame = t * container fps
    face = cv2.VideoCapture(sys.argv[2]) if len(sys.argv) > 2 else None
    # live_demo timed frames by the RAW video's container rate; an overlay video may carry the true rate instead -> optional 3rd arg = raw video
    face_fps = cv2.VideoCapture(sys.argv[3]).get(cv2.CAP_PROP_FPS) if len(sys.argv) > 3 else face.get(cv2.CAP_PROP_FPS) if face else 0
    # live_demo names the .log when recording starts and the .csv when the run ends (a few s apart): take the next csv after the log
    rows = list(csv.DictReader(open(min(p for p in (ROOT / "out").glob("live_*.csv") if p.stem >= log.stem))))
    client = carla.Client("localhost", 2000)
    client.set_timeout(60)
    world = client.load_world(MAP, map_layers=carla.MapLayer.Ground)  # ponytail: full Town04 at once crashes the 6 GB GPU
    for layer in (carla.MapLayer.Props, carla.MapLayer.StreetLights, carla.MapLayer.Buildings, carla.MapLayer.Foliage):
        world.load_map_layer(layer)
    s = world.get_settings()
    s.synchronous_mode, s.fixed_delta_seconds = True, DT
    world.apply_settings(s)
    print(client.replay_file(str(log.resolve()), 0, 0, 0, False))
    for _ in range(40):  # replayed actors appear after a few ticks
        world.tick()
        vehicles = world.get_actors().filter("vehicle.*")
        if vehicles:
            break
    assert vehicles, "replay produced no vehicle"
    hero = [v for v in vehicles if v.attributes.get("role_name") == "hero"]
    ego = hero[0] if hero else min(vehicles, key=lambda a: a.id)  # the replayed ego (older logs: the first vehicle spawned)
    bp = world.get_blueprint_library().find("sensor.camera.rgb")
    bp.set_attribute("image_size_x", str(W))
    bp.set_attribute("image_size_y", str(H))
    cam = world.spawn_actor(bp, carla.Transform(carla.Location(x=-10, z=4.5), carla.Rotation(pitch=-14)), attach_to=ego)
    q = queue.Queue()
    cam.listen(q.put)
    dst = ROOT / "out" / "videos" / f"{log.stem}_carla.mp4"
    dst.parent.mkdir(parents=True, exist_ok=True)
    FW = round(H * 16 / 9) if face else 0
    vw = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), 1 / DT, (W + FW, H))
    try:
        for r in rows:
            world.tick()
            img = q.get(timeout=10)
            f = np.frombuffer(img.raw_data, np.uint8).reshape(H, W, 4)[:, :, :3].copy()
            f[:70] = (f[:70] * 0.35).astype(np.uint8)
            lvl = r["level"]
            cv2.putText(f, f"t={float(r['t']):5.1f}s   driver: {'EYES CLOSED' if r['closed'] == '1' else 'attentive'}", (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(f, f"DMS level: {lvl.upper()}   speed {float(r['speed_kmh']):5.1f} km/h (target {float(r['target_kmh']):.0f})", (12, 58),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, COL.get(lvl, (255, 255, 255)), 2, cv2.LINE_AA)
            if face:
                face.set(cv2.CAP_PROP_POS_FRAMES, round(float(r["t"]) * face_fps))
                ok, fimg = face.read()
                f = np.hstack([cv2.resize(fimg, (FW, H)) if ok else np.zeros((H, FW, 3), np.uint8), f])
            vw.write(f)
    finally:
        vw.release()
        cam.destroy()
        client.stop_replayer(False)
        s.synchronous_mode = False
        world.apply_settings(s)
    print(dst)


if __name__ == "__main__":
    main()
