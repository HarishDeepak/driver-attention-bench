"""Live driver-in-the-loop demo: camera / video / synthetic DMS input -> FaceSignals -> Classifier -> Monitor -> Escalation -> CARLA ego.
SOURCE = camera index | video file | sim:A-B (eyes closed from A to B seconds, no camera).
Ego drives Town04_Opt under Traffic Manager autopilot; each tick the escalation target speed is given to the TM.
ALL ESCALATION NUMBERS ARE ILLUSTRATIVE ASSUMPTIONS (see src/escalation.py), not a certified function.
Needs a running CARLA 0.9.16 server (127.0.0.1:2000). Output: out/live_<time>.csv. q quits the window."""
import argparse
import csv
import math
import queue
import sys
import time
from datetime import datetime
from pathlib import Path

import carla
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from dms_core import Monitor  # noqa: E402
from escalation import Escalation  # noqa: E402

DT, MAP, CRUISE, AHEAD = 0.05, "Town04_Opt", 80.0, 800.0
ap = argparse.ArgumentParser()
ap.add_argument("source", help="camera index, video file or sim:A-B (eyes closed A..B seconds)")
ap.add_argument("--clahe", action="store_true")
ap.add_argument("--headless", action="store_true")
ap.add_argument("--seconds", type=float, default=60.0)
a = ap.parse_args()

sim = a.source.startswith("sim:")
live = a.source.isdigit()
if sim:
    c0, c1 = map(float, a.source[4:].split("-"))
else:
    from signals import Classifier, FaceSignals
    cap = cv2.VideoCapture(int(a.source) if live else a.source)
    assert cap.isOpened(), f"cannot open {a.source}"
    fs, clf = FaceSignals(use_clahe=a.clahe), Classifier()


def straight_start(m):
    """First waypoint with AHEAD m of junction-free road in front of it (ponytail: first hit, no scoring)."""
    for w in m.generate_waypoints(20.0):
        c, ok = w, not w.is_junction
        for _ in range(int(AHEAD / 20) if ok else 0):
            n = c.next(20.0)
            if not n or n[0].is_junction:
                ok = False
                break
            c = n[0]
        if ok:
            return w
    raise SystemExit("no long road found")


client = carla.Client("127.0.0.1", 2000)
client.set_timeout(60)
world = client.get_world()
orig = world.get_settings()
tm = client.get_trafficmanager()
actors, rows, changes = [], [], []
try:
    if MAP not in world.get_map().name:
        world = client.load_world(MAP, map_layers=carla.MapLayer.Ground)  # ponytail: ground-only, full Town04 crashes the 6 GB GPU
    (ROOT / "out").mkdir(exist_ok=True)  # CARLA recorder log: exact replay with client.replay_file(path, 0, 0, 0)
    print(client.start_recorder(str(ROOT / "out" / f"live_{datetime.now():%Y%m%d_%H%M%S}.log")))
    s = world.get_settings()
    s.synchronous_mode, s.fixed_delta_seconds = True, DT
    world.apply_settings(s)
    tm.set_synchronous_mode(True)
    wp = straight_start(world.get_map())
    tf = wp.transform
    tf.location.z += 0.3
    ego = world.spawn_actor(world.get_blueprint_library().find("vehicle.tesla.model3"), tf)
    actors.append(ego)
    fwd = wp.transform.get_forward_vector()
    ego.set_target_velocity(carla.Vector3D(fwd.x * CRUISE / 3.6, fwd.y * CRUISE / 3.6, fwd.z * CRUISE / 3.6))  # 3D: ~3.5% grade
    ego.set_autopilot(True, tm.get_port())
    tm.auto_lane_change(ego, False)
    tm.ignore_lights_percentage(ego, 100)
    tm.distance_to_leading_vehicle(ego, 5.0)
    tm.set_desired_speed(ego, CRUISE)
    chase = None
    if not a.headless:
        bp = world.get_blueprint_library().find("sensor.camera.rgb")
        bp.set_attribute("image_size_x", "640"), bp.set_attribute("image_size_y", "360")
        chase = world.spawn_actor(bp, carla.Transform(carla.Location(-8, 0, 3), carla.Rotation(pitch=-12)), attach_to=ego)
        actors.append(chase)
        q = queue.Queue()
        chase.listen(q.put)
    world.tick()

    mon, esc = Monitor(), Escalation(CRUISE)
    t_wall, tick, level, frame, trim = time.monotonic(), 0, "ok", None, 0.0
    print(f"map {world.get_map().name}, source {a.source}, cruise {CRUISE:.0f} km/h")
    while True:
        if sim:
            t, closed, away = tick * DT, c0 <= tick * DT < c1, False
        else:
            ok, frame = cap.read()
            if not ok:
                break
            # file: container timestamp; camera: sim time (the loop is paced by the camera, ponytail: drifts if camera fps < 20)
            t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000 if not live else tick * DT
            st = clf.step(t, fs(frame, t * 1000), fs.fb)
            closed, away = (bool(st[0]), st[1]) if st else (False, False)  # calibrating: assume attentive; eyes unknown (None) = not closed
        if t > a.seconds:
            break
        v = ego.get_velocity()
        speed = math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2) * 3.6
        warned = mon.step(t, closed, away)
        level, target = esc.step(t, closed, away, warned, speed)
        if level == "stopped":
            ego.set_autopilot(False, tm.get_port())  # TM would override the brake
            ego.apply_control(carla.VehicleControl(brake=1.0, hand_brake=True))
        else:
            # ponytail: TM holds ~3% under its desired speed; slow integral trim on the command closes that (not a real controller)
            trim = max(-10.0, min(10.0, trim + 0.01 * (target - speed)))
            tm.set_desired_speed(ego, max(target + trim, 0.1))
        # hazards from mrm on; cleared again if the driver recovers (mrm is latched in Escalation, so only matters for stopped->no)
        ego.set_light_state(carla.VehicleLightState(
            carla.VehicleLightState.LeftBlinker | carla.VehicleLightState.RightBlinker if level in ("mrm", "stopped")
            else carla.VehicleLightState.NONE))
        rows.append([round(t, 3), int(closed), int(away), "+".join(warned), level, round(target, 2), round(speed, 2)])
        if not changes or changes[-1][1] != level:
            changes.append((t, level, speed))
            print(f"t={t:6.2f}s  {level:<8} target {target:5.1f}  speed {speed:5.1f} km/h")
        world.tick()
        tick += 1
        if chase is not None:
            im = q.get(timeout=10)
            cam = np.frombuffer(im.raw_data, np.uint8).reshape(im.height, im.width, 4)[:, :, :3].copy()
            left = cv2.resize(frame, (480, 360)) if frame is not None else np.full((360, 480, 3), 30, np.uint8)
            txt = [f"t={t:.1f}s  closed={int(closed)} away={int(away)}" + ("  [sim]" if sim else ""),
                   f"level: {level.upper()}", f"target {target:5.1f}  actual {speed:5.1f} km/h",
                   "warnings: " + (", ".join(sorted(mon.fired)) or "-")]
            col = {"ok": (0, 200, 0), "warning": (0, 200, 255), "slowing": (0, 128, 255)}.get(level, (0, 0, 255))
            for i, line in enumerate(txt):
                cv2.putText(left, line, (8, 25 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col if i == 1 else (255, 255, 255), 2)
            cv2.imshow("live_demo", np.hstack([left, cam]))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            if sim:  # ponytail: pace sim to real time only when watched; render time usually dominates anyway
                time.sleep(max(0.0, t_wall + tick * DT - time.monotonic()))
finally:
    # order matters: TM autopilot off + sync off + world settings restored BEFORE destroying the ego, else libcarla aborts natively
    try:
        client.stop_recorder()
        ego.set_autopilot(False, tm.get_port())
        tm.set_synchronous_mode(False)
        world.apply_settings(orig)
    except Exception as e:  # noqa: BLE001  (server gone / nothing spawned: nothing left to restore)
        print("restore failed:", e)
    for x in reversed(actors):
        if x.type_id.startswith("sensor"):
            x.stop()
        x.destroy()
    if not a.headless:
        cv2.destroyAllWindows()
    if not sim:
        cap.release()

out = ROOT / "out"
out.mkdir(exist_ok=True)
path = out / f"live_{datetime.now():%Y%m%d_%H%M%S}.csv"
with open(path, "w", newline="") as fh:
    cw = csv.writer(fh)
    cw.writerow("t,closed,away,warnings,level,target_kmh,speed_kmh".split(","))
    cw.writerows(rows)
print("timeline:", " -> ".join(f"{l}@{t:.1f}s" for t, l, _ in changes))
print(f"final level {changes[-1][1] if changes else '-'}, final speed {rows[-1][6] if rows else '-'} km/h, "
      f"max speed {max((r[6] for r in rows), default=0)} km/h, {len(rows)} ticks -> {path}")
