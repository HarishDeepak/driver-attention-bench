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

# CARLA's reference navigation agents ship with the server package (needs shapely + networkx)
sys.path.append(str(Path(__import__("os").environ.get("CARLA_ROOT", r"C:\carla\CARLA_0.9.16")) / "PythonAPI" / "carla"))
from agents.navigation.basic_agent import BasicAgent  # noqa: E402
from agents.navigation.local_planner import RoadOption  # noqa: E402

DT, MAP, CRUISE, AHEAD, STRAIGHT = 0.05, "Town04_Opt", 80.0, 800.0, 200.0  # Town04's longest straight is ~220 m (a highway loop)
TRAFFIC = [(-1, 40.0), (-1, -30.0), (1, 25.0), (1, -45.0), (-1, 110.0), (1, 90.0)]  # (left -1 / right +1 lane, metres ahead of ego)
ap = argparse.ArgumentParser()
ap.add_argument("source", help="camera index, video file or sim:A-B (eyes closed A..B seconds)")
ap.add_argument("--clahe", action="store_true")
ap.add_argument("--headless", action="store_true")
ap.add_argument("--seconds", type=float, default=60.0)
ap.add_argument("--eye", choices=["blend", "ear"], default="blend", help="eye signal (ear is better at night, E11)")
ap.add_argument("--calib-s", type=float, default=3.0, help="calibration window (s)")
ap.add_argument("--video", help="save the window view (face + chase camera) to this .mp4; works with --headless")
a = ap.parse_args()

sim = a.source.startswith("sim:")
live = a.source.isdigit()
if sim:
    c0, c1 = map(float, a.source[4:].split("-"))
else:
    from signals import Classifier, FaceSignals
    cap = cv2.VideoCapture(int(a.source) if live else a.source)
    assert cap.isOpened(), f"cannot open {a.source}"
    fs, clf = FaceSignals(use_clahe=a.clahe), Classifier(calib_s=a.calib_s, eye=a.eye)


def straight_start(m, ahead=STRAIGHT, yaw_tol=3.0):
    """First waypoint with `ahead` m of straight (heading within yaw_tol deg), junction-free road in front, with a driving lane on
    each side for traffic (ponytail: first hit, no scoring)."""
    # neighbour lanes must drive the same way (same lane_id sign): a 2-lane country road's "left lane" is oncoming traffic
    drv = lambda x, w: x is not None and x.lane_type == carla.LaneType.Driving and x.lane_id * w.lane_id > 0
    for w in m.generate_waypoints(20.0):
        if w.is_junction or not (drv(w.get_left_lane(), w) and drv(w.get_right_lane(), w)):
            continue
        c, ok = w, True
        for _ in range(int(ahead / 20)):
            n = c.next(20.0)
            if not n or n[0].is_junction or abs((n[0].transform.rotation.yaw - w.transform.rotation.yaw + 180) % 360 - 180) > yaw_tol:
                ok = False
                break
            c = n[0]
        if ok:
            return w
    raise SystemExit("no straight road found")


class MinimalRiskManoeuvre:
    """Unresponsive driver: CARLA's reference BasicAgent (LocalPlanner + PID controllers) takes the wheel from the Traffic Manager.
    UN R157 baseline: slow down inside the lane, hazards on, standstill, no move-off (done by Escalation + the stopped branch).
    On top, as some production systems do: move one lane to the right when the markings allow it and the target lane is free front and
    rear (the same checks as CARLA's BehaviorAgent lane change), until the rightmost driving lane; there, stop at the lane's right edge
    (the planner's lateral offset). Town04's shoulders are 0.5 m wide, too narrow to stop on, so the car stays on the driving lane."""

    def __init__(self, ego, world):
        self.ego, self.map, self.target_lane = ego, world.get_map(), None
        self.agent = BasicAgent(ego, target_speed=3.6 * ego.get_velocity().length(), opt_dict={"max_brake": 0.5})
        self.agent.ignore_traffic_lights(True)
        self.agent.follow_speed_limits(False)
        self.agent.set_global_plan(lane_plan(self.map.get_waypoint(ego.get_location()), AHEAD))  # no random turns at splits

    def step(self, target_kmh, vehicles):
        wp, v = self.map.get_waypoint(self.ego.get_location()), self.ego.get_velocity().length()
        if self.target_lane is None or wp.lane_id == self.target_lane:  # not in the middle of a lane change
            self.target_lane = None
            r = wp.get_right_lane()
            right_ok = (wp.right_lane_marking.lane_change in (carla.LaneChange.Right, carla.LaneChange.Both) and r is not None
                        and r.lane_type == carla.LaneType.Driving and r.lane_id * wp.lane_id > 0)
            if right_ok:
                busy, _, _ = self.agent._vehicle_obstacle_detected(vehicles, max(15.0, 3 * v), up_angle_th=180, lane_offset=1)
                path = [] if busy else BasicAgent._generate_lane_change_path(
                    wp, "right", distance_same_lane=max(v, 2.0), distance_other_lane=AHEAD, lane_change_distance=max(3 * v, 12.0))
                if path:
                    self.agent.set_global_plan(path)
                    self.target_lane = r.lane_id
            elif r is None or r.lane_type != carla.LaneType.Driving or r.lane_id * wp.lane_id < 0:  # rightmost driving lane
                self.agent.set_offset(wp.lane_width / 2 - self.ego.bounding_box.extent.y - 0.25)  # + = right of the lane centre
        self.agent.set_target_speed(target_kmh)
        return self.agent.run_step()


def lane_plan(w, length):
    """[(waypoint, LANEFOLLOW)] every 2 m along w's own lane, for BasicAgent.set_global_plan."""
    plan = [(w, RoadOption.LANEFOLLOW)]
    for _ in range(int(length / 2)):
        nxt = plan[-1][0].next(2.0)
        if not nxt:
            break
        p = plan[-1][0]
        plan.append((next((n for n in nxt if n.road_id == p.road_id and n.lane_id == p.lane_id), nxt[0]), RoadOption.LANEFOLLOW))
    return plan


LEVEL_BGR = {"ok": (0, 200, 0), "warning": (0, 200, 255), "slowing": (0, 128, 255), "mrm": (0, 0, 255), "stopped": (0, 0, 160)}


def speed_strip(rows, t_max, w, h=150, v_max=100.0):
    """Speed-over-time plot of the run so far (rows: [t, closed, away, warned, level, target, speed]), coloured by DMS level."""
    img = np.full((h, w, 3), 25, np.uint8)
    x0, x1, y0, y1 = 70, w - 170, 15, h - 25
    X = lambda t: int(x0 + (x1 - x0) * min(t / t_max, 1))
    Y = lambda v: int(y1 - (y1 - y0) * min(v / v_max, 1))
    for v in (0, 50, 100):
        cv2.line(img, (x0, Y(v)), (x1, Y(v)), (70, 70, 70), 1)
        cv2.putText(img, f"{v}", (x0 - 40, Y(v) + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
    cv2.putText(img, "km/h", (5, y0 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
    for p, q in zip(rows, rows[1:]):
        cv2.line(img, (X(p[0]), Y(p[6])), (X(q[0]), Y(q[6])), LEVEL_BGR.get(q[4], (255, 255, 255)), 3)
    if rows:
        cv2.putText(img, f"{rows[-1][6]:3.0f} km/h", (x1 + 15, h // 2 + 5), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(img, rows[-1][4].upper(), (x1 + 15, h // 2 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, LEVEL_BGR.get(rows[-1][4], (255, 255, 255)), 2)
    cv2.putText(img, f"time (s), 0-{t_max:.0f}", (x0, h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
    return img


def lane_path(w, length):
    """Locations every 10 m along w's own lane (TM set_path: no random branch at splits)."""
    path = []
    for _ in range(int(length / 10)):
        nxt = w.next(10.0)
        if not nxt:
            break
        w = next((n for n in nxt if n.road_id == w.road_id and n.lane_id == w.lane_id), nxt[0])
        path.append(w.transform.location)
    return path


client = carla.Client("127.0.0.1", 2000)
client.set_timeout(60)
world = client.get_world()
orig = world.get_settings()
tm = client.get_trafficmanager()
actors, rows, changes = [], [], []
try:
    if MAP not in world.get_map().name:
        world = client.load_world(MAP, map_layers=carla.MapLayer.Ground)  # ponytail: full Town04 at once crashes the 6 GB GPU
    for layer in (carla.MapLayer.Props, carla.MapLayer.StreetLights, carla.MapLayer.Buildings, carla.MapLayer.Foliage):
        world.load_map_layer(layer)  # added one by one after the ground: fits (~2.7 GB) and gives the picture motion cues
    # leftovers (e.g. a replay ends by putting its vehicles on autopilot) would drive through this run and into its recording
    left = world.get_actors().filter("vehicle.*")
    client.apply_batch_sync([carla.command.DestroyActor(x) for x in left])  # _sync: waits until the server applied it
    (ROOT / "out").mkdir(exist_ok=True)  # CARLA recorder log: exact replay with client.replay_file(path, 0, 0, 0)
    print(client.start_recorder(str(ROOT / "out" / f"live_{datetime.now():%Y%m%d_%H%M%S}.log")))
    s = world.get_settings()
    s.synchronous_mode, s.fixed_delta_seconds = True, DT
    world.apply_settings(s)
    tm.set_synchronous_mode(True)
    wp = straight_start(world.get_map())
    tf = wp.transform
    tf.location.z += 0.3
    ebp = world.get_blueprint_library().find("vehicle.tesla.model3")
    ebp.set_attribute("role_name", "hero")  # lets the replay (carla_video.py) find the ego among the traffic
    ego = world.spawn_actor(ebp, tf)
    actors.append(ego)
    fwd = wp.transform.get_forward_vector()
    ego.set_target_velocity(carla.Vector3D(fwd.x * CRUISE / 3.6, fwd.y * CRUISE / 3.6, fwd.z * CRUISE / 3.6))  # 3D: ~3.5% grade
    ego.set_autopilot(True, tm.get_port())
    tm.auto_lane_change(ego, False)
    tm.ignore_lights_percentage(ego, 100)
    tm.distance_to_leading_vehicle(ego, 5.0)
    tm.set_desired_speed(ego, CRUISE)
    # pin the TM to the start lane: at highway splits it otherwise picks a random branch (seen in a replay: the minimal-risk
    # stop ended at an exit gore instead of in lane)
    route = max(AHEAD, 1.2 * CRUISE / 3.6 * a.seconds)  # the TM stops where its path ends: cover the whole run
    tm.set_path(ego, lane_path(wp, route))
    # traffic in the neighbouring lanes at normal speed: shows the ego slowing and stopping, and makes the stop a stop in traffic
    cars = [b for b in world.get_blueprint_library().filter("vehicle.*") if int(b.get_attribute("number_of_wheels")) == 4]
    for i, (side, ahead) in enumerate(TRAFFIC):
        lane = wp.get_left_lane() if side < 0 else wp.get_right_lane()
        at = lane.next(ahead)[0] if ahead > 0 else lane.previous(-ahead)[0]
        t2 = at.transform
        t2.location.z += 0.3
        car = world.try_spawn_actor(cars[(7 * i) % len(cars)], t2)
        if car is None:
            continue
        actors.append(car)
        f2 = at.transform.get_forward_vector()
        car.set_target_velocity(carla.Vector3D(f2.x * CRUISE / 3.6, f2.y * CRUISE / 3.6, f2.z * CRUISE / 3.6))
        car.set_autopilot(True, tm.get_port())
        tm.auto_lane_change(car, False)
        tm.set_desired_speed(car, CRUISE + 5 * (i % 3))
        tm.set_path(car, lane_path(at, route))
    chase, vw = None, None
    if not a.headless or a.video:
        bp = world.get_blueprint_library().find("sensor.camera.rgb")
        bp.set_attribute("image_size_x", "640"), bp.set_attribute("image_size_y", "360")
        chase = world.spawn_actor(bp, carla.Transform(carla.Location(-10, 0, 4.5), carla.Rotation(pitch=-14)), attach_to=ego)
        actors.append(chase)
        q = queue.Queue()
        chase.listen(q.put)
        # top-down "drone" inset: lane dashes and traffic stream past at speed and stop with the car (a chase camera hides speed)
        bp.set_attribute("image_size_x", "240"), bp.set_attribute("image_size_y", "240")
        top = world.spawn_actor(bp, carla.Transform(carla.Location(0, 0, 35), carla.Rotation(pitch=-90)), attach_to=ego)
        actors.append(top)
        q2 = queue.Queue()
        top.listen(q2.put)
    world.tick()

    mon, esc = Monitor(), Escalation(CRUISE)
    t_wall, tick, level, frame, trim, pending, mrm = time.monotonic(), 0, "ok", None, 0.0, [], None
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
            if not live and t < tick * DT:  # video (25-30 fps) faster than the 20 Hz sim: every frame feeds the DMS, the car steps at sim rate
                pending += mon.step(t, closed, away)
                continue
        if t > a.seconds:
            break
        v = ego.get_velocity()
        speed = math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2) * 3.6
        warned, pending = pending + mon.step(t, closed, away), []
        level, target = esc.step(t, closed, away, warned, speed)
        if level == "stopped":
            ego.set_autopilot(False, tm.get_port())  # TM would override the brake
            ego.apply_control(carla.VehicleControl(brake=1.0, hand_brake=True))
        elif level == "mrm":
            if mrm is None:  # take over from the TM once, at the start of the manoeuvre
                ego.set_autopilot(False, tm.get_port())
                mrm = MinimalRiskManoeuvre(ego, world)
            ego.apply_control(mrm.step(target, world.get_actors().filter("vehicle.*")))
        else:
            # ponytail: TM holds ~3% under its desired speed; slow integral trim on the command closes that (not a real controller)
            trim = max(-10.0, min(10.0, trim + 0.01 * (target - speed)))
            tm.set_desired_speed(ego, max(target + trim, 0.1))
        # lights are not automatic for a client-controlled car: hazards from the start of the MRM (R157) and kept at standstill,
        # brake lights whenever the brake is applied, position lights always
        L = carla.VehicleLightState
        hazards, braking = level in ("mrm", "stopped"), ego.get_control().brake > 0.05 or level == "stopped"
        ego.set_light_state(L(int(L.Position) | (int(L.LeftBlinker) | int(L.RightBlinker) if hazards else 0) | (int(L.Brake) if braking else 0)))
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
                   "warnings: " + (", ".join(sorted(mon.fired)) or "-"),
                   "lights: " + (" ".join(n for n, on in (("HAZARDS", hazards), ("BRAKE", braking)) if on) or "-"),
                   "lane: " + (f"{world.get_map().get_waypoint(ego.get_location()).lane_id}" + ("  -> moving right" if mrm and mrm.target_lane else ""))]
            col = {"ok": (0, 200, 0), "warning": (0, 200, 255), "slowing": (0, 128, 255)}.get(level, (0, 0, 255))
            for i, line in enumerate(txt):
                cv2.putText(left, line, (8, 25 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col if i == 1 else (255, 255, 255), 2)
            tim = q2.get(timeout=10)
            pip = cv2.resize(np.frombuffer(tim.raw_data, np.uint8).reshape(tim.height, tim.width, 4)[:, :, :3], (190, 190))
            cam[8:198, -198:-8] = pip
            cv2.rectangle(cam, (cam.shape[1] - 199, 7), (cam.shape[1] - 7, 199), (255, 255, 255), 1)
            view = np.vstack([np.hstack([left, cam]), speed_strip(rows, a.seconds, 480 + cam.shape[1])])
            if a.video:  # one frame per sim tick -> the file plays at sim time (20 fps)
                vw = vw or cv2.VideoWriter(a.video, cv2.VideoWriter_fourcc(*"mp4v"), 1 / DT, (view.shape[1], view.shape[0]))
                vw.write(view)
            if not a.headless:
                cv2.imshow("live_demo", view)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                if sim:  # ponytail: pace sim to real time only when watched; render time usually dominates anyway
                    time.sleep(max(0.0, t_wall + tick * DT - time.monotonic()))
finally:
    if vw:
        vw.release()
    # order matters: TM autopilot off + sync off + world settings restored BEFORE destroying the ego, else libcarla aborts natively
    try:
        client.stop_recorder()
        for x in actors:
            if x.type_id.startswith("vehicle"):
                x.set_autopilot(False, tm.get_port())
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
