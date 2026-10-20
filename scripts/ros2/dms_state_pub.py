"""Publish the DMS state on /dms/state (std_msgs/String, JSON: t, closed, away, warnings, level) at real time.
SOURCE = CSV with t,closed,away columns (live_demo out/live_*.csv or run_clip *_frames.csv) | sim:A-B (eyes closed A..B s).
Warnings/level are recomputed with src/dms_core.Monitor + src/escalation.Escalation, so any such CSV works.
Run in WSL: source /opt/ros/jazzy/setup.bash; ~/dmsvenv/bin/python scripts/ros2/dms_state_pub.py sim:2-40"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import rclpy
from std_msgs.msg import String

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from dms_core import Monitor  # noqa: E402
from escalation import Escalation  # noqa: E402

DT, CRUISE = 0.05, 80.0


def samples(src, seconds):
    """Yield (t, closed, away)."""
    if src.startswith("sim:"):
        c0, c1 = map(float, src[4:].split("-"))
        for i in range(int(seconds / DT)):
            yield i * DT, c0 <= i * DT < c1, False
    else:
        for r in csv.DictReader(open(src)):
            yield float(r["t"]), bool(int(float(r["closed"]))), bool(int(float(r["away"])))


def states(src, seconds):
    """Yield the state dict per sample (pure, no ROS)."""
    mon, esc, speed = Monitor(), Escalation(CRUISE), CRUISE
    for t, closed, away in samples(src, seconds):
        level, speed = esc.step(t, closed, away, mon.step(t, closed, away), speed)  # ponytail: ideal speed tracking, not the CARLA speed
        yield dict(t=round(t, 3), closed=closed, away=away, warnings=sorted(mon.fired), level=level)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("--seconds", type=float, default=30.0, help="sim: only")
    ap.add_argument("--topic", default="/dms/state")
    ap.add_argument("--loop", action="store_true")
    a = ap.parse_args()
    rclpy.init()
    node = rclpy.create_node("dms_state_pub")
    pub = node.create_publisher(String, a.topic, 10)
    try:
        while rclpy.ok():
            t0, w0 = None, time.monotonic()
            for s in states(a.source, a.seconds):
                t0 = s["t"] if t0 is None else t0
                time.sleep(max(0.0, w0 + s["t"] - t0 - time.monotonic()))  # real time
                pub.publish(String(data=json.dumps(s)))
            if not a.loop:
                break
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    if len(sys.argv) == 1:  # self-check, no ROS graph needed
        lv = [s["level"] for s in states("sim:2-40", 40)]
        assert lv[0] == "ok" and lv[-1] == "stopped" and "slowing" in lv, lv
        assert next(s for s in states("sim:2-40", 40) if s["warnings"])["t"] == 3.5
        print("dms_state_pub OK")
    else:
        main()
