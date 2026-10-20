#!/usr/bin/env bash
# Record /dms/state with rosbag2, then play it back. Usage (WSL, repo root): bash scripts/ros2/record.sh [SOURCE=sim:2-40] [SECONDS=25]
set -eo pipefail  # no -u: ROS setup.bash uses unset vars
SRC="${1:-sim:2-40}"; SEC="${2:-25}"
set -m  # job control: background jobs keep the default SIGINT (rosbag2 stops cleanly on it)
source /opt/ros/jazzy/setup.bash
PY="${PY:-$HOME/dmsvenv/bin/python}"
BAG=out/ros2/bag; rm -rf "$BAG"; mkdir -p out/ros2
ros2 bag record -o "$BAG" /dms/state & REC=$!
sleep 3                                                   # let the recorder subscribe before the first sample
"$PY" scripts/ros2/dms_state_pub.py "$SRC" --seconds "$SEC"
sleep 1; kill -INT $REC; wait $REC || true
ros2 bag info "$BAG"
echo "--- playback (first 5 messages on /dms/state)"
ros2 bag play "$BAG" -r 10 & PLAY=$!
timeout 6 ros2 topic echo --once /dms/state std_msgs/msg/String --field data || true
wait $PLAY
