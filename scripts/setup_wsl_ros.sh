#!/usr/bin/env bash
# Installs ROS 2 Jazzy (base) on Ubuntu 24.04. Run as root: wsl -d Ubuntu -u root bash scripts/setup_wsl_ros.sh
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y software-properties-common curl
add-apt-repository -y universe
curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo "$UBUNTU_CODENAME") main" > /etc/apt/sources.list.d/ros2.list
apt-get update
apt-get install -y ros-jazzy-ros-base python3-colcon-common-extensions python3-pip python3-venv
echo "ROS_OK: $(ls /opt/ros)"
