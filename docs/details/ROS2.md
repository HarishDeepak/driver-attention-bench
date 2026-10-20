# ROS 2 piece (Jazzy in WSL) and the CARLA link

Tags: **[verified]** = run on this machine 2026-10-20, **[doc]** = read in a source (linked), **[U]** = unverified.

## What works now: DMS state as a ROS 2 topic + rosbag2

```bash
# WSL, repo root. Self-check (no ROS graph needed):
wsl bash -lc 'cd /mnt/c/<path-to>/driver-attention-bench && source /opt/ros/jazzy/setup.bash && ~/dmsvenv/bin/python scripts/ros2/dms_state_pub.py'
# publish at real time (sim: eyes closed 2..40 s, or a live_demo / run_clip CSV with t,closed,away columns):
~/dmsvenv/bin/python scripts/ros2/dms_state_pub.py sim:2-40 --seconds 30
~/dmsvenv/bin/python scripts/ros2/dms_state_pub.py out/live_20261020_155832.csv
# record -> info -> play back -> echo, in one go (bag lands in out/ros2/bag):
bash scripts/ros2/record.sh sim:2-40 25
```

- `scripts/ros2/dms_state_pub.py`: node `dms_state_pub`, topic `/dms/state`, `std_msgs/String` with JSON
  `{"t", "closed", "away", "warnings", "level"}`. `warnings` = SD-202 warnings active now (`src/dms_core.Monitor`), `level` =
  escalation level `ok/warning/slowing/mrm/stopped` (`src/escalation.Escalation`). Both are recomputed from `closed/away`, so any CSV
  with those columns works. Escalation assumes ideal speed tracking (not the CARLA speed).
- `scripts/ros2/record.sh`: `ros2 bag record` (mcap) + publisher + `ros2 bag info` + `ros2 bag play -r 10` + `ros2 topic echo`.
  It needs job control (`set -m`): background jobs of a non-interactive bash ignore SIGINT, so rosbag2 would never stop.
- **[verified]** `record.sh sim:2-40 25`: bag `out/ros2/bag`, storage mcap, ROS Distro jazzy, duration 24.9 s, 499 messages
  on `/dms/state` (`std_msgs/msg/String`, cdr); playback echoed e.g.
  `{"t": 5.45, "closed": true, "away": false, "warnings": ["microsleep", "sleep"], "level": "warning"}`.
  CSV input (`out/live_20261020_155832.csv`) also verified: `ros2 topic list` shows `/dms/state`, publisher count 1.
- Plain String+JSON on purpose (no custom message package to build). A typed message is the next step if other nodes need to consume it.

## CARLA <-> ROS 2 options

| Option | Status |
|---|---|
| **A. CARLA's native ROS 2 on the Windows package** | **Not usable [verified].** The 0.9.16 Windows package accepts the API (`ros_name`, `ros_frame_id`, `ros_publish_tf` attributes exist; `sensor.enable_for_ros()` / `is_enabled_for_ros()` return True) and the server starts with `-ros2`, but: `CarlaUE4-Win64-Shipping.exe` contains no FastDDS/eprosima/DomainParticipant strings, and with `-ros2` + an enabled camera the server process opens zero UDP sockets (DDS discovery uses UDP). So the calls are stubs. Docs show the feature only with `./CarlaUnreal.sh --ros2` ([ROS2 native interface](https://carla-ue5.readthedocs.io/en/latest/ros2_native/), [quickstart extras](https://carla.readthedocs.io/en/0.9.16/ext_quickstart/)): a Linux build. [U] whether a Linux package has it compiled in. |
| **B. WSL ROS 2 node + `carla` Python client (recommended)** | **Works [verified]** since port 2000 is reachable from WSL (below): `~/dmsvenv` has the `carla` 0.9.16 wheel; from WSL `carla.Client("172.25.192.1", 2000)` connected, spawned a vehicle + camera and received a streamed frame (ports 2000-2002). A node there can publish ego speed/images and subscribe to `/dms/state` to drive the ego, with no DDS inside CARLA. Not written yet (YAGNI until needed). |
| **C. ROS 2 on Windows (Jazzy/Humble binaries) + `carla` client** | [U] Windows binaries exist for both distros; same architecture as B but without WSL networking. Not tried. |
| **D. Linux CARLA server + native ROS 2 (Humble container)** | [U] The only route to the native publisher. Needs the Linux package (~20 GB), Docker with GPU in WSL, and 6 GB VRAM is below the docs' 8 GB recommendation. Not tried. |

### Jazzy vs FastDDS (why native ROS 2 + Jazzy is a trap)

[doc] [carla-simulator/carla #9551](https://github.com/carla-simulator/carla/issues/9551) (open, Feb 2026): LibCarla is linked against
FastDDS 2.11.2. Humble ships 2.6.1 (compatible); Jazzy ships 2.14.5, and when a Jazzy node subscribes to CARLA topics, FastDDS
`recvfrom` blocks for seconds. CARLA publishes on the game thread, so the whole simulation stalls. Workaround from the thread: rebuild
CARLA with FastDDS 2.14.x (PR #9451 proposes 2.14.6). So: Humble for native ROS 2, or option B with Jazzy.

## Networking: WSL (NAT) -> Windows CARLA

Measured 2026-10-20: WSL 2.4.13, NAT mode, Windows host `172.25.192.1` (`ip route | grep default`), no `.wslconfig`.

- **Currently reachable [verified].** TCP connect to `172.25.192.1:2000` works and the client talks to the server (including sensor
  streaming). Reason: Windows has two inbound Allow rules named `CARLA UE4` (TCP and UDP, any port, profile **Public**, scoped to
  `...\CarlaUE4\Binaries\Win64\CarlaUE4-Win64-Shipping.exe`), which is what the Windows "allow access" prompt creates on first launch.
  I did not create or change any firewall rule. An earlier session saw the port blocked, i.e. this depends on that prompt having been
  accepted. The rule is program-scoped, so it survives server restarts.
- **If it is blocked again** (rule deleted, new CARLA folder, network profile changed), the decision is yours; I applied none:
  1. *Firewall rule for the WSL vNIC* (admin PowerShell), narrow:
     `New-NetFirewallRule -DisplayName "CARLA from WSL" -Direction Inbound -Action Allow -Protocol TCP -LocalPort 2000-2002 -InterfaceAlias "vEthernet (WSL (Hyper-V firewall))"`
     Pro: minimal, keeps NAT. Con: needs admin; the WSL gateway IP can change after a reboot (use `ip route` each time, do not hardcode).
     On current WSL there is also the Hyper-V firewall (`Get-NetFirewallHyperVVMSetting`), which can block inbound WSL->host traffic separately [U].
  2. *Mirrored networking* (`%UserProfile%\.wslconfig`: `[wsl2]` / `networkingMode=mirrored`, then `wsl --shutdown`). Pro: WSL reaches Windows
     services on `127.0.0.1`, so scripts need no host IP. Con: changes a system-wide WSL setting, may need a Hyper-V firewall setting too
     (`Set-NetFirewallHyperVVMSetting ... -DefaultInboundAction Allow`), and breaks setups relying on NAT. Not tested here.
- DDS between WSL and Windows would need multicast discovery, which NAT does not forward [U]; irrelevant for option B (only TCP to CARLA).

## Files

`scripts/ros2/dms_state_pub.py`, `scripts/ros2/record.sh`; bag `out/ros2/bag` (regenerated by `record.sh`).
