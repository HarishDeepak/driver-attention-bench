# CARLA 0.9.16 for this project: learning guide

Scope: how the three scripts in `scripts/` (`live_demo.py`, `carla_fcw.py`, `carla_video.py`) use CARLA, and the parts of the 0.9.16 docs that matter for them. Doc root: https://carla.readthedocs.io/en/0.9.16/ . Items marked [verify] were not confirmed in the docs we could read.

---

## 1. Mental model

CARLA is two programs talking over TCP.

```
 +---------------------------+         TCP 2000 / 2001        +------------------------+
 |  CARLA server (Unreal)    | <-----------------------------> |  Python client          |
 |  - physics, rendering     |   commands, queries, ticks      |  - carla.Client          |
 |  - sensors render frames  |   sensor data stream            |  - world / actors        |
 |  - Traffic Manager (TM)*  |                                 |  - your loop + OpenCV    |
 +---------------------------+                                 +------------------------+
   * the TM runs as a client-side module; it talks to the server in batches
```

- **Client** (`carla.Client`): connects, loads maps, gets the world, sends commands. Source: https://carla.readthedocs.io/en/0.9.16/core_world/
- **World** (`carla.World`): owns the actors, the map, weather, settings, and the tick. Recreated on `load_world()`; the old one is destroyed. Same page.
- **Actor**: anything in the sim: vehicles, sensors, the spectator. Spawned from a blueprint with a `carla.Transform`. Actors are *not* destroyed when the script exits. https://carla.readthedocs.io/en/0.9.16/core_actors/
- **Tick**: one simulation step. `world.tick()` advances the server one step.

### Sync vs async

| | Fixed time-step | Variable time-step |
|---|---|---|
| **Synchronous** (server waits for `tick()`) | Client rules the sim. Use this for sensors and reproducibility. | Docs call this "almost for sure a non-desirable state". |
| **Asynchronous** (server runs as fast as it can) | Good time references, runs fast. | Default. Not repeatable. |

Source: https://carla.readthedocs.io/en/0.9.16/adv_synchrony_timestep/

Rules from that page:
- In sync mode, always use a fixed time-step. The project uses `DT = 0.05` (20 Hz).
- Substepping: physics is split into substeps. Default is max 10 substeps of 0.01 s. Constraint: `fixed_delta_seconds <= max_substep_delta_time * max_substeps` (0.05 <= 0.1 holds). https://carla.readthedocs.io/en/0.9.16/adv_synchrony_timestep/#physics-substepping
- The Traffic Manager must be in sync mode too when the world is sync.

Loop shape in sync mode:

```
world.tick()  ->  server computes one step  ->  sensors deliver data  ->  your code reads it
   ^                                                                          |
   +--------------------------------------------------------------------------+
```

---

## 2. The 10 API patterns this project needs

### P1. Connect, set timeout, get world
```python
client = carla.Client("127.0.0.1", 2000)
client.set_timeout(60)                       # every network call gives up after 60 s
world = client.get_world()
if "Town04_Opt" not in world.get_map().name:
    world = client.load_world("Town04_Opt", map_layers=carla.MapLayer.Ground)
```
Pitfall: `load_world` destroys the current world and all its actors. Check the map name first, as the scripts do. The 6 GB GPU note in the scripts ("full Town04 crashed") is the project's own measurement, not from the docs. Source: https://carla.readthedocs.io/en/0.9.16/core_map/#changing-the-map

### P2. Turn on sync mode with a fixed step, restore it on exit
```python
orig = world.get_settings()
s = world.get_settings()
s.synchronous_mode, s.fixed_delta_seconds = True, 0.05
world.apply_settings(s)
# ... loop with world.tick() ...
world.apply_settings(orig)                   # in finally
```
Pitfall: if the script ends while sync is on, the server waits forever for a tick that never comes. Turn sync off (world and TM) before exit. Source: https://carla.readthedocs.io/en/0.9.16/adv_traffic_manager/#synchronous-mode

### P3. Sensor data via one queue per sensor, one tick per frame
```python
q = queue.Queue()
cam = world.spawn_actor(cam_bp, carla.Transform(carla.Location(-8, 0, 3)), attach_to=ego)
cam.listen(q.put)
while running:
    world.tick()
    img = q.get(timeout=10)                  # blocks until this tick's image arrives
```
Pitfall: GPU sensors (cameras) deliver data a couple of frames late, so the queue is the only thing keeping frames aligned. Use `timeout` so a dead server fails loudly. Source: https://carla.readthedocs.io/en/0.9.16/adv_synchrony_timestep/#client-server-synchrony and https://carla.readthedocs.io/en/0.9.16/core_sensors/

### P4. Spawn from spawn points, tag the ego with `role_name=hero`
```python
ebp = world.get_blueprint_library().find("vehicle.tesla.model3")
ebp.set_attribute("role_name", "hero")
tf = world.get_map().get_spawn_points()[0]
ego = world.try_spawn_actor(ebp, tf)         # None on collision, no exception
if ego is None: raise SystemExit("spawn blocked")
```
Pitfall: `spawn_actor` raises on collision; `try_spawn_actor` returns `None`. Spawn points may be occupied. `role_name = hero` is what the recorder collision query and hybrid mode look for. Sources: https://carla.readthedocs.io/en/0.9.16/core_actors/#spawning , https://carla.readthedocs.io/en/0.9.16/adv_recorder/#queries , https://carla.readthedocs.io/en/0.9.16/adv_traffic_manager/#hybrid-physics-mode

### P5. Hand a vehicle to the Traffic Manager, sync TM with the world
```python
tm = client.get_trafficmanager()             # default port 8000
tm.set_synchronous_mode(True)                # with world sync on
car.set_autopilot(True, tm.get_port())
tm.auto_lane_change(car, False)
tm.ignore_lights_percentage(car, 100)
tm.distance_to_leading_vehicle(car, 5.0)
```
Pitfall: TM is designed for sync mode; async gives "unexpected and undesirable results". Two TM-sync flags on different TMs break sync. Source: https://carla.readthedocs.io/en/0.9.16/adv_traffic_manager/#synchronous-mode

### P6. Pin a route with `set_path` (no random branches)
```python
path = [w.transform.location for w in waypoints_ahead]   # carla.Location list
tm.set_path(ego, path)
```
Why: at junctions the TM "chooses a path randomly". `set_path` takes locations and keeps the vehicle on them. The project builds the path every 10 m along its own lane. Source: https://carla.readthedocs.io/en/0.9.16/adv_traffic_manager/#localization and https://carla.readthedocs.io/en/0.9.16/tuto_G_traffic_manager/#specify-routes-for-vehicles

Pitfall: the TM path is endless and vehicles are not goal-oriented. Without `set_path`, an ego can take an exit. Real seen case in the project: the stop ended at an exit gore.

### P7. Direct control of the ego (brake, velocity)
```python
ego.apply_control(carla.VehicleControl(brake=1.0, hand_brake=True))
ego.set_autopilot(False, tm.get_port())      # TM would override your brake
ego.set_target_velocity(carla.Vector3D(vx, vy, vz))  # m/s, sets speed directly
```
Pitfall: while the TM controls a vehicle, your `apply_control` is overridden. Turn autopilot off first. Setting velocity directly is not physics braking. Source: https://carla.readthedocs.io/en/0.9.16/core_actors/#vehicles

### P8. Batch spawn and batch destroy
```python
tm_port = tm.get_port()
batch = [carla.command.SpawnActor(bp, tf).then(
            carla.command.SetAutopilot(carla.command.FutureActor, True, tm_port))
         for bp, tf in pairs]
results = client.apply_batch_sync(batch, True)
client.apply_batch([carla.command.DestroyActor(x) for x in vehicles])
```
Pitfall: `apply_batch_sync` is the only batch call that guarantees execution or a failure response, and the docs call it out for determinism. Single calls can be lost "in a busy simulation". Batches run in one frame. Sources: https://carla.readthedocs.io/en/0.9.16/adv_synchrony_timestep/#physics-determinism , https://carla.readthedocs.io/en/0.9.16/tuto_G_traffic_manager/

### P9. Record, then replay with a camera on the ego
```python
client.start_recorder(str(abs_path))                 # absolute path; relative names go to CarlaUE4/Saved
# ... run ...
client.stop_recorder()
client.replay_file(str(abs_path), 0, 0, ego_id)      # start=0, duration=0 (all), camera=actor id
world.tick()                                          # needed in sync mode before actors appear
```
Pitfall: at the end of a replay, vehicles are set to autopilot and pedestrians stop. Any script that expects a static scene after the replay is wrong. Camera id `0` lets the spectator move freely. Sources: https://carla.readthedocs.io/en/0.9.16/adv_recorder/#simulation-playback , https://carla.readthedocs.io/en/0.9.16/adv_synchrony_timestep/#physics-determinism

### P10. Safe shutdown order
```python
finally:
    client.stop_recorder()
    for v in vehicles: v.set_autopilot(False, tm.get_port())
    tm.set_synchronous_mode(False)
    world.apply_settings(orig)               # sync off on the world
    for s in sensors: s.stop(); s.destroy()
    for a in actors:  a.destroy()
```
Pitfall: order matters. Turn TM autopilot off and sync off before destroying actors, or the libcarla side can abort (the comment in `live_demo.py` says so). `destroy()` blocks the simulator until it finishes. Actors are not cleaned up when the script exits. Sources: https://carla.readthedocs.io/en/0.9.16/core_actors/#destruction , https://carla.readthedocs.io/en/0.9.16/adv_traffic_manager/#stopping-a-traffic-manager

### Map layers (used by P1 and carla_video): the 6 GB note
```python
world = client.load_world("Town04_Opt", map_layers=carla.MapLayer.Ground)
for layer in (carla.MapLayer.Props, carla.MapLayer.StreetLights,
              carla.MapLayer.Buildings, carla.MapLayer.Foliage):
    world.load_map_layer(layer)              # add one by one
```
Docs: layered maps have a `_Opt` suffix; the minimum layout (roads, sidewalks, lights, signs) always stays; layers can be loaded and unloaded. Docs give no VRAM figure, so the 6 GB claim is [verify]. Source: https://carla.readthedocs.io/en/0.9.16/core_map/#layered-maps

---

## 3. The project scripts against the docs

### live_demo.py
Uses: P1, P2, P3 (chase camera), P4 (hero), P5 + P6 (TM, set_path, lane changes off), P7 (ego brake), P9 (recorder), P10 (shutdown), map layers.

Improvements the docs suggest:
1. **Collision events for the recorder.** The recorder only logs collisions if a collision sensor is attached to the vehicle. The script attaches none, so `show_recorder_collisions(log, "h", "a")` will be empty. Attach `sensor.other.collision` to the ego. Source: https://carla.readthedocs.io/en/0.9.16/adv_recorder/#collisions
2. **Brake lights on the TM cars.** By default the TM does not update vehicle lights. `tm.update_vehicle_lights(car, True)` on each traffic car makes the scene read better to the camera. Source: https://carla.readthedocs.io/en/0.9.16/adv_traffic_manager/#delegating-the-traffic-manager-to-automatically-update-vehicle-lights
3. **Hybrid physics for the far traffic.** `tm.set_hybrid_physics_mode(True)` with `set_hybrid_physics_radius(r)` disables physics for cars far from the hero. This cuts load; the docs say vehicles outside the radius move by teleport. Check it does not change the TTC/escalation timing. Source: https://carla.readthedocs.io/en/0.9.16/adv_traffic_manager/#hybrid-physics-mode

Also note: the script loads the world before enabling sync. The determinism section says sync must be on before loading the world. The script does not claim determinism, so this is only a problem if you want exact repeats.

### carla_fcw.py
Uses: P1, P2, P4, P7 (target `hand_brake`, ego `set_target_velocity` per tick), P9 (recorder), plus collision sensor (`sensor.other.collision`) on the ego.

Improvements the docs suggest:
1. **Sync before load.** The docs say sync must be enabled before loading or reloading the world for determinism. Move the sync setting ahead of `load_world`, and call `set_random_device_seed` on the TM if you want repeat runs. Source: https://carla.readthedocs.io/en/0.9.16/adv_synchrony_timestep/#physics-determinism
2. **Wheel physics.** The script notes that physics braking gave 2.5-8 m/s2 and switched to commanded velocity. The docs offer a per-vehicle switch, `physics_control.use_sweep_wheel_collision = True`, which checks the full wheel volume. Test it before replacing the kinematic brake. Source: https://carla.readthedocs.io/en/0.9.16/core_actors/#vehicles
3. **Keep velocity in the log.** `start_recorder(path, True)` stores linear and angular velocity, which lets you check speed and deceleration from the log instead of only from the CSV. Source: https://carla.readthedocs.io/en/0.9.16/adv_recorder/#recording

### carla_video.py
Uses: P1, map layers, P9 (replay with camera id = hero), P3 (camera queue after replay), P10 (partial).

Improvements the docs suggest:
1. **Stop the camera before destroying it.** The script calls `cam.destroy()` without `cam.stop()`. The sensor docs define `stop()` as the way to stop listening. Source: https://carla.readthedocs.io/en/0.9.16/core_sensors/#data
2. **Check the log before replay.** `client.show_recorder_file_info(log)` prints map, date and frame count. A cheap check against the wrong `.log`. Source: https://carla.readthedocs.io/en/0.9.16/adv_recorder/#recorded-file
3. **Match the replay step to the recording step.** The docs say a fixed step replays cleanly when the server uses the same step as the original run. The script hard-codes `DT = 0.05`. Read the step from the run (or store it next to the log) and fail if it differs. Source: https://carla.readthedocs.io/en/0.9.16/adv_synchrony_timestep/#tips-when-recording-the-simulation

Pairing frames: the script pops one image per tick and assumes it belongs to that tick. The docs say GPU sensors lag by a couple of frames. Compare `img.frame` to `world.get_snapshot().frame` and drop stale frames. Source: https://carla.readthedocs.io/en/0.9.16/core_sensors/#data

---

## 4. What the bigger tools would add here, and their cost

**ScenarioRunner + OpenSCENARIO.** The docs state, for the 0.9.12 overview page, that ScenarioRunner is a separate install, supports predefined scenarios, Python scenarios and OpenSCENARIO 1.0, and that OpenSCENARIO is for multi-vehicle maneuvers. The 0.9.16 pages for ScenarioRunner returned 404 in our reads, so this is [verify] for 0.9.16. Source: https://carla.readthedocs.io/en/0.9.12/ts_traffic_simulation_overview/ . The Traffic Manager page says ScenarioRunner starts its own TM in sync mode. Source: https://carla.readthedocs.io/en/0.9.16/adv_traffic_manager/#synchronous-mode
- Adds: repeatable scripted cut-ins and stops, which would make FCW tests more standard than the hand-built `carla_fcw.py` setup.
- Cost: extra install and Python setup, and learning OpenSCENARIO. No licence cost. Honest read: for this project, it would replace the hand-built scenario, not add a capability. Do it only if the scenario count grows.

**ROS 2.** The ecosystem page says the native ROS 2 interface is built into the server and recommended for latency. The ROS bridge works with ROS 1 and ROS 2 and adds latency. Source: https://carla.readthedocs.io/en/latest/ecosys_ros/ . The quickstart lists a `--ros2` server flag. Source: https://carla.readthedocs.io/en/latest/ext_quickstart/ . Both pages are `latest`, not 0.9.16 [verify].
- Adds: publishing the DMS/escalation output as ROS topics, if a ROS stack ever consumes it. The repo already has a ROS 2 Humble side project (see memory note `project_ros2_manip`), so the install is not new work.
- Cost: a ROS 2 install, Docker or Ubuntu, and version pinning. Free, but the setup time is real. Not needed for the CSV/video outputs this project produces.

**NVIDIA Cosmos / NuRec.** The AI rendering page says Cosmos Transfer is a style-transfer tool built on NVIDIA's Cosmos Transfer1 model. It turns CARLA sequences plus text prompts into photoreal video, for sim-to-real and edge-case testing. Source: https://carla.readthedocs.io/en/latest/ai_rendering/ (fetched from `latest`; the 0.9.16 URL returned 404 to our fetch [verify]). The page gives no hardware, licence or cost figures, and we did not read the separate Cosmos docs or NuRec pages [verify].
- Adds: photoreal variants of the in-cabin or road footage for robustness tests.
- Cost, honest: a large generative model, so it needs a big GPU; a 6 GB card is very unlikely to run it locally [verify with the Cosmos requirements]. Free-tier use depends on NVIDIA's licence and any hosted endpoint [verify]. Not worth it for this portfolio unless a free hosted option is confirmed.

**Leaderboard.** The maps page links the CARLA Leaderboard and says Town08 and Town09 are the secret towns used there. Source: https://carla.readthedocs.io/en/0.9.16/core_map/#carla-maps . We did not read the leaderboard rules [verify]. It is an evaluation for driving agents, not a driver-monitoring tool, so it does not fit this project.

---

## 5. Fifteen interview questions

1. **What is the difference between the CARLA server and the client?** Server runs the Unreal simulation and renders sensors; client is your Python script that sends commands and reads data over TCP 2000/2001.
2. **What does `world.tick()` do in sync mode?** It tells the server to compute one step. Without it, the server waits forever.
3. **Why fixed time-step with sync mode?** Physics and sensor data line up with simulated time, so results are repeatable and the client can run slower than real time.
4. **What is substepping?** Physics runs in smaller substeps (default 0.01 s, up to 10) inside each frame, so a 0.05 s frame is still physically accurate.
5. **What breaks if sync is left on when the script exits?** The server waits for a tick that never comes and blocks.
6. **Why must the TM be in sync mode too?** TM is built for sync; async gives unexpected results.
7. **What is `set_path` for?** To pin an autopilot vehicle to a route; without it the TM picks random branches at junctions.
8. **Difference between `spawn_actor` and `try_spawn_actor`?** The first raises on collision; the second returns `None`.
9. **Why set `role_name = hero`?** Recorder collision queries and hybrid mode use it to find the ego.
10. **Why use a queue per sensor in sync mode?** GPU sensors deliver data a few frames late; the queue keeps each frame paired with its tick.
11. **What does `apply_batch_sync` give you that a single call does not?** One frame for all commands and a guaranteed execution or failure response.
12. **What happens to vehicles at the end of a replay?** They are set to autopilot; pedestrians stop.
13. **Why does the recorder need a collision sensor for collision queries?** Collisions are logged from attached collision detectors.
14. **Map layers: why load them one by one?** The docs say the minimum layout always loads and layers can be loaded and unloaded; the project loads them one at a time to fit its GPU [project measurement, not docs].
15. **What is the difference between the recorder and determinism?** The recorder replays a stored log; determinism means the same inputs give the same run (needs sync, fixed step, a TM seed and a reloaded world).

Short answers are in the list above; each one is backed by the sections in this guide.

---

## Pitfalls summary (top 5)

1. Sync on with no ticking loop, or exit with sync still on: the server blocks.
2. `destroy()` order: turn autopilot and TM sync off before destroying; destroy blocks.
3. Replay ends with vehicles on autopilot, not static.
4. Sensor frames in sync mode lag; pair by the queue/tick, check `frame`.
5. TM without `set_path` takes random branches; TM is endless, not goal-oriented.

## Places where the scripts differ from the docs

- `carla_video.py` destroys its camera without `stop()` (sensor docs).
- `carla_fcw.py` and `live_demo.py` load the world before enabling sync; the docs want sync before the world load for determinism. Only matters for exact repeats.
- `live_demo.py` attaches no collision sensor, so recorder collision queries return nothing.
- The "6 GB VRAM" and "crashed on full Town04" notes are project measurements, not in the docs.

Not read in this pass: the 0.9.16 ScenarioRunner, ROS 2 native and Cosmos pages (404 or not fetched), the leaderboard rules, the NuRec page, and the full tuto_G tutorial list.
