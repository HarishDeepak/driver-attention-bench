"""Stage-0 check: can this GPU run CARLA? Spawns a car + RGB camera, grabs frames, prints VRAM and FPS."""
import subprocess, sys, time, queue
import carla

host = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1"
vram = lambda: subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader"],
                              capture_output=True, text=True).stdout.strip()

client = carla.Client(host, 2000); client.set_timeout(60)
world = client.load_world("Town01")
print("map:", world.get_map().name, "| VRAM:", vram())
bp = world.get_blueprint_library()
car = world.spawn_actor(bp.filter("vehicle.lincoln.mkz*")[0], world.get_map().get_spawn_points()[0])
cam_bp = bp.find("sensor.camera.rgb"); cam_bp.set_attribute("image_size_x", "640"); cam_bp.set_attribute("image_size_y", "480")
cam = world.spawn_actor(cam_bp, carla.Transform(carla.Location(x=1.5, z=1.4)), attach_to=car)
q = queue.Queue(); cam.listen(q.put)
car.set_autopilot(True)
t0, n = time.time(), 0
while time.time() - t0 < 15:
    q.get(timeout=10); n += 1
print(f"frames: {n} in 15 s = {n/15:.1f} fps | VRAM under load: {vram()}")
cam.stop(); cam.destroy(); car.destroy()
print("SMOKE_OK")
