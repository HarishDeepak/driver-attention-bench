"""Render one synthetic DMS clip (Blender 4.5 + MPFB2) with exact labels.

blender -b -P tools/blender/make_driver.py -- --clip synth01_microsleep --seed 1 [--acc none|sunglasses|mask] [--still T]
Writes data/synthetic/<clip>.mp4 and data/synthetic/_labels/<clip>.csv (merged into labels.csv by render_all.sh). Protocol = docs/details/RECORDING.md.
Convention: yaw/pitch/gaze angles are driver-centric degrees; +yaw = to the driver's right (mirror), +pitch = down.
"""
import argparse, csv, glob, math, os, random, subprocess, sys, shutil
import bpy
from mathutils import Matrix, Vector

ap = argparse.ArgumentParser()
ap.add_argument("--clip", required=True)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--acc", default="none", choices=["none", "sunglasses", "mask"])
ap.add_argument("--cam-az", type=float, default=25, help="camera azimuth, deg to the driver's right of straight ahead")
ap.add_argument("--cam-dist", type=float, default=0.6)
ap.add_argument("--cam-el", type=float, default=-8, help="camera elevation vs eye level, deg (negative = below)")
ap.add_argument("--samples", type=int, default=12)
ap.add_argument("--still", type=float, help="render one PNG at this time (s) to tmp_still.png and exit")
a = ap.parse_args(sys.argv[sys.argv.index("--") + 1:])

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "data", "synthetic")
FPS, W, H, CALIB = 25, 640, 480, 3.5  # 3 s calibration + 0.5 s margin
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- script (what the driver does)
# channel -> (value in deg or closure 0..1, ramp s). event(ch, value, hold_s) advances the cursor by ramp + hold + ramp.
RAMP = dict(lid=0.12, blink=0.06, head=0.35, eye=0.08)
EV, cur = [], [CALIB]


def ev(ch, val, hold, ramp):
    EV.append((ch, val, cur[0], ramp, hold))
    cur[0] += 2 * ramp + hold


def gap(s):
    cur[0] += s


def closed(hold):  ev("lid", 1.0, hold, RAMP["lid"])
def blink():       ev("lid", 1.0, 0.08, RAMP["blink"])
def head(yaw, pitch, hold): ev("yaw", yaw, hold, RAMP["head"]); EV.append(("pitch", pitch, cur[0] - 2 * RAMP["head"] - hold, RAMP["head"], hold))
def eyes(yaw, pitch, hold): ev("eyaw", yaw, hold, RAMP["eye"]); EV.append(("epitch", pitch, cur[0] - 2 * RAMP["eye"] - hold, RAMP["eye"], hold))


CLIPS = {
    "synth01_microsleep": lambda: [blink(), gap(.8), blink(), gap(.8), blink(), gap(1.0), closed(2), gap(4), closed(2), gap(3)],
    "synth02_sleep": lambda: [closed(5), gap(4), closed(4), gap(3)],
    "synth03_owl": lambda: [head(50, 0, 4), gap(4), head(-40, 25, 4), gap(3), head(50, 0, 1), gap(3)],
    "synth04_lizard": lambda: [eyes(42, 0, 4), gap(4), eyes(0, 38, 4), gap(3)],
    "synth05_sunglasses": lambda: [closed(2), gap(4), closed(5), gap(4), head(50, 0, 4), gap(3)],
    "synth06_mask": lambda: [closed(2), gap(4), closed(5), gap(4), head(50, 0, 4), gap(3)],
}
CLIPS[a.clip]()
DUR = cur[0] + 0.5


def sm(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def state(t):
    s = dict(lid=0.0, yaw=0.0, pitch=0.0, eyaw=0.0, epitch=0.0)
    for ch, val, t0, r, hold in EV:
        s[ch] += val * min(sm((t - t0) / r), sm((t0 + 2 * r + hold - t) / r))  # events never overlap per channel
    return s


def labels():
    """closed: lid >= 0.95 (fully closed); away: head or eyes more than 30 deg off forward (1 ms sampling). Blinks and the 1 s glance are not labelled."""
    rows, n = [], int(DUR * 1000)
    for kind, f in (("closed", lambda s: s["lid"] >= 0.95), ("away", lambda s: math.hypot(s["yaw"] + s["eyaw"], s["pitch"] + s["epitch"]) > 30)):
        on = None
        for i in range(n + 1):
            v, t = f(state(i / 1000)), i / 1000
            if v and on is None: on = t
            if not v and on is not None:
                rows.append((kind, on, t)); on = None
    # the protocol does not label blinks or the 1 s glance: keep closed >= 1 s, away >= 2 s
    return [(k, s, e) for k, s, e in rows if e - s >= (2.0 if k == "away" else 1.0)]


# ---------------------------------------------------------------- scene
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.preferences.addon_enable(module="bl_ext.user_default.mpfb")
from bl_ext.user_default.mpfb.services.humanservice import HumanService
from bl_ext.user_default.mpfb.services.targetservice import TargetService

DATA = os.path.join(bpy.utils.extension_path_user("bl_ext.user_default.mpfb"), "data")
rng = random.Random(a.seed)
male, race, midage = rng.random() < 0.5, rng.choice(["caucasian", "asian", "african"]), rng.random() < 0.5
macro = dict(gender=1.0 if male else 0.0, age=0.62 if midage else 0.5, muscle=rng.uniform(.35, .65), weight=rng.uniform(.35, .6),
             proportions=rng.uniform(.4, .6), height=rng.uniform(.4, .65), cupsize=.5, firmness=.5,
             race={r: (0.8 if r == race else 0.1) for r in ("asian", "caucasian", "african")})
skin = f"{'middleage' if midage else 'young'}_{race}_{'male' if male else 'female'}"
print("HUMAN", a.seed, macro["gender"], race, skin)

bm = HumanService.create_human(macro_detail_dict=macro)
rig = HumanService.add_builtin_rig(bm, "default")
HumanService.set_character_skin(f"{DATA}/skins/{skin}/{skin}.mhmat", bm, skin_type="MAKESKIN")


def asset(sub, name, kind, **kw):
    return HumanService.add_mhclo_asset(f"{DATA}/{sub}/{name}/{name}.mhclo", bm, asset_type=kind, **kw)


def mhclo_uuid(path):
    return next(l.split()[1] for l in open(path) if l.startswith("uuid"))


eye_mat = rng.choice(["blue", "brown", "brownlight"])
eyes_path = f"{DATA}/eyes/high-poly/high-poly.mhclo"
asset("eyes", "high-poly", "Eyes", alternative_materials={mhclo_uuid(eyes_path): f"eyes/materials/{eye_mat}.mhmat"})
asset("eyebrows", f"eyebrow{rng.randint(1, 12):03d}", "Eyebrows")
hair = rng.choice(["short01", "short02", "short03", "short04"] if male else ["long01", "ponytail01"])
asset("hair", hair, "Hair")
asset("clothes", "male_casualsuit01" if male else "female_casualsuit01", "Clothes")

# eyelid closure: MakeHuman face units shipped with MPFB2, loaded as shape keys on the basemesh (value 1 = closed)
import bl_ext.user_default.mpfb as mpfb
UNITS = os.path.join(os.path.dirname(mpfb.__file__), "data", "targets", "expression", "units", race)
KEYS = [TargetService.load_target(bm, os.path.join(UNITS, f"eye-{s}-closure.target.gz"), weight=0.0, name=f"close_{s}") for s in ("left", "right")]
for k in KEYS: k.slider_max = 1.5

# ---------------------------------------------------------------- accessories (simple meshes rigidly bound to the head bone)
eyeL, eyeR = (rig.data.bones[f"eye.{s}"].head_local.copy() for s in "LR")
mid = (eyeL + eyeR) / 2


def bind_head(obj, color, rough=0.3, alpha=1.0):
    obj.parent = rig
    g = obj.vertex_groups.new(name="head")
    g.add(list(range(len(obj.data.vertices))), 1.0, "REPLACE")
    m = obj.modifiers.new("Armature", "ARMATURE"); m.object = rig
    mat = bpy.data.materials.new(obj.name); mat.use_nodes = True
    b = mat.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value, b.inputs["Roughness"].default_value = color, rough
    obj.data.materials.append(mat)


def box(name, c0, c1, th):
    """thin bar between two armature-local points"""
    bpy.ops.mesh.primitive_cube_add(size=1)
    o = bpy.context.active_object; o.name = name
    d = c1 - c0
    o.location = rig.matrix_world.inverted() @ (rig.matrix_world @ ((c0 + c1) / 2))
    o.scale = (th, d.length, th)
    o.rotation_euler = d.to_track_quat("Y", "Z").to_euler()
    return o


if a.acc == "sunglasses":
    for e in (eyeL, eyeR):
        bpy.ops.mesh.primitive_uv_sphere_add(segments=24, ring_count=12, radius=1)
        o = bpy.context.active_object; o.name = "lens"
        o.location, o.scale = e + Vector((0, -0.030, 0)), (0.029, 0.003, 0.023)
        bind_head(o, (0.01, 0.01, 0.012, 1), rough=0.05)
    fy = mid.y - 0.030
    parts = [box("bridge", Vector((eyeR.x, fy, eyeL.z + 0.008)), Vector((eyeL.x, fy, eyeL.z + 0.008)), 0.003)]
    for e in (eyeL, eyeR):
        s = 1 if e.x > 0 else -1
        parts += [box("arm", Vector((e.x + s * 0.029, fy, e.z)), Vector((e.x + s * 0.05, e.y + 0.09, e.z + 0.005)), 0.004),
                  box("rim", Vector((e.x - 0.03, fy - 0.001, e.z + 0.021)), Vector((e.x + 0.03, fy - 0.001, e.z + 0.021)), 0.003)]
    for p in parts: bind_head(p, (0.02, 0.02, 0.02, 1), rough=0.4)
elif a.acc == "mask":
    bpy.ops.mesh.primitive_grid_add(x_subdivisions=24, y_subdivisions=18, size=1)
    o = bpy.context.active_object; o.name = "mask"
    o.scale = (0.115, 0.11, 1)
    o.rotation_euler = (math.pi / 2, 0, 0)  # grid plane faces -Y (towards the camera, away from the face)
    o.location = Vector((mid.x, mid.y - 0.12, mid.z - 0.072))
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    sw = o.modifiers.new("fit", "SHRINKWRAP")
    sw.target, sw.wrap_method, sw.project_limit, sw.use_project_y, sw.use_negative_direction, sw.use_positive_direction = bm, "PROJECT", 0, True, False, True
    sw.offset = 0.004
    bpy.context.view_layer.objects.active = o
    bpy.ops.object.modifier_apply(modifier="fit")
    bind_head(o, (0.3, 0.55, 0.75, 1), rough=0.9)

# ---------------------------------------------------------------- stage: camera, light, backdrop
scn = bpy.context.scene
scn.render.engine = "BLENDER_EEVEE_NEXT"
scn.eevee.taa_render_samples = a.samples
scn.render.resolution_x, scn.render.resolution_y, scn.render.fps = W, H, FPS
scn.render.image_settings.file_format = "PNG"
scn.view_settings.view_transform = "Standard"
scn.world = bpy.data.worlds.new("w"); scn.world.use_nodes = True
scn.world.node_tree.nodes["Background"].inputs[0].default_value = (0.01, 0.01, 0.012, 1)

wmid = rig.matrix_world @ mid
az, el = math.radians(a.cam_az), math.radians(a.cam_el)
right = Vector((-1, 0, 0)); fwd = Vector((0, -1, 0))
cam_loc = wmid + a.cam_dist * (math.cos(el) * (math.cos(az) * fwd + math.sin(az) * right) + Vector((0, 0, math.sin(el))))
bpy.ops.object.camera_add(location=cam_loc)
cam = bpy.context.active_object; scn.camera = cam
cam.data.lens, cam.data.sensor_width = 30, 36
target = bpy.data.objects.new("t", None); scn.collection.objects.link(target)
target.location = wmid + Vector((0, 0, -0.03))
c = cam.constraints.new("TRACK_TO"); c.target, c.track_axis, c.up_axis = target, "TRACK_NEGATIVE_Z", "UP_Y"

bpy.ops.object.light_add(type="AREA", location=wmid + Vector((-0.45, -0.7, 0.4)))  # front, above, towards the camera side
L = bpy.context.active_object; L.data.energy, L.data.size = 30, 0.7
L.rotation_euler = (wmid + Vector((0, 0, 0)) - L.location).to_track_quat("-Z", "Y").to_euler()
bpy.ops.object.light_add(type="AREA", location=wmid + Vector((0.7, -0.2, 0.1)))  # weak fill from the other side
F = bpy.context.active_object; F.data.energy, F.data.size = 12, 0.7
F.rotation_euler = (wmid - F.location).to_track_quat("-Z", "Y").to_euler()

bpy.ops.mesh.primitive_plane_add(size=6, location=(0, mid.y + 0.9, wmid.z), rotation=(math.pi / 2, 0, 0))  # dark cabin wall
bpy.ops.mesh.primitive_cube_add(size=1, location=(0, mid.y + 0.16, wmid.z - 0.1))  # headrest
hr = bpy.context.active_object; hr.scale = (0.3, 0.1, 0.45)
for o in bpy.context.selected_objects + [bpy.data.objects["Plane"]]:
    m = bpy.data.materials.new("cabin"); m.use_nodes = True
    m.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.012, 0.012, 0.014, 1)
    m.node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value = 0.9
    o.data.materials.append(m)

# ---------------------------------------------------------------- pose
CHAIN = {"neck01": .15, "neck02": .15, "neck03": .2, "head": .5}
POSE = rig.pose.bones
for pb in POSE: pb.rotation_mode = "QUATERNION"


def turn(name, share_yaw, share_pitch):
    """world-axis rotation (yaw about Z, +yaw = driver's right = -Z angle; pitch about X, +pitch = down) expressed in the bone's local frame"""
    pb = POSE[name]
    Rb = pb.bone.matrix_local.to_3x3()
    R = Matrix.Rotation(-share_yaw, 3, "Z") @ Matrix.Rotation(share_pitch, 3, "X")
    pb.rotation_quaternion = (Rb.inverted() @ R @ Rb).to_quaternion()


def pose(t):
    s = state(t)
    for n, sh in CHAIN.items(): turn(n, math.radians(s["yaw"]) * sh, math.radians(s["pitch"]) * sh)
    for n in ("eye.L", "eye.R"): turn(n, math.radians(s["eyaw"]), math.radians(s["epitch"]))
    for k in KEYS: k.value = s["lid"]
    bpy.context.view_layer.update()


if a.still is not None:
    pose(a.still)
    scn.render.filepath = os.path.join(ROOT, "tools", "blender", "tmp_still.png")
    bpy.ops.render.render(write_still=True)
    sys.exit()

# ---------------------------------------------------------------- render, encode, label
frames = os.path.join(OUT, "_frames_" + a.clip)
os.makedirs(frames, exist_ok=True)
for i in range(int(DUR * FPS)):
    pose(i / FPS)
    scn.render.filepath = os.path.join(frames, f"{i:05d}.png")
    bpy.ops.render.render(write_still=True)
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", os.path.join(frames, "%05d.png"),
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", os.path.join(OUT, a.clip + ".mp4")], check=True)
shutil.rmtree(frames)

os.makedirs(os.path.join(OUT, "_labels"), exist_ok=True)  # one file per clip (parallel-safe); render_all.sh merges them into labels.csv
with open(os.path.join(OUT, "_labels", a.clip + ".csv"), "w", newline="") as fh:
    csv.writer(fh).writerows([a.clip, k, f"{s:.2f}", f"{e:.2f}"] for k, s, e in labels())
print("DONE", a.clip, DUR, labels())
