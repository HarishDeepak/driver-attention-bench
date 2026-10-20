"""CCRs-like sweep: ego drives at constant speed toward a stationary car; a simple forward collision warning (FCW)
fires at TTC < thr, thr depends on the DMS state when 'gated' (2.0 s attentive, 3.0 s inattentive; ungated: always 2.0 s).
Simple driver: reaction time after the warning 1.0 s (attentive) / 1.6 s (inattentive), then full brake at A_BRAKE = 8 m/s2 (kinematic, see ponytail note in run()).
ALL NUMBERS ARE ILLUSTRATIVE ASSUMPTIONS, not Euro NCAP values and not a certified function.
Needs a running CARLA 0.9.16 server (127.0.0.1:2000). Output: out/carla_fcw.csv, out/carla_fcw.png."""
import csv, itertools, math, sys, time
import carla
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

DT, RANGE, T_MAX = 0.05, 150.0, 40.0
THR = {"attentive": 2.0, "inattentive": 3.0}   # gated FCW TTC threshold [s]
RT = {"attentive": 1.0, "inattentive": 1.6}    # driver reaction time after warning [s]
UNGATED_THR = 2.0
A_BRAKE = 8.0                                  # driver full-brake deceleration [m/s2], assumption (dry asphalt ~8-9)
MAP = "Town04_Opt"                                # Town06 is not in the Windows base package; Town04 has a highway

def straight_start(m):
    """First lane waypoint with RANGE+10 m of straight, junction-free same-lane road behind it (ponytail: first hit, no scoring)."""
    for w in m.generate_waypoints(20.0):
        if w.is_junction: continue
        ok, c = True, w
        for _ in range(int((RANGE + 10) / 5)):
            p = c.previous(5.0)
            if not p or p[0].is_junction or abs((p[0].transform.rotation.yaw - w.transform.rotation.yaw + 180) % 360 - 180) > 1.0:
                ok = False; break
            c = p[0]
        if ok: return w
    raise SystemExit("no straight lane found")

def run(world, bp, wp, v_kmh, state, gated):
    """One approach; returns result dict. Ego speed is commanded every tick: v, then ramped down at A_BRAKE after the reaction time."""
    v = v_kmh / 3.6
    t_tf = wp.transform; t_tf.location.z += 0.3
    e_wp = wp.previous(RANGE)[0]; e_tf = e_wp.transform; e_tf.location.z += 0.3
    fwd = e_wp.transform.get_forward_vector()
    target = world.spawn_actor(bp.find("vehicle.audi.a2"), t_tf)
    ego = world.spawn_actor(bp.find("vehicle.tesla.model3"), e_tf)
    col = world.spawn_actor(bp.find("sensor.other.collision"), carla.Transform(), attach_to=ego)
    hit = []
    col.listen(lambda e: hit.append(e.frame))
    try:
        target.apply_control(carla.VehicleControl(hand_brake=True))
        half = target.bounding_box.extent.x + ego.bounding_box.extent.x
        thr = THR[state] if gated else UNGATED_THR
        t_warn = ttc_w = gap_w = t_brk = None
        min_gap, t, vc, impact = 1e9, 0.0, v, None
        world.tick()
        while t < T_MAX:
            gap = ego.get_location().distance(target.get_location()) - half
            min_gap = min(min_gap, gap)
            if t_warn is None and gap / v < thr:
                t_warn, ttc_w, gap_w = t, gap / v, gap
            if t_warn is not None and t_brk is None and t - t_warn >= RT[state]:
                t_brk = t
            if t_brk is not None: vc = max(0.0, vc - A_BRAKE * DT)
            # ponytail: kinematic braking. CARLA wheel physics gave 2.5-8 m/s2 for the same brake=1.0 because set_target_velocity
            # leaves the wheels unspun (tyre slip at hand-over); the speed is commanded instead. Real tyre/brake model: not covered.
            ego.set_target_velocity(carla.Vector3D(fwd.x * vc, fwd.y * vc, fwd.z * vc))  # 3D: ~3.5% grade, a flat velocity made the car fall off the road
            world.tick(); t += DT
            if hit: impact = vc * 3.6; break
            if vc == 0.0: break
        collided = bool(hit)
        return dict(v_kmh=v_kmh, state=state, fcw="gated" if gated else "ungated",
                    t_warn=t_warn, ttc_warn=ttc_w, gap_warn=gap_w, collided=collided,
                    impact_kmh=impact if collided else 0.0,
                    min_gap=0.0 if collided else min_gap)  # contact = zero gap by definition; the per-tick gap is quantised to v*dt
    finally:
        col.stop(); col.destroy(); ego.destroy(); target.destroy()

def main():
    client = carla.Client("127.0.0.1", 2000); client.set_timeout(60)
    world = client.get_world()
    orig = world.get_settings()
    t0, rows = time.time(), []
    try:
        if MAP not in world.get_map().name: world = client.load_world(MAP, map_layers=carla.MapLayer.Ground)  # ponytail: ground-only layer, 6 GB VRAM crashed on full Town04
        s = world.get_settings(); s.synchronous_mode = True; s.fixed_delta_seconds = DT; world.apply_settings(s)
        assert MAP in [m.split("/")[-1] for m in client.get_available_maps()]
        client.start_recorder(str(__import__("pathlib").Path("out/carla_fcw.log").resolve()))  # exact replay: client.replay_file(path, 0, 0, 0)
        wp = straight_start(world.get_map())
        print(f"map {world.get_map().name}, target at {wp.transform.location}, yaw {wp.transform.rotation.yaw:.1f}")
        for v, st, g in itertools.product([30, 50, 80], ["attentive", "inattentive"], [True, False]):
            r = run(world, world.get_blueprint_library(), wp, v, st, g); rows.append(r)
            print(f"  done {v} km/h {st} {r['fcw']}", file=sys.stderr)
    finally:
        client.stop_recorder()
        world.apply_settings(orig)
    with open("out/carla_fcw.csv", "w", newline="") as f:
        w = csv.DictWriter(f, rows[0].keys()); w.writeheader(); w.writerows(rows)
    print("illustrative assumptions: thr 2.0/3.0 s, reaction 1.0/1.6 s, full brake\n"
          "v   state        fcw      t_warn ttc  gap_w  coll impact min_gap")
    for r in rows:
        f = lambda x, p=1: "  -  " if x is None else f"{x:.{p}f}"
        print(f"{r['v_kmh']:<3} {r['state']:<12} {r['fcw']:<8} {f(r['t_warn'])}  {f(r['ttc_warn'])} {f(r['gap_warn']):>6} "
              f"{str(r['collided']):<5} {r['impact_kmh']:5.1f} {r['min_gap']:6.1f}")
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    combos = [(s, g) for s in THR for g in ("gated", "ungated")]
    for k, (a, key, lab) in enumerate([(ax[0], "impact_kmh", "impact speed [km/h]"), (ax[1], "min_gap", "min gap [m]")]):
        for i, (s, g) in enumerate(combos):
            ys = [next(r[key] for r in rows if r["v_kmh"] == v and r["state"] == s and r["fcw"] == g) for v in (30, 50, 80)]
            a.bar([j + (i - 1.5) * 0.2 for j in range(3)], ys, 0.2, label=f"{s}, {g}")
        a.set_xticks(range(3)); a.set_xticklabels(["30", "50", "80"]); a.set_xlabel("ego speed [km/h]"); a.set_ylabel(lab)
    ax[0].legend(fontsize=8); fig.suptitle("CCRs-like sweep, simple FCW (illustrative assumptions)"); fig.tight_layout()
    fig.savefig("out/carla_fcw.png", dpi=120)
    print(f"runtime {time.time() - t0:.0f} s")

if __name__ == "__main__":
    main()
