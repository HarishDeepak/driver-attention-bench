"""Escalation for an inattentive / unresponsive driver (Euro NCAP 2026 idea): warn -> slow -> minimal-risk stop in lane.
Pure Python, no CARLA: step() returns (level, target speed) and the CARLA demo applies it."""

# [SD-202] unresponsive driver: eyes closed >= 13 s or gaze away >= 14 s. SLOW_S, SLOW_KMH, MRM_DECEL, RECOVER_S are our assumptions.
UNRESP_CLOSED_S, UNRESP_AWAY_S = 13.0, 14.0
SLOW_S, SLOW_KMH, MRM_DECEL, RECOVER_S = 8.0, 60.0, 2.0, 2.0


class Escalation:
    def __init__(self, cruise_kmh=80.0):
        self.cruise, self.level, self.target = cruise_kmh, "ok", cruise_kmh
        self.closed0 = self.away0 = self.inatt0 = self.att0 = self.t_last = None

    def step(self, t, closed, away, warned, speed_kmh):
        """closed/away: DMS booleans this frame; warned: kinds fired this frame (dms_core.Monitor.step); speed in km/h."""
        dt = 0.0 if self.t_last is None else t - self.t_last
        self.t_last = t
        self.closed0 = (t if self.closed0 is None else self.closed0) if closed else None
        self.away0 = (t if self.away0 is None else self.away0) if away else None
        inatt = closed or away
        self.inatt0 = (t if self.inatt0 is None else self.inatt0) if inatt else None
        self.att0 = None if inatt else (t if self.att0 is None else self.att0)

        if self.level == "stopped":
            return self.level, 0.0
        if self.level != "ok" and self.att0 is not None and t - self.att0 >= RECOVER_S:
            self.level, self.target = "ok", self.cruise
        elif (self.closed0 is not None and t - self.closed0 >= UNRESP_CLOSED_S) or \
                (self.away0 is not None and t - self.away0 >= UNRESP_AWAY_S) or self.level == "mrm":
            if self.level != "mrm":  # start the ramp from the current speed, once
                self.target = min(self.target, speed_kmh)
            self.level = "mrm"
            # time-based ramp: feeding the measured speed back every tick let any controller overshoot ratchet the target down
            # (seen in CARLA: ~5 m/s2 instead of 2)
            self.target = max(0.0, self.target - MRM_DECEL * 3.6 * dt)
            if self.target == 0.0 and speed_kmh < 1.0:
                self.level = "stopped"
        elif self.inatt0 is not None and t - self.inatt0 >= SLOW_S and self.level in ("warning", "slowing"):
            self.level, self.target = "slowing", SLOW_KMH
        elif warned and self.level == "ok":
            self.level = "warning"
        return self.level, self.target


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from dms_core import Monitor

    def sim(closed_until, total, hz=20):
        e, m, v, seen = Escalation(), Monitor(), 80.0, []
        for i in range(int(total * hz)):
            t = i / hz
            c = t < closed_until
            lvl, tgt = e.step(t, c, False, m.step(t, c, False), v)
            v = tgt  # ideal speed tracking
            if not seen or seen[-1][1] != lvl:
                seen.append((round(t, 2), lvl))
        return seen

    unresp = sim(closed_until=1e9, total=40)
    assert [l for _, l in unresp] == ["ok", "warning", "slowing", "mrm", "stopped"], unresp
    assert dict((l, t) for t, l in unresp)["mrm"] == 13.0
    # the MRM ramp is 2 m/s2 even if the car brakes harder than asked (speed below target must not pull the target down)
    e, m = Escalation(), Monitor()
    for i in range(int(13 * 20) + 1):
        e.step(i / 20, True, False, m.step(i / 20, True, False), 60.0)
    lvl0, tg0 = e.level, e.target
    _, tg1 = e.step(13.0 + 1.0, True, False, [], 30.0)  # 1 s later, car already far slower than the target
    assert lvl0 == "mrm" and abs((tg0 - tg1) - 2.0 * 3.6 * 1.0) < 1e-6, (lvl0, tg0, tg1)
    recover = sim(closed_until=5.0, total=10)
    assert [l for _, l in recover] == ["ok", "warning", "ok"], recover
    print("escalation OK", unresp, recover)
