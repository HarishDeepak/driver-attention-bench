"""Euro NCAP SD-202-style driver-state warnings. Pure Python: no ROS or CV deps, so the evaluator can reuse it."""

# SD-202 v1.2: sleep = 3 s eye closure (fixed); microsleep 1-2 s and long distraction 3-4 s are OEM-declared; tolerance 0.2 s
MICROSLEEP_S, SLEEP_S, LONG_DISTRACTION_S, TOL_S = 1.5, 3.0, 3.0, 0.2


class Monitor:
    """Online version: step(t, eyes_closed, gaze_off_road) -> kinds that fire on this sample (one per episode).
    eyes_closed None = eyes unknown: not counted as closed, and it restarts the closure timer (no warning from unobserved eyes)."""

    def __init__(self, microsleep=MICROSLEEP_S, sleep=SLEEP_S, long_dist=LONG_DISTRACTION_S):
        self.thr = {"microsleep": microsleep, "sleep": sleep, "long_distraction": long_dist}
        self.closed0 = self.away0 = self.last_away = None
        self.fired = set()

    BRIDGE_S = 0.5  # ponytail: fixed gap; tuned on one live take only

    def step(self, t, closed, away):
        self.closed0 = (t if self.closed0 is None else self.closed0) if closed else None
        # short "closed" gaps (<= BRIDGE_S) do not end a look-away episode: looking down lowers the lids, which reads as closed for a
        # few frames (found on a live take: the flicker restarted the distraction timer). A long closure is sleep, not distraction.
        if away:
            self.last_away = t
        bridge = closed and self.away0 is not None and t - self.last_away <= self.BRIDGE_S
        self.away0 = (t if self.away0 is None else self.away0) if away or bridge else None
        if self.closed0 is None:
            self.fired -= {"microsleep", "sleep"}
        if self.away0 is None:
            self.fired.discard("long_distraction")
        start = {"microsleep": self.closed0, "sleep": self.closed0, "long_distraction": self.away0}
        out = [k for k, s0 in start.items() if s0 is not None and t - s0 >= self.thr[k] and k not in self.fired]
        self.fired.update(out)
        return out


def detect(samples, **thr):
    """samples: iterable of (t_seconds, eyes_closed, gaze_off_road) -> list of (t_warn, kind); one warning per episode."""
    m = Monitor(**thr)
    return [(t, k) for t, c, a in samples for k in m.step(t, c, a)]


def timing_error(t_warn, t_event_start, required_s):
    """Seconds the warning is late (+) or early (-) vs. event start + required duration."""
    return t_warn - (t_event_start + required_s)


def within_tol(err, tol=TOL_S):
    return abs(err) <= tol


if __name__ == "__main__":
    hz = 25
    # eyes closed 1.0-6.0 s, gaze off-road 8.0-12.0 s
    s = [(i / hz, 1.0 <= i / hz < 6.0, 8.0 <= i / hz < 12.0) for i in range(15 * hz)]
    w = dict((k, t) for t, k in detect(s))
    assert set(w) == {"microsleep", "sleep", "long_distraction"}, w
    assert within_tol(timing_error(w["microsleep"], 1.0, MICROSLEEP_S))
    assert within_tol(timing_error(w["sleep"], 1.0, SLEEP_S))
    assert within_tol(timing_error(w["long_distraction"], 8.0, LONG_DISTRACTION_S))
    # a 1.0 s blink must not warn
    assert not detect([(i / hz, 0.5 <= i / hz < 1.5, False) for i in range(5 * hz)])
    # unknown eyes (None) never count as closed: 5 s of None -> no warning; None in the middle of a closure restarts the timer
    assert not detect([(i / hz, None, False) for i in range(5 * hz)])
    assert not detect([(i / hz, None if 1.0 <= i / hz < 1.2 else 0.0 <= i / hz < 2.0, False) for i in range(2 * hz)])
    # looking down: away with short "closed" flickers -> one distraction episode, warning 3 s after the first away frame
    flick = lambda x: 9.0 <= x < 9.2 or 10.0 <= x < 10.1
    w = detect([(i / hz, flick(i / hz), 8.0 <= i / hz < 12.0 and not flick(i / hz)) for i in range(15 * hz)])
    assert [k for _, k in w] == ["long_distraction"] and within_tol(timing_error(w[0][0], 8.0, LONG_DISTRACTION_S)), w
    # but closed eyes alone never start a distraction episode, and one away frame before a long closure does not become one
    assert [k for _, k in detect([(i / hz, 1.0 <= i / hz < 6.0, False) for i in range(8 * hz)])] == ["microsleep", "sleep"]
    assert [k for _, k in detect([(i / hz, 1.04 <= i / hz < 6.0, i == 25) for i in range(8 * hz)])] == ["microsleep", "sleep"]
    print("dms_core OK", dict((k, t) for t, k in detect(s)))
