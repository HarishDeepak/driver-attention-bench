"""Drowsiness level from repeated eye closures (E12). Pure Python, sits beside dms_core.Monitor and does not change its SD-202 timers.

Euro NCAP Driver Engagement (2026) scores drowsiness (KSS > 7 at the latest) with a warning plus more sensitive Forward / Lane Support;
EU DDAW (Reg. 2021/1341) warns at KSS >= 8. Neither asks the car to slow down for drowsiness: that only happens for an unresponsive
driver (escalation.py). Levels, KSS-like:
  0 alert      nothing
  1 tiring     KSS ~7: rest advice ("coffee cup")
  2 drowsy     KSS >= 8: urgent drowsiness warning + sensitive ADAS (earlier collision warning, longer ACC gap)
  3 severe     repeated microsleeps despite the level-2 warning: degraded mode (speed cap, escalation.py) until the driver rests
All thresholds are our assumptions (marked below), not product values."""
from collections import deque

WIN_S = 300.0  # microsleeps counted over the last 5 min
BLINK_WIN_S = 60.0  # long blinks counted over the last minute
PERCLOS_WARMUP_S = 60.0  # = Classifier.perclos_s
LONG_BLINK_S = 0.5  # normal blinks last ~0.1-0.4 s; slower, longer blinks are an early drowsiness sign
# (level, PERCLOS, microsleeps in WIN_S, long blinks in BLINK_WIN_S); a level is reached when any one of its cues is met.
# ponytail: PERCLOS 0.15 is the common P80 "drowsy" cut-off; the rest are guesses to be tuned on progressive data (UL-DD / DROZY)
RULES = [(3, 0.30, 3, None), (2, 0.15, 2, None), (1, 0.08, 1, 6)]


class Drowsiness:
    """step(t, closed, warned, perclos, away) -> level 0-3. closed, away: Classifier output (closed True/False/None); warned: kinds
    fired by Monitor.step this frame; perclos: Classifier.perclos (None until measurable). The level follows the windows, so it can fall."""

    def __init__(self):
        self.ms, self.blinks, self.c0, self.level, self.t0 = deque(), deque(), None, 0, None

    def step(self, t, closed, warned, perclos, away=False):
        self.t0 = t if self.t0 is None else self.t0
        # closures while the head is out of the forward cone are not counted: looking down lowers the lids. On DMD every microsleep
        # warning in the alert gaze sessions (s6) fired with the head away (4/4 blend, 24/24 EAR), in the drowsy sessions 1/26 and 2/40.
        # ponytail: also drops a nodding-off closure; the SD-202 sleep warning and the unresponsive-driver MRM still catch that one
        closed = closed and not away
        if "microsleep" in warned and not away:
            self.ms.append(t)
        if closed and self.c0 is None:
            self.c0 = t
        elif not closed and self.c0 is not None:  # a closure ended (None = eyes unknown also ends it, as in Monitor)
            if t - self.c0 >= LONG_BLINK_S:
                self.blinks.append(t)
            self.c0 = None
        for q, w in ((self.ms, WIN_S), (self.blinks, BLINK_WIN_S)):
            while q and q[0] < t - w:
                q.popleft()
        # PERCLOS over the first seconds after calibration is one blink over a short window (NITYMED: 0.6 at 21 s): use it only
        # once its 60 s window is full
        p = (perclos or 0.0) if t - self.t0 >= PERCLOS_WARMUP_S else 0.0
        self.level = next((lv for lv, pc, n_ms, n_bl in RULES
                           if p >= pc or len(self.ms) >= n_ms or (n_bl is not None and len(self.blinks) >= n_bl)), 0)
        return self.level


if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from dms_core import Monitor

    def run(closures, total, hz=25, perclos=None):
        """closures: [(start, dur)] -> list of (t, level) changes"""
        m, d, out = Monitor(), Drowsiness(), []
        for i in range(int(total * hz)):
            t = i / hz
            c = any(a <= t < a + b for a, b in closures)
            lv = d.step(t, c, m.step(t, c, False), perclos)
            if not out or out[-1][1] != lv:
                out.append((round(t, 2), lv))
        return out

    assert run([(10 + 4 * k, 0.2) for k in range(20)], 100) == [(0.0, 0)]  # normal blinks: alert
    assert [lv for _, lv in run([(10 + 8 * k, 0.6) for k in range(8)], 80)] == [0, 1]  # 6 long blinks in a minute: tiring
    # one microsleep -> 1; a second within 5 min -> 2; a third -> 3 (two ignored drowsiness warnings)
    lv = run([(10, 1.6), (60, 1.6), (110, 1.6)], 130)
    assert [x for _, x in lv] == [0, 1, 2, 3] and abs(lv[1][0] - 11.5) < 0.1 and abs(lv[3][0] - 111.5) < 0.1, lv
    # microsleeps 6 min apart never pile up: the window lets the level fall back
    assert [x for _, x in run([(10, 1.6), (400, 1.6)], 420)] == [0, 1, 0, 1]
    d, m = Drowsiness(), Monitor()  # three microsleeps while looking down (head away) count for nothing
    assert max(d.step(i / 25, c, m.step(i / 25, c, True), None, away=True) for i in range(4000) for c in [i % 1000 < 50]) == 0
    assert run([], 70, perclos=0.2) == [(0.0, 0), (60.0, 2)]  # PERCLOS alone can raise it, once its window is full
    print("drowsiness OK", lv)
