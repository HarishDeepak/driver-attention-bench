"""Spoken-guide cue log + run_clip's <run>_t0.txt -> labels.csv (clip,kind,t_start,t_end), without looking at any detection.
A cue's label lasts until the next cue. Truth starts at the cue, so it is early by the person's reaction time (~0.3-1 s):
use it to check WHICH actions were detected and false alerts, not the ±0.2 s timing.
Usage: python scripts/cues_to_labels.py <cues.csv> <run>_t0.txt <clip-name> > labels.csv"""
import csv
import sys
from datetime import datetime

KIND = {"close your eyes": "closed", "look down": "away", "turn your head": "away"}


def labels(cues, t0, clip):
    t = [(datetime.fromisoformat(c["wall_time"]).timestamp() - t0, c["cue"].lower()) for c in cues]
    return [(clip, k, round(a, 3), round(b, 3)) for (a, cue), (b, _) in zip(t, t[1:]) for key, k in KIND.items() if cue.startswith(key)]


if __name__ == "__main__":
    if len(sys.argv) == 1:  # self-check
        base = datetime.fromisoformat("2026-10-20T21:00:00+02:00").timestamp()
        cues = [dict(wall_time=f"2026-10-20T21:00:{s:02d}+02:00", cue=c) for s, c in
                [(5, "Get ready."), (12, "Close your eyes now."), (17, "Open your eyes."), (23, "Look down at your lap."), (28, "Look ahead.")]]
        assert labels(cues, base + 2, "c") == [("c", "closed", 10.0, 15.0), ("c", "away", 21.0, 26.0)], labels(cues, base + 2, "c")
        print("cues_to_labels OK")
        sys.exit()
    cues = list(csv.DictReader(open(sys.argv[1], newline="")))
    w = csv.writer(sys.stdout, lineterminator="\n")
    w.writerow(["clip", "kind", "t_start", "t_end"])
    w.writerows(labels(cues, float(open(sys.argv[2]).read()), sys.argv[3]))
