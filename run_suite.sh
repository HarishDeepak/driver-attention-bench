#!/usr/bin/env bash
# Run the whole suite inside WSL Ubuntu: bash run_suite.sh [scenarios/<name>.toml] (from the repo root; default suite.toml)
set -euo pipefail
CFG="${1:-scenarios/suite.toml}"
PY="${PY:-$HOME/dmsvenv/bin/python}"
cfg() { "$PY" -c "import tomllib; c = tomllib.load(open('$CFG', 'rb')); $1"; }
OUT=$(cfg "print(c['out'])")
LABELS=$(cfg "print(c['labels'])")
CALIB=$(cfg "print(c.get('calib_s', 2))")  # optional: calibration window in s
SCOPE=$(cfg "print(c.get('scope', ''))")   # optional: clip,kind,t_start,t_end where ground truth exists
mkdir -p "$OUT/runs"
# one line per job: <video> <variant> <--clahe or ->
JOBS=$(cfg "
import glob, os
vids = sorted(p for e in ('mp4', 'mov', 'avi', 'mkv') for p in glob.glob(os.path.join(c['clips_dir'], '*.' + e)))
for v in vids:
    for var in c['variants']:
        for cl in c['clahe']:
            print(v, var, '--clahe' if cl else '-')")
[ -n "$JOBS" ] || { echo "no clips found in clips_dir of $CFG" >&2; exit 1; }
while read -r video variant clahe; do
  echo "== $video $variant ${clahe#-}"
  if [ "$clahe" = "-" ]; then
    "$PY" scripts/run_clip.py "$video" --variant "$variant" --calib-s "$CALIB" --out "$OUT/runs"
  else
    "$PY" scripts/run_clip.py "$video" --variant "$variant" --clahe --calib-s "$CALIB" --out "$OUT/runs"
  fi
done <<< "$JOBS"
"$PY" src/evaluate.py --labels "$LABELS" --runs "$OUT/runs" --out "$OUT" ${SCOPE:+--scope "$SCOPE"}
"$PY" src/report.py --out "$OUT" --labels "$LABELS"
echo "Report: $PWD/$OUT/report.pdf"
