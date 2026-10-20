#!/usr/bin/env bash
# Render the 6 synthetic clips (3 Blender processes in parallel) and merge labels -> data/synthetic/labels.csv. Run from anywhere.
set -euo pipefail
B="${BLENDER:-/c/blender/blender-4.5.9-windows-x64/blender.exe}"
cd "$(dirname "$0")/../.."
mkdir -p data/synthetic
run() { "$B" -b -P tools/blender/make_driver.py -- --clip "$1" --seed "$2" --acc "$3" 2>&1 | grep -E "^HUMAN|^DONE|Error|Traceback"; }
{ run synth01_microsleep 1 none; run synth04_lizard 1 none; } &
{ run synth02_sleep 2 none; run synth05_sunglasses 2 sunglasses; } &
{ run synth03_owl 3 none; run synth06_mask 3 mask; } &
wait
{ echo clip,kind,t_start,t_end; cat data/synthetic/_labels/*.csv; } > data/synthetic/labels.csv
cat data/synthetic/labels.csv
