#!/usr/bin/env bash
# Calibrate --cfg over eight songs, so "use 7" becomes a number worth trusting.
#
# The four-song sweep established two things firmly - the 1.5 default is inert, and 10+ degenerates - but
# the effect size inside the band ranged from 0.108 to 1.192 at cfg 7 on three candidates per cell, which
# is too noisy to pick a default from. This runs eight contexts (two of the user's own projects and six
# genres from doom to classical) at eight candidates, and runs/cfg_aggregate.py averages them.
#
# One song at a time on purpose: each riff.py run loads the checkpoint and holds the GPU, and two at once
# would halve throughput the way the duplicate trainers did on 2026-11-13.
set -u
cd /home/kapost/riff_gen || exit 1
PY=.venv/bin/python
OUT=runs/cfg_cal
mkdir -p "$OUT"

# song|gap - the gap sits mid-song, 8 bars, clear of the intro
SONGS=(
  "tabs - own - cs - own/sentimental sel.gp5|25-32"
  "tabs - own - vc2 - own/old AKC.gp5|49-56"
  "tabs - doom-stoner-sludge/Clutch - A Shogun Named Marcus__v2.gp5|49-56"
  "tabs - classical/Agustin Barrios Mangore - Julia Florida Barcarola.gp5|25-32"
  "tabs - jazz/Antonio Carlos Jobim - Quiet Night Of Quiet Stars.gp5|25-32"
  "tabs - metal xtreme/Advent Sorrow - Skin To Suffer In.gp5|41-48"
  "tabs - rock/(HED)P.E. - I Got You.gp5|33-40"
  "tabs - acoustic/Bert Jansch - Alice's Wonderland (2).gp5|25-32"
)

i=0
for entry in "${SONGS[@]}"; do
    f="${entry%|*}"; g="${entry##*|}"
    i=$((i + 1))
    tag=$(printf '%02d' "$i")
    echo "[$(date '+%F %T')] $i/${#SONGS[@]}  $f  gap $g"
    $PY -u runs/cfg_control.py --song "data/$f" --gap "$g" --cands 8 \
        > "$OUT/$tag.txt" 2>&1 || echo "  FAILED (see $OUT/$tag.txt)"
done
echo "[$(date '+%F %T')] done -> $OUT"
$PY runs/cfg_aggregate.py "$OUT"
