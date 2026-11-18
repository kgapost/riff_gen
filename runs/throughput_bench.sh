#!/usr/bin/env bash
# Measured tokens/second for the Stage A candidates. VRAM says what FITS; this says what it COSTS.
#
# runs/vram_probe.py established that all four widths fit on the 24GiB card, that "wide" (d=896, 157M)
# fits with activation checkpointing OFF at 16.9GiB while "wider" (203M) does not, and that ctx 8192 is
# almost free in memory. What none of that says is wall-clock, which is what decides whether a Stage A
# rerun is 17 hours or 40. So each configuration trains a few million tokens and the steady-state rate is
# read off the log, skipping the first lines while the allocator and (where used) the compiler warm up.
#
# One at a time: two runs sharing this GPU halved throughput on 2026-11-13 and the measurement would be
# meaningless anyway.
set -u
cd /home/kapost/riff_gen || exit 1
PY=.venv/bin/python
OUT=runs/bench
mkdir -p "$OUT"
BUDGET=${BUDGET:-6e6}

# name|model|extra flags
RUNS=(
  "a_today|baseline|--micro-tokens 16384 --ckpt"
  "b_today_compile|baseline|--micro-tokens 16384 --ckpt --compile"
  "c_today_micro32k|baseline|--micro-tokens 32768 --ckpt"
  "d_wide_ckpt|wide|--micro-tokens 16384 --ckpt"
  "e_wide_nockpt|wide|--micro-tokens 16384"
  "f_wide_nockpt_compile|wide|--micro-tokens 16384 --compile"
  "g_wider_ckpt32k|wider|--micro-tokens 32768 --ckpt"
)

for entry in "${RUNS[@]}"; do
    IFS='|' read -r tag model extra <<< "$entry"
    echo "[$(date '+%F %T')] $tag  ($model $extra)"
    rm -rf "/tmp/bench_$tag"
    # shellcheck disable=SC2086
    $PY -u riff.py train --model "$model" $extra --budget "$BUDGET" --batch-tokens 16384 \
        --run "/tmp/bench_$tag" > "$OUT/$tag.log" 2>&1 || echo "   FAILED - see $OUT/$tag.log"
    # steady state: drop the first three progress lines, average the rest
    $PY - "$OUT/$tag.log" <<'EOF'
import re, sys, statistics as st
rates = [float(m) for m in re.findall(r"(\d+)k tok/s", open(sys.argv[1]).read())]
if len(rates) > 4:
    s = rates[3:]
    print(f"   {st.median(s):.0f}k tok/s median over {len(s)} samples (peak {max(s):.0f}k)")
else:
    print(f"   too few samples ({len(rates)}) - see the log")
EOF
done
echo "[$(date '+%F %T')] done -> $OUT"
