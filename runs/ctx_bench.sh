#!/usr/bin/env bash
# What does a longer window COST? Memory and throughput for ctx 4096 / 8192 / 16384.
#
# There is time for exactly one Stage A rerun, so the window has to be chosen on evidence. Memory is not
# the constraint - runs/vram_probe.py found ctx 8192 costs 0.01GiB more than 4096, because attention here
# is memory-linear - so the question is throughput and, separately, quality.
#
# What a longer window buys, measured on the corpus (fraction of songs that fit whole in 0.95*ctx):
#   4096   28.3 % of songs,  6.5 % of tokens
#   8192   53.8 % of songs, 24.5 % of tokens
#  16384   88.6 % of songs, 70.4 % of tokens
# The 8192 -> 16384 step is much larger than 4096 -> 8192, which is why it is worth measuring rather than
# assuming. Those numbers predate the chunking pass and will shift once prepare runs again.
#
# NOTE this measures cost, not quality. A window the model cannot yet exploit looks free in tok/s and
# still trains a worse model, so the quality probe is separate.
set -u
cd /home/kapost/riff_gen || exit 1
PY=.venv/bin/python
OUT=runs/bench_ctx
mkdir -p "$OUT"
BUDGET=${BUDGET:-6e6}
MODEL=${MODEL:-baseline}

for ctx in 4096 8192 16384; do
    for micro in 16384 32768; do
        tag="${MODEL}_ctx${ctx}_m${micro}"
        echo "[$(date '+%F %T')] $tag"
        rm -rf "/tmp/bench_$tag"
        $PY -u riff.py train --model "$MODEL" --ctx "$ctx" --micro-tokens "$micro" --ckpt \
            --budget "$BUDGET" --batch-tokens 32768 --run "/tmp/bench_$tag" \
            > "$OUT/$tag.log" 2>&1 || { echo "   FAILED"; grep -oE 'OutOfMemoryError|Error' "$OUT/$tag.log" | head -1; continue; }
        $PY - "$OUT/$tag.log" <<'EOF'
import re, sys, statistics as st
t = open(sys.argv[1]).read()
rates = [float(m) for m in re.findall(r"(\d+)k tok/s", t)]
mem = re.findall(r"([\d.]+)GiB", t)
if len(rates) > 4:
    s = rates[3:]
    print(f"   {st.median(s):.0f}k tok/s median over {len(s)} samples")
else:
    print(f"   too few samples ({len(rates)})")
EOF
    done
done
echo "[$(date '+%F %T')] done -> $OUT"
