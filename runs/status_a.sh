#!/usr/bin/env bash
# Keeps runs/status.txt current while stage A runs, so the hourly report is one file read instead of a log
# scan. Exits by itself when the pretrain process is gone (stage B/C driver then owns status.txt).
set -u
cd /home/kapost/riff_gen || exit 1
LOG=runs/final10.log

# epochs x epoch_tokens, read from the run's own run_info.json so this stays right when the target epoch
# count changes (10 -> 8) instead of hardcoding the total.
read_total() {
    .venv/bin/python -c "
import json
d = json.load(open('runs/final10/run_info.json'))
print(f\"{(d['budget'] or d['epochs'] * d['epoch_tokens']) / 1e6:.0f} {d['epochs']:.0f}\")" 2>/dev/null || echo "4880 10"
}

while ps -eo args | grep -q '[t]rain_n_test.py'; do
    set -- $(read_total)
    TOTAL=${1:-4880}
    EPOCHS=${2:-10}
    LAST=$(grep -oE 'step [0-9]+ ep [0-9.]+ loss [0-9.]+ acc [0-9.]+ lr [0-9.e-]+ [0-9]+k tok/s' "$LOG" | tail -1)
    EP=$(echo "$LAST" | sed -nE 's/.*ep ([0-9.]+).*/\1/p')
    TPS=$(echo "$LAST" | sed -nE 's/.*[^0-9]([0-9]+)k tok\/s.*/\1/p')
    SEEN=$(echo "$EP" | awk '{printf "%.0f", $1 * 488}')
    LEFT=""
    if [ -n "${TPS:-}" ] && [ "${TPS:-0}" -gt 0 ] && [ -n "${SEEN:-}" ]; then
        # t,s are in MILLIONS of tokens, r in thousands of tokens/s
        LEFT=$(awk -v t=$TOTAL -v s=$SEEN -v r=$TPS 'BEGIN{h=(t-s)*1000.0/r/3600.0;printf "%.1fh", h}')
    fi
    VAL=$(grep -iE '^\s*(best val|last val|  val)|val loss' "$LOG" | tail -1 | cut -c1-110)
    { echo "stage      A pretrain (${EPOCHS} epochs)"
      echo "run        1/1 final10"
      echo "progress   epoch ${EP:-?}/${EPOCHS}  (~${SEEN:-?}M of ${TOTAL}M tokens)"
      echo "train      $(echo "$LAST" | grep -oE 'loss [0-9.]+ acc [0-9.]+')  at ${TPS:-?}k tok/s"
      echo "val        ${VAL:-none yet}"
      echo "eta        stage A ~${LEFT:-?} left"
      echo "updated    $(date '+%F %T')"
    } > runs/status.txt
    sleep 300
done
