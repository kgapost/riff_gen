#!/usr/bin/env bash
# Re-queue the remaining stage B2 genres by base validation loss, highest first.
#
# Why: run_stage_b2.sh defaults to descending BUDGET order, which turns out to be close to the opposite of
# descending usefulness. The first four runs (metal_xtreme, rock, prog, punk - the four largest genres) all
# came back flat, every delta inside +-0.0007 against a 0.0036 noise floor, because the base model already
# fits those genres well. The gains live where the base model FAILS: classical (base 0.3677) improved by
# 0.018 in the pilot, and shred (0.5601), blues_soul_rnb (0.3612) and jazz (0.3452) are still unqueued.
# Ordering by base loss puts every promising run in the next few hours instead of on Wednesday evening, so
# the question "is label fine-tuning worth doing at all" gets answered while there is still time to act.
#
# Base losses come from runs/genre_summary.csv (the first stage B measured the pretrained checkpoint on
# every genre, which is exactly the number wanted here) so nothing has to be recomputed on the GPU.
#
# Nothing already finished is lost: the driver skips any genre whose best.pt exists. This waits for the
# ACTIVE run to finish first, so no partial work is thrown away either.
set -u
cd /home/kapost/riff_gen || exit 1
PY=.venv/bin/python
SCHED=runs/schedule_b2.json

# ---------------------------------------------------------------- wait for the active run to complete
AG=$(grep -oP '^tuning\s+genre:\K\S+' runs/status.txt 2>/dev/null)
# Completion is a ROW IN THE SUMMARY CSV, not the existence of best.pt: train_run writes best.pt at every
# evaluation that improves, so best.pt appears a few minutes into a run and would make the wait below fall
# straight through and kill the active trainer half-trained. The driver appends to the csv only after the
# run has returned, so that is the only honest "finished" signal. (The driver's own skip test does use
# best.pt, which is why the active genre must be allowed to finish rather than re-queued: it would be
# skipped on the next pass and left partially trained.)
done_row() { [ -f runs/b2_summary.csv ] && cut -d, -f1 runs/b2_summary.csv | grep -qx "$1"; }
if [ -n "${AG:-}" ] && ! done_row "$AG"; then
    echo "[$(date '+%F %T')] waiting for the active run ($AG) to finish before re-queueing"
    while ! done_row "$AG"; do
        ps -eo args | grep -q '[r]un_stage_b2.sh' || { echo "driver already gone"; break; }
        sleep 60
    done
    sleep 20
fi

# ---------------------------------------------------------------- stop the driver, then the trainer
# Wait on specific PIDs, never on a name pattern: a pattern also matches the replacement this script is
# about to launch, so an "is it gone yet?" loop would succeed instantly and leave two runs on the GPU.
# That happened on 2026-11-13 and halved throughput (58k -> 22k tok/s) for an hour and three quarters.
DPID=$(ps -eo pid,args | grep '[r]un_stage_b2.sh' | awk '{print $1}' | head -1)
[ -n "${DPID:-}" ] && { echo "[$(date '+%F %T')] stopping driver $DPID"; kill "$DPID" 2>/dev/null; }
TPID=$(ps -eo pid,args | grep '[r]iff.py train' | awk '{print $1}' | head -1)
for P in ${DPID:-} ${TPID:-}; do
    kill "$P" 2>/dev/null
    for _ in $(seq 1 60); do ps -p "$P" >/dev/null 2>&1 || break; sleep 2; done
    if ps -p "$P" >/dev/null 2>&1; then
        kill -9 "$P" 2>/dev/null
        for _ in $(seq 1 30); do ps -p "$P" >/dev/null 2>&1 || break; sleep 2; done
    fi
    if ps -p "$P" >/dev/null 2>&1; then
        echo "[$(date '+%F %T')] could not stop $P - NOT relaunching, two runs would share the GPU"
        exit 1
    fi
done
# The stopped trainer must have released the card before the replacement starts - but test for THOSE pids,
# not for an empty compute-app list: label_best_watcher.py also holds ~740MB on the GPU for its whole life,
# so "no compute apps at all" never becomes true and the wait would just burn its full timeout.
for _ in $(seq 1 30); do
    GONE=1
    for P in ${DPID:-} ${TPID:-}; do
        nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -qw "$P" && GONE=0
    done
    [ "$GONE" = 1 ] && break
    sleep 2
done
sleep 10

# ---------------------------------------------------------------- the new order
ORDER=$($PY -c "
import csv, json, os
base = {}
with open('runs/genre_summary.csv') as fh:
    for r in csv.DictReader(fh):
        if r['base_val_loss']:
            base[r['genre']] = float(r['base_val_loss'])
todo = [g for g in json.load(open('$SCHED'))['genres']
        if not os.path.exists(f'runs/b2_{g}/best.pt')]
# unmeasured genres last rather than first: an unknown is not evidence of headroom
todo.sort(key=lambda g: -base.get(g, -1.0))
print(' '.join(todo))")
echo "[$(date '+%F %T')] re-queued order: $ORDER"
[ -n "$ORDER" ] || { echo "nothing left to run"; exit 0; }

setsid nohup bash runs/run_stage_b2.sh "$SCHED" b2 $ORDER >> runs/logs/b2_driver.log 2>&1 < /dev/null &
sleep 30
if ps -eo args | grep -q '[r]un_stage_b2.sh'; then
    echo "[$(date '+%F %T')] re-queued driver is up"
else
    echo "[$(date '+%F %T')] RELAUNCH FAILED - see runs/logs/b2_driver.log"
    exit 1
fi
