#!/usr/bin/env bash
# Re-aim stage A from 10 epochs to 8 WITHOUT throwing away the tokens already trained.
#
# Why it waits instead of acting now: train_run only writes best.pt/last.pt at an evaluation (riff.py:1826),
# and the first evaluation is at epoch 1.0. Until then there is nothing to --resume from, so killing the run
# early would restart it from zero. This waits for last.pt, then restarts the same run with --epochs 8
# --resume, which reloads the model AND the AdamW state from last.pt and continues at the same token count.
#
# The point of the restart (rather than just stopping at epoch 8) is the learning rate: lr_at() takes
# seen/total, so a run told "10 epochs" is still well above the floor at epoch 8. Re-aiming at 8 makes the
# cosine decay land on its 10% floor exactly at the end, which is what the schedule is for.
set -u
cd /home/kapost/riff_gen || exit 1
PY=.venv/bin/python
CK=runs/final10/last.pt
LOG=runs/final10.log

echo "[$(date '+%F %T')] waiting for $CK (first evaluation, epoch 1.0)"
while [ ! -f "$CK" ]; do
    if ! ps -eo args | grep -q '[t]rain_n_test.py'; then
        echo "[$(date '+%F %T')] pretrain is gone and no $CK - nothing to switch, leaving it alone"
        exit 1
    fi
    sleep 60
done
# the checkpoint is written by save_ckpt as tmp+rename, so its presence means it is complete
sleep 20
echo "[$(date '+%F %T')] $CK is there; stopping the 10-epoch run"

# Wait on the SPECIFIC pid, never on a name pattern: a pattern match also matches the replacement process
# this script is about to start, so the "is it gone yet?" loop would succeed instantly and leave the old
# trainer running. That happened on 2026-11-13 and put two runs on the GPU at once (58k -> 22k tok/s).
PID=$(ps -eo pid,args | grep '[t]rain_n_test.py' | awk '{print $1}' | head -1)
if [ -n "${PID:-}" ]; then
    kill "$PID" 2>/dev/null
    for _ in $(seq 1 60); do ps -p "$PID" >/dev/null 2>&1 || break; sleep 2; done
    if ps -p "$PID" >/dev/null 2>&1; then
        kill -9 "$PID" 2>/dev/null
        for _ in $(seq 1 30); do ps -p "$PID" >/dev/null 2>&1 || break; sleep 2; done
    fi
    if ps -p "$PID" >/dev/null 2>&1; then
        echo "[$(date '+%F %T')] could not stop $PID - NOT relaunching, two runs would share the GPU"
        exit 1
    fi
fi
sleep 10
# and the GPU must actually be free of it before the replacement starts
for _ in $(seq 1 30); do
    nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -qw "$PID" || break
    sleep 2
done

echo "[$(date '+%F %T')] relaunching at --epochs 8 --resume" | tee -a "$LOG"
setsid nohup $PY train_n_test.py --epochs 8 --no-prepare --run runs/final10 --resume \
    >> "$LOG" 2>&1 < /dev/null &
sleep 30
if ps -eo args | grep -q '[t]rain_n_test.py'; then
    echo "[$(date '+%F %T')] 8-epoch run is up"
else
    echo "[$(date '+%F %T')] RELAUNCH FAILED - see $LOG"
    exit 1
fi
# status_a.sh exited when the old process died; bring it back for the new total
ps -eo args | grep -q '[s]tatus_a.sh' || setsid nohup bash runs/status_a.sh >> runs/logs/status_a.log 2>&1 < /dev/null &
echo "[$(date '+%F %T')] done"
