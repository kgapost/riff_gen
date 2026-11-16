#!/usr/bin/env bash
# Stage B2: genre fine-tunes as LoRA adapters, replacing the full fine-tunes of the first stage B.
#
#   bash runs/run_stage_b2.sh <schedule.json> <run-prefix> [genre ...]
#
# With no genre names it runs every genre in the schedule, in descending budget order so the long runs
# start first and the tail is cheap. With names it runs only those (that is how the pilot is run).
#
# Differences from runs/run_stages_bc.sh, all of them consequences of the stage B post-mortem:
#   --lora R    base weights frozen, so the learning rate cannot undo pretraining (riff.py:1719 lr_at
#               warms UP to the peak, which is what wrecked stage B at lr 1e-4 after stage A had
#               annealed to 3.74e-05). R comes from the schedule: 32 for the large genres.
#   --lr        3.74e-4, the pretraining peak, safe here for the same reason it was safe in stage C.
#   budget      from the schedule's multiplier, ~2-3x the first stage B.
#   eval_every  16 evaluations per run instead of 8, because train_run only writes last.pt at an
#               evaluation and those checkpoints are what label_best_watcher.py selects among.
#
# Selection: best.pt is chosen by riff.py on the WHOLE validation split, which for a genre run is ~96%
# other genres - genre_acoustic improved globally (0.2012 -> 0.1976) while getting worse on acoustic
# itself (0.2501 -> 0.2598). So runs/label_best_watcher.py must be running alongside this; it writes
# label_best.pt per run. This script starts it if it is not already up.
#
# Restartable (a run whose best.pt exists is skipped) and fault-tolerant (a failed run is recorded and
# the loop continues). Nothing here edits riff.py or train_n_test.py.
set -u
cd /home/kapost/riff_gen || exit 1

PY=.venv/bin/python
SCHED=${1:?usage: run_stage_b2.sh <schedule.json> <run-prefix> [genre ...]}
PREFIX=${2:?usage: run_stage_b2.sh <schedule.json> <run-prefix> [genre ...]}
shift 2
INIT=runs/final10/best.pt
STATUS=runs/status.txt
CSV=runs/${PREFIX}_summary.csv
mkdir -p runs/logs

[ -f "$INIT" ] || { echo "no pretrained checkpoint at $INIT"; exit 1; }
[ -f "$SCHED" ] || { echo "no schedule at $SCHED - run runs/schedule_b2.py first"; exit 1; }

# the schedule path has to reach the helpers too, so export it rather than hardcoding runs/schedule.json
export SCHED
field() { $PY -c "import json,os;print(json.load(open(os.environ['SCHED']))['genres']['$1']['$2'])"; }

if [ "$#" -gt 0 ]; then
    GENRES="$*"
else
    GENRES=$($PY -c "
import json, os
g = json.load(open(os.environ['SCHED']))['genres']
print(' '.join(sorted(g, key=lambda k: -g[k]['budget'])))")
fi
TOTAL=$(echo "$GENRES" | wc -w)
TOTBUD=$($PY -c "
import json, os, sys
g = json.load(open(os.environ['SCHED']))['genres']
print(sum(g[k]['budget'] for k in sys.argv[1:] if k in g))" $GENRES)
DONE=0
START=$(date +%s)
SEEN_BUD=0

# The watcher keeps the checkpoint that is best on the tuned genre's OWN val songs. It reads
# runs/status.txt to know which run is active, so it must start after this script has written one.
status() {
    local now el
    now=$(date +%s); el=$((now - START))
    { echo "stage      B2 genre"
      echo "run        $1"
      echo "rundir     ${CURRUN:-}"
      echo "tuning     ${CURLBL:-}"
      echo "elapsed    $((el / 3600))h$(( (el % 3600) / 60 ))m"
      echo "done       $DONE/$TOTAL runs"
      echo "tokens     $((SEEN_BUD / 1000000))M/$((TOTBUD / 1000000))M planned"
      echo "$2"
      echo "updated    $(date '+%F %T')"
    } > "$STATUS"
}

label_loss() {
    [ -f "$1" ] || { echo ""; return; }
    $PY runs/eval_label.py "$1" "$2" 2>/dev/null | $PY runs/jloss.py
}

status "0/$TOTAL starting" "schedule   $SCHED"
# never match a bare name that this script's own command line also contains; check the script name only
if ! ps -eo args | grep -q '[l]abel_best_watcher.py'; then
    echo "starting label_best_watcher.py"
    setsid nohup $PY runs/label_best_watcher.py >> runs/logs/label_best_watcher.log 2>&1 < /dev/null &
    sleep 5
fi

[ -f "$CSV" ] || echo "genre,n_train_songs,T_g,lora,W,budget,passes,base_val_loss,tuned_val_loss,label_best_loss,minutes,status" > "$CSV"

for g in $GENRES; do
    RUN=runs/${PREFIX}_$g
    LOG=runs/logs/${PREFIX}_$g.log
    if [ -f "$RUN/best.pt" ]; then
        echo "skip $g (best.pt exists)"; DONE=$((DONE + 1)); continue
    fi
    W=$(field "$g" W);        BUDGET=$(field "$g" budget); E=$(field "$g" eval_every)
    R=$(field "$g" lora);     N=$(field "$g" n_train_songs); TG=$(field "$g" T_g)
    P=$(field "$g" passes);   LR=$(field "$g" lr)
    CURRUN=$RUN; CURLBL="genre:$g"
    status "$((DONE + 1))/$TOTAL $g" "config     lora $R, lr $LR, budget $((BUDGET / 1000000))M, ${P} passes, W=$W"
    echo "=== $g: lora=$R lr=$LR W=$W budget=$BUDGET passes=$P eval_every=$E"
    T0=$(date +%s)
    BASE=$(label_loss "$INIT" "genre:$g")
    # --ckpt and --micro-tokens match stage A: riff.py train defaults --ckpt OFF, and without it a
    # 16384-token micro-batch stores every activation and OOMs this card. Same arithmetic either way.
    $PY riff.py train --model baseline --init "$INIT" --run "$RUN" --lora "$R" \
        --focus "genre:$g=$W,src:early=0.2" --budget "$BUDGET" --lr "$LR" \
        --batch-tokens 16384 --micro-tokens 16384 --ckpt --eval-every "$E" > "$LOG" 2>&1
    RC=$?
    MIN=$(( ($(date +%s) - T0) / 60 ))
    TUNED=$(label_loss "$RUN/best.pt" "genre:$g")
    LBEST=$(label_loss "$RUN/label_best.pt" "genre:$g")
    ST=ok; [ $RC -eq 0 ] || ST="failed_rc$RC"
    echo "$g,$N,$TG,$R,$W,$BUDGET,$P,$BASE,$TUNED,$LBEST,$MIN,$ST" >> "$CSV"
    DONE=$((DONE + 1)); SEEN_BUD=$((SEEN_BUD + BUDGET))
    status "$DONE/$TOTAL $g" "last       base=$BASE best=$TUNED label_best=$LBEST ($ST, ${MIN}m)"
done

# clear these so the watcher stops pointing at the last run's checkpoint once the stage is over
CURRUN=""; CURLBL=""
status "done $DONE/$TOTAL" "finished   $(date '+%F %T')"
echo "=== stage B2 finished: $CSV"
