#!/usr/bin/env bash
# Stage B (28 genre fine-tunes, full weights) then stage C (13 album fine-tunes, LoRA 16), sequentially.
#
# Reads the schedule computed by runs/label_tokens.py (runs/schedule.json): per run the focus weight W, the
# token budget and eval_every. Writes one log per run into runs/logs/, keeps runs/status.txt current, appends
# a row per run to runs/genre_summary.csv / runs/album_summary.csv.
#
# Restartable: a run whose best.pt already exists is skipped. A run that FAILS does not stop the stage - the
# row goes into the csv with the error and the driver moves on, as asked.
#
# Nothing here edits riff.py or train_n_test.py; it only calls them.
set -u
cd /home/kapost/riff_gen || exit 1

PY=.venv/bin/python
SCHED=runs/schedule.json
INIT=runs/final10/best.pt
STATUS=runs/status.txt
mkdir -p runs/logs

[ -f "$INIT" ] || { echo "no pretrained checkpoint at $INIT - stage A must finish first"; exit 1; }
[ -f "$SCHED" ] || { echo "no $SCHED - run runs/label_tokens.py first"; exit 1; }

START=$(date +%s)
GENRES=$($PY -c 'import json;print(" ".join(json.load(open("runs/schedule.json"))["genres"]))')
ALBUMS=$($PY -c 'import json;print(" ".join(json.load(open("runs/schedule.json"))["albums"]))')
NG=$(echo "$GENRES" | wc -w)
NA=$(echo "$ALBUMS" | wc -w)
TOTAL=$((NG + NA))
DONE=0

# field <kind> <name> <key> -> one value out of schedule.json
field() { $PY -c "import json,sys;print(json.load(open('runs/schedule.json'))['$1']['$2']['$3'])"; }

# status <stage> <run label> <extra>
status() {
    local now el
    now=$(date +%s); el=$((now - START))
    { echo "stage      $1"
      echo "run        $2"
      echo "elapsed    $((el / 3600))h$(( (el % 3600) / 60 ))m"
      echo "done       $DONE/$TOTAL runs"
      echo "$3"
      echo "updated    $(date '+%F %T')"
    } > "$STATUS"
}

# the last "loss ... acc ..." progress line of a run log, and the last validation line
last_loss() { grep -oE 'loss [0-9.]+ acc [0-9.]+' "$1" 2>/dev/null | tail -1; }
last_val()  { grep -iE 'val|eval' "$1" 2>/dev/null | tail -1 | cut -c1-120; }

# label_loss <ckpt> <label> -> "loss" or "" ; prints nothing on failure
label_loss() {
    [ -f "$1" ] || { echo ""; return; }
    $PY runs/eval_label.py "$1" "$2" 2>/dev/null | $PY runs/jloss.py
}

# ---------------------------------------------------------------- stage B: genres, full fine-tune, lr 1e-4
CSV=runs/genre_summary.csv
[ -f "$CSV" ] || echo "genre,n_train_songs,T_g,W,budget,base_val_loss,tuned_val_loss,minutes,status" > "$CSV"

for g in $GENRES; do
    RUN=runs/genre_$g
    LOG=runs/logs/genre_$g.log
    if [ -f "$RUN/best.pt" ]; then
        echo "skip genre $g (best.pt exists)"; DONE=$((DONE + 1)); continue
    fi
    W=$(field genres "$g" W)
    BUDGET=$(field genres "$g" budget)
    E=$(field genres "$g" eval_every)
    N=$(field genres "$g" n_train_songs)
    TG=$(field genres "$g" T_g)
    status "B genre" "$((DONE + 1))/$TOTAL $g" "budget    ${BUDGET} tokens, W=$W, eval_every=$E"
    echo "=== genre $g: W=$W budget=$BUDGET eval_every=$E"
    T0=$(date +%s)
    BASE=$(label_loss "$INIT" "genre:$g")
    # --ckpt and --micro-tokens match stage A exactly. riff.py train defaults --ckpt OFF, and without it a
    # 16384-token micro-batch stores every activation: the smoke test OOMed on this card at the default
    # batch. Same arithmetic either way, so this only changes memory.
    $PY riff.py train --model baseline --init "$INIT" --run "$RUN" \
        --focus "genre:$g=$W,src:early=0.2" --budget "$BUDGET" --lr 1e-4 \
        --batch-tokens 16384 --micro-tokens 16384 --ckpt --eval-every "$E" > "$LOG" 2>&1
    RC=$?
    MIN=$(( ($(date +%s) - T0) / 60 ))
    TUNED=$(label_loss "$RUN/best.pt" "genre:$g")
    ST=ok; [ $RC -eq 0 ] || ST="failed_rc$RC"
    echo "$g,$N,$TG,$W,$BUDGET,$BASE,$TUNED,$MIN,$ST" >> "$CSV"
    DONE=$((DONE + 1))
    status "B genre" "$((DONE))/$TOTAL $g" "last       base=$BASE tuned=$TUNED ($ST, ${MIN}m)"
done

# ---------------------------------------------------------------- stage C: albums, LoRA 16, lr 3.74e-4
CSV=runs/album_summary.csv
[ -f "$CSV" ] || echo "album,n_own,n_insp,T_a,W,budget,base_val_loss,tuned_val_loss,minutes,status" > "$CSV"

for a in $ALBUMS; do
    RUN=runs/album_$a
    LOG=runs/logs/album_$a.log
    if [ -f "$RUN/best.pt" ]; then
        echo "skip album $a (best.pt exists)"; DONE=$((DONE + 1)); continue
    fi
    W=$(field albums "$a" W)
    BUDGET=$(field albums "$a" budget)
    E=$(field albums "$a" eval_every)
    OWN=$(field albums "$a" n_own)
    INSP=$(field albums "$a" n_insp)
    TA=$(field albums "$a" T_a)
    # README section 13: replay from the album's genres, as --focus rows at weight 2
    REPLAY=$($PY -c "import json;g=json.load(open('runs/schedule.json'))['albums']['$a']['replay_genres'];print(''.join(f',genre:{x}=2' for x in g))")
    status "C album" "$((DONE + 1))/$TOTAL $a" "budget    ${BUDGET} tokens, W=$W, lora 16, replay${REPLAY:-' none'}"
    echo "=== album $a: W=$W budget=$BUDGET eval_every=$E replay=$REPLAY"
    T0=$(date +%s)
    BASE=$(label_loss "$INIT" "album:$a")
    $PY riff.py train --model baseline --init "$INIT" --run "$RUN" --lora 16 \
        --focus "album:$a=$W$REPLAY" --budget "$BUDGET" --lr 3.74e-4 \
        --batch-tokens 16384 --micro-tokens 16384 --ckpt --eval-every "$E" > "$LOG" 2>&1
    RC=$?
    MIN=$(( ($(date +%s) - T0) / 60 ))
    TUNED=$(label_loss "$RUN/best.pt" "album:$a")
    ST=ok; [ $RC -eq 0 ] || ST="failed_rc$RC"
    echo "$a,$OWN,$INSP,$TA,$W,$BUDGET,$BASE,$TUNED,$MIN,$ST" >> "$CSV"
    DONE=$((DONE + 1))
    status "C album" "$((DONE))/$TOTAL $a" "last       base=$BASE tuned=$TUNED ($ST, ${MIN}m)"
done

status "done" "$DONE/$TOTAL" "finished   $(date '+%F %T')"
echo "=== stages B and C finished"
