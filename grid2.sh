#!/usr/bin/env bash
# Second grid (Nov 7). One stage per call; grid2_all.sh runs them all in order.
#
#   usage: ./grid2.sh base-new | huge-new | base-focal | base-old | huge-old
#
# Three things the Nov 5 grid could not answer:
#
#   1. every winner sat at the lowest batch AND the lowest learning rate of its grid, so the optimum is
#      outside it: this one extends both axes downward (bm 0.125/0.25, lm 0.25/0.354/0.5).
#   2. label smoothing hurt both models that tried it, so the opposite is worth a look: --focal rescales
#      each token by (1-p)^gamma, spending the step on what the model still gets wrong.
#   3. the loss it ranked on is not comparable across architectures - the flat window carries the
#      neighbouring bars as plain tokens while the hierarchical models see one vector per bar. Every run
#      here also reports loss_cmp (the same loss over the 114 val gaps that nothing in the song repeats and
#      that the flat builder does not shorten, where both builders weight an identical set of target
#      tokens) and lift_cmp (nats per token below the corpus's own bigram, from riff.py ngram).
#
# big is dropped: all 36 huge runs of the Nov 5 grid beat big's best (0.9732), and no big run beat any huge
# run. The loss axis is fixed at what that grid settled - smooth 0 for baseline, aux 0.1 for huge - so this
# grid moves one thing at a time.
#
# Everything that decides the validation loss is pinned to the Nov 5 values, so the new rows and the 84 old
# ones belong in one table: the same corpus (data/encoded_grid, untouched since Oct 30 - re-preparing it
# would change the gaps and the fit and void the old rows), the same 410.2M-token budget, the same 400
# evaluation gaps, the same eval cadence and the same seed. --micro-tokens 16384 reproduces the micro-batch
# the baseline runs used (its model default has since changed to 32768); huge's default 49152 already
# matches. The *-old stages re-run the Nov 5 cells only to attach the two new metrics to them: that grid
# kept no weights (keep_ckpt=False), so training them again is the only way to put the configurations it
# tested on the new metrics. The first run of base-new is the Nov 5 winner, so its val_loss must come back
# at 0.1981; if it does not, these runs are not measuring what the old rows measured.
#
# --ckpt is on for baseline because it has to be: at micro 16384 it peaks at 22.5 GiB of the 4090's 23.5,
# which the Nov 5 runs just fit and this machine no longer does. Recomputing the activations instead of
# storing them is the same arithmetic on the same batch, packing and learning rate - it only costs time
# (73k -> 57k tok/s, 1.6 -> 2.0 h a run) - so unlike a smaller micro-batch it leaves the comparison with
# the Nov 5 rows intact.
#
# The stages run one at a time on purpose. Two streams at once do fit on the card (4.0 GiB for baseline
# against 9.3 for huge's heaviest cell, with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to stop the
# allocators hoarding), but measured together they reach ~51k tok/s against 57k for a single stream: the
# 4090 is already compute-bound, so sharing it loses time instead of saving it.
#
# Finished runs are skipped, so this can be stopped and started again at will.
set -u
cd "$(dirname "$0")"

COMMON="--data data/encoded_grid --out runs/grid2 --budget 410200000
        --eval-gaps 400 --eval-every 0.125 --seed 1234 --workers 4"
BASE="--models baseline --micro-tokens 16384 --ckpt"
HUGE="--models huge --aux 0.1"
NEW="--batches 0.25,0.125 --lrs 0.5,0.354,0.25"
OLD="--batches 0.25,0.5,1 --lrs 0.5,0.707,1,1.414"

echo "=== $(date '+%F %T')  stage ${1:-?}"
case "${1:-}" in
base-new)   python riff.py grid $COMMON $BASE --smooth 0 $NEW ;;          # 6 runs, ~12 h
huge-new)   python riff.py grid $COMMON $HUGE            $NEW ;;          # 6 runs, ~16 h
base-focal) python riff.py grid $COMMON $BASE --focal 1,2 --batches 0.25 --lrs 0.5 ;;   # 2 runs, ~4 h
base-old)   python riff.py grid $COMMON $BASE --smooth 0 $OLD ;;          # 11 new runs, ~22 h
huge-old)   python riff.py grid $COMMON $HUGE            $OLD ;;          # 11 new runs, ~24 h
*) echo "usage: $0 base-new | huge-new | base-focal | base-old | huge-old" >&2; exit 2 ;;
esac
echo "=== $(date '+%F %T')  stage $1 done"
