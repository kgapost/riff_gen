#!/usr/bin/env bash
# Third grid (Nov 9): two cells, closing the one edge the second grid left open.
#
# Every batch size in grid 2 wanted an absolute learning rate in the 4-8e-4 band, nearly independent of the
# batch:
#
#     b16k   best lr 3.7e-04 -> 0.19654   HIGH edge - nothing higher was tried
#     b32k   best lr 7.5e-04 -> 0.19920   interior, a clean U
#     b64k   best lr 7.5e-04 -> 0.19916   LOW edge - nothing lower was tried
#
# so the sqrt(batch) scaling in grid_points fans the lr axis out per batch further than the optimum actually
# moves, and both pegged edges are artifacts of that fanning rather than of the loss surface. b16k holds the
# best run of the whole grid (0.19654) and is also the one batch never given a learning rate near that band,
# which makes it the only cell worth paying for: --lrs 0.707,1 at --batches 0.125 lands on 5.29e-04 and
# 7.48e-04 (verified against riff.py's own suggest/grid_points, not by hand).
#
# b64k's low edge was deliberately left alone: the top four runs of grid 2 span 0.0028 against a measured
# reproduction gap of 0.00125, so a cell there would land inside the noise. Nothing closer than ~0.0013 can be
# ranked by adding cells - that needs repeated seeds - so for the final model, pick on throughput among the
# tied top group rather than on the nominal winner.
#
# Everything that decides the validation loss is pinned to the grid 2 values so these two rows join that table:
# the same corpus (data_encoded/encoded_grid - do not re-prepare it), the same 410.2M-token budget (2.00 epochs of that
# corpus), the same 400 evaluation gaps, cadence, seed, micro-batch and activation checkpointing.
#
# Finished runs are skipped, so this can be stopped and started again at will.
set -u
cd "$(dirname "$0")"

COMMON="--data data_encoded/encoded_grid --out runs/grid2 --budget 410200000
        --eval-gaps 400 --eval-every 0.125 --seed 1234 --workers 4"
BASE="--models baseline --micro-tokens 16384 --ckpt"

echo "=== $(date '+%F %T')  grid3: b16k at lr 5.3e-04 and 7.5e-04"
python riff.py grid $COMMON $BASE --smooth 0 --batches 0.125 --lrs 0.707,1      # 2 runs, ~4-5 h
echo "=== $(date '+%F %T')  grid3 done"
