#!/usr/bin/env bash
# All five stages of the second grid, in order (see grid2.sh for what they are and why).
#
# 36 runs of 410.2M tokens, about 78 h on the one 4090. The order puts the answers first: the reproduction
# check and both models' low-batch region land in the first ~28 h, the focusing probe next, and the stages
# that only attach the new metrics to already-known cells go last.
set -u
cd "$(dirname "$0")"

for s in base-new huge-new base-focal base-old huge-old; do
    ./grid2.sh "$s" || echo "!!! stage $s exited $? - carrying on with the next one"
done
echo "=== $(date '+%F %T')  all stages finished"
