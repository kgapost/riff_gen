#!/usr/bin/env bash
# All five stages of the second grid, in order (see grid2.sh for what they are and why).
#
# 36 runs of 410.2M tokens, about 78 h on the one 4090. The order puts the answers first: the reproduction
# check and both models' low-batch region land in the first ~28 h, the focusing probe next, and the stages
# that only attach the new metrics to already-known cells go last.
set -u
cd "$(dirname "$0")"

# huge-old was cancelled on Nov 9 after huge-new reported: all six huge runs lose to every baseline run on
# loss_cmp (1.21 against 0.61), the architecture-neutral metric those stages existed to produce, so re-running
# eleven more Nov 5 huge cells to attach it to them would have bought nothing for ~29 h of card time.
for s in base-new huge-new base-focal base-old; do
    ./grid2.sh "$s" || echo "!!! stage $s exited $? - carrying on with the next one"
done
echo "=== $(date '+%F %T')  all stages finished"
