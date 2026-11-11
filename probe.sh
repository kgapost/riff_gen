#!/usr/bin/env bash
# Run the fourth-round probes (see probe.py for what they are and why), logging to runs/probe_log.txt.
#
# Nine runs, ~18.5 h. Finished runs are skipped on their results row, so this can be stopped, restarted,
# or lost to a reboot at any point and will carry on where it left off - which matters on this machine:
# it has been rebooted from another account around 10:30 on Oct 12, Oct 15, Oct 30 and Nov 9.
#
# Do NOT re-prepare data/encoded_grid while this runs: it would change the fixed validation gaps and void
# both this table and runs/grid2's.
set -u
cd "$(dirname "$0")"
echo "=== $(date '+%F %T')  probes starting"
python probe.py
echo "=== $(date '+%F %T')  probes done"
