#!/usr/bin/env bash
# Copy the encoded index the assignment UI needs (docs/assignment-tool.md) to <dest>, so the server can
# run on a machine that does not hold the 61k tab files. Move <dest> to that machine as data/encoded/.
#
#   usage: scripts/export_index.sh [--no-snippets] <dest>
#
# ~950 MB, or ~156 MB with --no-snippets, which leaves out tokens.bin: everything still works except
# rendering a bar of notation on a card. The tabs themselves are deliberately not copied - the UI only
# ever writes corpus-relative paths, and `riff.py albums apply` does the copying where the tabs live.
set -euo pipefail

SNIPPETS=1
if [ "${1:-}" = "--no-snippets" ]; then SNIPPETS=0; shift; fi
DEST=${1:-}
[ -n "$DEST" ] || { sed -n '2,9p' "$0"; exit 2; }

SRC=$(cd "$(dirname "$0")/.." && pwd)/data/encoded
[ -f "$SRC/meta.json" ] || { echo "no corpus at $SRC - run riff.py prepare first" >&2; exit 1; }

FILES=(meta.json stats.json bar_off.npy bar_hash.npy bar_pc.npy bar_on.npy bar_gn.npy bar_low.npy bar_sec.npy)
[ "$SNIPPETS" = 1 ] && FILES+=(tokens.bin)

mkdir -p "$DEST"
for f in "${FILES[@]}"; do
    [ -f "$SRC/$f" ] || { echo "missing: $SRC/$f" >&2; exit 1; }
    cp -v "$SRC/$f" "$DEST/$f"
done
echo
echo "$(du -sh "$DEST" | cut -f1) in $DEST"
[ "$SNIPPETS" = 1 ] || echo "(no tokens.bin: cards show fingerprints and metadata, not notation)"
echo "move it to the server machine as data/encoded/"
