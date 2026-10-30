#!/usr/bin/env bash
# Build an album store. Usage: scripts/preprocess_album.sh <album>  (<album> = folder in data/own)
# Thin wrapper (D37): everything also works as "python -m riffgen.dataset.preprocess album ..." from PowerShell.
set -euo pipefail
cd "$(dirname "$0")/.."
exec python -m riffgen.dataset.preprocess album "$@"
