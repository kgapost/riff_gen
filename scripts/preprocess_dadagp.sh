#!/usr/bin/env bash
# Build data/processed/dadagp. Usage: scripts/preprocess_dadagp.sh [--workers 12]
# Thin wrapper (D37): everything also works as "python -m riffgen.dataset.preprocess dadagp ..." from PowerShell.
set -euo pipefail
cd "$(dirname "$0")/.."
exec python -m riffgen.dataset.preprocess dadagp "$@"
