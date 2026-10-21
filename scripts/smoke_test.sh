#!/usr/bin/env bash
# Run the whole pipeline on synthetic data (CPU, ~1 min). Usage: scripts/smoke_test.sh [--huge]
# Thin wrapper (D37): everything also works as "python -m riffgen.tools.smoke ..." from PowerShell.
set -euo pipefail
cd "$(dirname "$0")/.."
exec python -m riffgen.tools.smoke "$@"
