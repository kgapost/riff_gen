#!/usr/bin/env bash
# Evaluate a model. Usage: scripts/evaluate.sh --checkpoint <best.pt> --store <store> --split holdout
# Thin wrapper (D37): everything also works as "python -m riffgen.eval ..." from PowerShell.
set -euo pipefail
cd "$(dirname "$0")/.."
exec python -m riffgen.eval "$@"
