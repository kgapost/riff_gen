#!/usr/bin/env bash
# Fine-tune one album. Usage: scripts/finetune.sh <album> [--model big]  (<album> = folder in data/own)
# Thin wrapper (D37): everything also works as "python -m riffgen.train.finetune ..." from PowerShell.
set -euo pipefail
cd "$(dirname "$0")/.."
exec python -m riffgen.train.finetune "$@"
