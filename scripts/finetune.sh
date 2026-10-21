#!/usr/bin/env bash
# Fine-tune one album. Usage: scripts/finetune.sh configs/albums/<album>.yaml [--model big]
# Thin wrapper (D37): everything also works as "python -m riffgen.train.finetune ..." from PowerShell.
set -euo pipefail
cd "$(dirname "$0")/.."
exec python -m riffgen.train.finetune "$@"
