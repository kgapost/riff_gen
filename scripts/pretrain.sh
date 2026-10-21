#!/usr/bin/env bash
# Pretrain on DadaGP. Usage: scripts/pretrain.sh --model baseline|big|huge [--resume] [key=value ...]
# Thin wrapper (D37): everything also works as "python -m riffgen.train.pretrain ..." from PowerShell.
set -euo pipefail
cd "$(dirname "$0")/.."
exec python -m riffgen.train.pretrain "$@"
