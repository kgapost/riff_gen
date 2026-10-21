#!/usr/bin/env bash
# Look at the raw DadaGP download. Usage: scripts/inspect_dadagp.sh [--src data/dadaGP]
# Thin wrapper (D37): everything also works as "python -m riffgen.dataset.inspect ..." from PowerShell.
set -euo pipefail
cd "$(dirname "$0")/.."
exec python -m riffgen.dataset.inspect "$@"
