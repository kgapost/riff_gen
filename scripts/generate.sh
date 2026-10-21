#!/usr/bin/env bash
# Fill gaps. Usage: scripts/generate.sh --album <album> --song <file.gp5> [--gap 33-40] [--section bridge]
# Thin wrapper (D37): everything also works as "python -m riffgen.generate ..." from PowerShell.
set -euo pipefail
cd "$(dirname "$0")/.."
exec python -m riffgen.generate "$@"
