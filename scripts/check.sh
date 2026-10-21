#!/usr/bin/env bash
# Check a folder. Usage: scripts/check.sh --album <album> --folder <dir> [--fix --mark]
# Thin wrapper (D37): everything also works as "python -m riffgen.check ..." from PowerShell.
set -euo pipefail
cd "$(dirname "$0")/.."
exec python -m riffgen.check "$@"
