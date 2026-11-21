#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
BASE="${Q550_BASE_ROOT:-/root/elab1/hysteresis}"
PYTHON="${Q550_PYTHON:-python3.13}"
"$PYTHON" -c 'import numpy,scipy,mpmath' || { echo 'Missing authenticated Q550 Python environment'; exit 2; }
cd "$HERE"
"$PYTHON" -m unittest discover -s tests -v
"$PYTHON" scripts/run_audit.py --base-root "$BASE" --output-dir "$HERE/results" "$@"
"$PYTHON" scripts/pack_sendback.py
