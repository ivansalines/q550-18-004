#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
BASE="${Q550_BASE_ROOT:-/root/elab1/hysteresis}"
PY="${Q550_PYTHON:-python3.13}"
"$PY" -c 'import numpy,scipy,mpmath' || { echo "Missing numpy/scipy/mpmath in $PY" >&2; exit 2; }
cd "$HERE"
"$PY" -m unittest discover -s tests -v
"$PY" scripts/run_audit.py --base-root "$BASE" --output-dir "$HERE/results" "$@"
if "$PY" -c "import matplotlib" >/dev/null 2>&1; then
  "$PY" scripts/plot_preflight.py
else
  echo "matplotlib not available; optional figure skipped"
fi
"$PY" scripts/pack_sendback.py
