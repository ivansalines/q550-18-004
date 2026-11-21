#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
BASE="${Q550_BASE_ROOT:-/root/elab1/hysteresis}"
PY="${Q550_PYTHON:-python3.13}"
cd "$HERE"
"$PY" -c 'import numpy,scipy,mpmath' || { echo "Missing numpy/scipy/mpmath in $PY" >&2; exit 2; }
"$PY" -m unittest discover -s tests -v
"$PY" scripts/extract_parent.py
# Recalculate actual 11D IVPs from authenticated original v18.48h code.
# No interpolation of states or observers and no edits to historic sources.
PARENT="$HERE/parent_authenticated/q550_kappa_physical_section_v18_48h_bundle"
mkdir -p "$HERE/inputs/refined_v18_48h" "$HERE/results"
"$PY" "$PARENT/scripts/run_audit.py" \
  --base-root "$BASE" --sections "${Q550_I_SECTIONS:-41}" \
  --output-dir "$HERE/inputs/refined_v18_48h" "$@"
"$PY" scripts/run_audit.py \
  --parent-results "$HERE/inputs/refined_v18_48h" \
  --output-dir "$HERE/results"
"$PY" scripts/pack_sendback.py
