#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
BASE="${Q550_BASE_ROOT:-/root/elab1/hysteresis}"
PARENT="${Q550_V47_ROOT:-$BASE/q550_kick203_hessian_kernel_compensation_v18_47_bundle}"
PYTHON="${Q550_PYTHON:-python3.13}"
"$PYTHON" -c 'import numpy,scipy,mpmath,pandas' || { echo 'Missing original Q550 Python environment';exit 2; }
[[ -f "$PARENT/scripts/original_hessian_kernels_v18_47.py" ]] || { echo "Missing unzipped v18.47 parent: $PARENT";exit 2; }
cd "$HERE"
"$PYTHON" -m unittest discover -s tests -v
"$PYTHON" scripts/run_audit.py --base-root "$BASE" --v47-root "$PARENT" --output-dir "$HERE/results" "$@"
"$PYTHON" scripts/pack_sendback.py
