#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
BASE="${Q550_BASE_ROOT:-/root/elab1/hysteresis}"
PARENT="${Q550_V47_ROOT:-$BASE/q550_kick203_hessian_kernel_compensation_v18_47_bundle}"
PYTHON="${Q550_PYTHON:-python3.13}"
[[ -f "$PARENT/scripts/original_hessian_kernels_v18_47.py" ]] || { echo "Missing extracted v18.47: $PARENT"; exit 2; }
"$PYTHON" -c 'import numpy, scipy, mpmath; print("Python dependencies verified")'
mkdir -p "$HERE/results"
cd "$HERE"
"$PYTHON" scripts/run_audit.py --base-root "$BASE" --v47-root "$PARENT" --output-dir "$HERE/results" --restart "$@"
"$PYTHON" scripts/pack_sendback.py
