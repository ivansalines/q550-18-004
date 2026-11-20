#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
BASE="${Q550_BASE_ROOT:-/root/elab1/hysteresis}"
PARENT="${Q550_V47_ROOT:-$BASE/q550_kick203_hessian_kernel_compensation_v18_47_bundle}"
[[ -f "$PARENT/scripts/original_hessian_kernels_v18_47.py" ]] || { echo "Missing unzipped v18.47 parent at $PARENT"; exit 2; }
mkdir -p "$HERE/results"
cd "$HERE"
python3.13 scripts/run_audit.py --base-root "$BASE" --v47-root "$PARENT" --output-dir "$HERE/results" --restart "$@"
python3.13 scripts/pack_sendback.py
