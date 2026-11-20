#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PYTHON=${PYTHON:-python3.13}
command -v "$PYTHON" >/dev/null 2>&1 || PYTHON=python3
V47_ROOT=${V47_ROOT:-../q550_kick203_hessian_kernel_compensation_v18_47_bundle}
BASE_ROOT=${BASE_ROOT:-/root/elab1/hysteresis}
"$PYTHON" scripts/compare_kappa.py --v47-root "$V47_ROOT" --base-root "$BASE_ROOT" --output results/kappa_comparison.json
