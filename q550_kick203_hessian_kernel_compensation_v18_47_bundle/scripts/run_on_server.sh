#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON=${PYTHON:-python3.13}; command -v "$PYTHON" >/dev/null 2>&1 || PYTHON=python3
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
bash scripts/run_preflight_v18_47.sh
"$PYTHON" scripts/original_hessian_kernels_v18_47.py "$@"
"$PYTHON" scripts/plot_v18_47.py
"$PYTHON" scripts/build_manifest_v18_47.py
"$PYTHON" scripts/pack_sendback_v18_47.py
