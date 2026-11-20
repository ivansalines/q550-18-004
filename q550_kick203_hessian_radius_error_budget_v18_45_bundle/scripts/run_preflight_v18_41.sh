#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON=${PYTHON:-python3.13}
command -v "$PYTHON" >/dev/null 2>&1 || PYTHON=python3
"$PYTHON" -m unittest discover -s tests -p 'test_plane_curvature_v18_41.py' -v
"$PYTHON" scripts/frozen_preflight_v18_41.py
