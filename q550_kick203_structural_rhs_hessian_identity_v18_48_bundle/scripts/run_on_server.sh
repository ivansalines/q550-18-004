#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${Q550_V1848_ADAPTER:?Set Q550_V1848_ADAPTER to the authenticated model adapter PY file}"
: "${Q550_V1848_STATES:?Set Q550_V1848_STATES to an NPZ with states (N,11)}"
python3 "$ROOT/scripts/audit_v18_48.py" --adapter "$Q550_V1848_ADAPTER" --states "$Q550_V1848_STATES" --output "$ROOT/outputs/v18_48_report.json" "$@"
