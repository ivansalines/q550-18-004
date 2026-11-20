#!/usr/bin/env python3
"""Manifest of all portable scripts, original parent data, and new original results."""
from pathlib import Path
import hashlib,json
R=Path(__file__).resolve().parents[1]
files={}
for p in sorted(R.rglob('*')):
    if not p.is_file():continue
    rel=p.relative_to(R).as_posix()
    if rel=='SHA256SUMS_OUTPUT.json' or rel.startswith('.runtime/') or '__pycache__' in rel or rel.endswith(('.pyc','.log','.tmp')):continue
    files[rel]=hashlib.sha256(p.read_bytes()).hexdigest()
(R/'SHA256SUMS_OUTPUT.json').write_text(json.dumps({'algorithm':'SHA256','files':files},indent=2)+'\n')
print('V18_47_SHA256',len(files))
