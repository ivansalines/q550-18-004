#!/usr/bin/env python3
"""Manifest all authored input, code and generated output, omit ephemeral runtime."""
from pathlib import Path
import hashlib,json
R=Path(__file__).resolve().parents[1]
files={}
for p in sorted(R.rglob('*')):
    if not p.is_file():continue
    name=p.relative_to(R).as_posix()
    if name=='SHA256SUMS_OUTPUT.json' or name=='inputs/v18_43_original_server_sendback.zip' or name.startswith('.runtime/') or '__pycache__' in name or name.endswith(('.pyc','.log','.tmp')):continue
    files[name]=hashlib.sha256(p.read_bytes()).hexdigest()
(R/'SHA256SUMS_OUTPUT.json').write_text(json.dumps({'algorithm':'SHA256','files':files},indent=2)+'\n')
print('V18_45_SHA_MANIFEST',len(files))
