#!/usr/bin/env python3
"""SHA256 manifest: complete source, input and output, excluding runtime caches."""
from pathlib import Path
import json,hashlib
R=Path(__file__).resolve().parents[1]
sums={}
for p in sorted(R.rglob('*')):
    if not p.is_file():continue
    rel=p.relative_to(R).as_posix()
    if rel in ('SHA256SUMS_OUTPUT.json','inputs/v18_43_original_server_sendback.zip') or rel.startswith('.runtime/') or '__pycache__' in rel or rel.endswith(('.pyc','.log','.tmp')):continue
    sums[rel]=hashlib.sha256(p.read_bytes()).hexdigest()
(R/'SHA256SUMS_OUTPUT.json').write_text(json.dumps({'algorithm':'SHA256','files':sums},indent=2)+'\n')
print('V18_46_SHA_MANIFEST',len(sums))
