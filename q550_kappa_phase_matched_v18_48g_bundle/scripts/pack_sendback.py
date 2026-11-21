#!/usr/bin/env python3
"""Deterministic v18.48g sendback genealogy, archived paths checked."""
import hashlib,json,pathlib,zipfile
ROOT=pathlib.Path(__file__).resolve().parents[1]
paths=[p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name not in ('sendback.zip','SHA256SUMS_OUTPUT.json','SHA256SUMS.json')]
records={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
(ROOT/'SHA256SUMS_OUTPUT.json').write_text(json.dumps({'version':'v18.48g','files':records},indent=2)+'\n')
with zipfile.ZipFile(ROOT/'sendback.zip','w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for p in sorted(paths)+[ROOT/'SHA256SUMS_OUTPUT.json']:
  z.write(p,arcname=ROOT.name+'/'+str(p.relative_to(ROOT)))
with zipfile.ZipFile(ROOT/'sendback.zip') as z:
 bad=z.testzip()
 if bad:raise RuntimeError('Corrupted zip member '+bad)
print('SENDBACK',ROOT/'sendback.zip', 'members',len(paths)+1)
