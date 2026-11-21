#!/usr/bin/env python3
"""Package full executed genealogy with actual archive root hierarchy and SHA checks."""
from pathlib import Path
import hashlib,json,zipfile
ROOT=Path(__file__).resolve().parents[1]
paths=sorted(p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name not in ('sendback.zip','SHA256SUMS_OUTPUT.json','SHA256SUMS.json'))
records={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
(ROOT/'SHA256SUMS_OUTPUT.json').write_text(json.dumps({'version':'v18.48h','files':records},indent=2)+'\n')
with zipfile.ZipFile(ROOT/'sendback.zip','w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for p in paths+[ROOT/'SHA256SUMS_OUTPUT.json']:
  z.write(p,arcname=ROOT.name+'/'+str(p.relative_to(ROOT)))
with zipfile.ZipFile(ROOT/'sendback.zip') as z:
 if z.testzip():raise RuntimeError('sendback ZIP CRC failure')
 names=z.namelist()
 if not all(x.startswith(ROOT.name+'/') for x in names):raise RuntimeError('Archive root hierarchy mismatch')
print('SENDBACK',ROOT/'sendback.zip',len(paths)+1,'members')
