#!/usr/bin/env python3
"""Pack verified self-contained executable genealogy and logs, honoring root path."""
from pathlib import Path
import hashlib,json,zipfile
ROOT=Path(__file__).resolve().parents[1]
files=sorted(p for p in ROOT.rglob('*') if p.is_file() and 'parent_authenticated' not in p.parts and '__pycache__' not in p.parts and p.name not in ('sendback.zip','SHA256SUMS_OUTPUT.json'))
records={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
manifest={'version':'v18.48i','archive_root':ROOT.name,'file_count':len(files),'files':records}
(ROOT/'SHA256SUMS_OUTPUT.json').write_text(json.dumps(manifest,indent=2)+'\n')
with zipfile.ZipFile(ROOT/'sendback.zip','w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for p in files+[ROOT/'SHA256SUMS_OUTPUT.json']:
  z.write(p,ROOT.name+'/'+p.relative_to(ROOT).as_posix())
with zipfile.ZipFile(ROOT/'sendback.zip') as z:
 if z.testzip():raise RuntimeError('Bad ZIP')
 for name,digest in records.items():
  if hashlib.sha256(z.read(ROOT.name+'/'+name)).hexdigest()!=digest:
   raise RuntimeError('ZIP SHA mismatch: '+name)
print('SENDBACK',ROOT/'sendback.zip','members',len(files)+1)
