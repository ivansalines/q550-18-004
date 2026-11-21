#!/usr/bin/env python3
"""Package executed outputs + Python generator, frozen sources, lineage and logs."""
from pathlib import Path
import zipfile,json,hashlib,datetime
ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'sendback.zip'
IGNORE={'sendback.zip'}
files=[p for p in ROOT.rglob('*') if p.is_file() and p.name not in IGNORE and '__pycache__' not in p.parts and not p.name.endswith('.pyc')]
manifest={'version':'v18.48e','archive_root':ROOT.name,'UTC_created':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'files':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
(ROOT/'SHA256SUMS_OUTPUT.json').write_text(json.dumps(manifest,indent=2)+'\n')
files=[p for p in ROOT.rglob('*') if p.is_file() and p.name not in IGNORE and '__pycache__' not in p.parts and not p.name.endswith('.pyc')]
with zipfile.ZipFile(OUTPUT,'w',zipfile.ZIP_DEFLATED,compresslevel=8) as z:
    for p in sorted(files):z.write(p,str(Path(ROOT.name)/p.relative_to(ROOT)))
with zipfile.ZipFile(OUTPUT) as z:
    if z.testzip() is not None:raise RuntimeError('Archive CRC error')
    names=z.namelist()
    assert all(name.startswith(ROOT.name+'/') for name in names)
print('SENDBACK',OUTPUT,'members',len(files))
