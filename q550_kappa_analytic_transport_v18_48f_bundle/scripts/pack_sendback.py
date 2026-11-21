#!/usr/bin/env python3
"""Write portable v18.48f sendback with all scripts/inputs/output and SHA256 manifest."""
import hashlib,json,zipfile,datetime
from pathlib import Path
root=Path(__file__).resolve().parents[1]
final=root/'sendback.zip'
files=[p for p in root.rglob('*') if p.is_file() and p!=final and p.name!='SHA256SUMS_OUTPUT.json'
       and not p.name.endswith('.pyc') and '__pycache__' not in p.parts]
manifest={'version':'v18.48f','archive_root':root.name,
 'utc_created':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'files':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}}
(root/'SHA256SUMS_OUTPUT.json').write_text(json.dumps(manifest,indent=2)+'\n')
files.append(root/'SHA256SUMS_OUTPUT.json')
with zipfile.ZipFile(final,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=8) as z:
 for p in sorted(files):z.write(p,str(Path(root.name)/p.relative_to(root)))
with zipfile.ZipFile(final) as z:
 if z.testzip() is not None or not all(x.startswith(root.name+'/') for x in z.namelist()):
  raise RuntimeError('Bad sendback zip structure or CRC')
print('SENDBACK',final,'files',len(files))
