"""Create an archive with functional scripts, authenticated inputs, results, dependencies and SHA-256."""
from pathlib import Path
import json,hashlib,zipfile
ROOT=Path(__file__).resolve().parents[1]
def skip(rel):return (rel.startswith('runtime/') or '/__pycache__/' in '/'+rel or rel.endswith('.pyc') or rel.endswith('_server.log') or rel in ('SHA256SUMS_OUTPUT.json','sendback.zip'))
files=[(p.relative_to(ROOT).as_posix(),p) for p in sorted(ROOT.rglob('*')) if p.is_file() and not skip(p.relative_to(ROOT).as_posix())]
manifest=dict(version='v18.48l',root=ROOT.name,file_count=len(files),files={r:hashlib.sha256(p.read_bytes()).hexdigest() for r,p in files})
(ROOT/'SHA256SUMS_OUTPUT.json').write_text(json.dumps(manifest,indent=2)+'\n')
with zipfile.ZipFile(ROOT/'sendback.zip','w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for rel,p in files:z.write(p,ROOT.name+'/'+rel)
    z.write(ROOT/'SHA256SUMS_OUTPUT.json',ROOT.name+'/SHA256SUMS_OUTPUT.json')
with zipfile.ZipFile(ROOT/'sendback.zip') as z:assert z.testzip() is None
print('SENDBACK',ROOT/'sendback.zip','VERIFIED_FILES',len(files))
