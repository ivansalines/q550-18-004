from pathlib import Path
import zipfile,hashlib,json
root=Path(__file__).resolve().parents[1];files=[p for p in root.rglob('*') if p.is_file() and p.name!='sendback.zip' and '__pycache__' not in p.parts]
manifest={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
(root/'SHA256SUMS_OUTPUT.json').write_text(json.dumps(manifest,indent=2)+'\n')
with zipfile.ZipFile(root/'sendback.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
 for p in files+[root/'SHA256SUMS_OUTPUT.json']:z.write(p,p.relative_to(root.parent))
print('SENDBACK',root/'sendback.zip')
