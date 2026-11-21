#!/usr/bin/env python3
"""Authenticate EVERY archived v18.48h member, then materialize exact parent root.

Never assume archive path/ZIP hierarchy from its filename. Refuse zip-slip.
Original user-server sendback is preserved byte-for-byte in inputs.
"""
from pathlib import Path
import hashlib,json,zipfile
ROOT=Path(__file__).resolve().parents[1]

def sha(bs):return hashlib.sha256(bs).hexdigest()
def extract():
    info=json.loads((ROOT/'inputs/PARENT_SHA256.json').read_text())
    p=ROOT/'inputs'/info['filename']
    if sha(p.read_bytes())!=info['sha256']:raise RuntimeError('v18.48h parent SHA256 drift')
    with zipfile.ZipFile(p) as z:
        if z.testzip():raise RuntimeError('parent archive CRC failure')
        members=z.namelist()
        roots={v.split('/')[0] for v in members}
        if len(roots)!=1:raise RuntimeError('Unexpected multiple parent roots')
        prefix=next(iter(roots))+'/'
        mf=prefix+'SHA256SUMS_OUTPUT.json'
        if mf not in members:raise RuntimeError('parent manifest missing')
        manifest=json.loads(z.read(mf))
        expected=manifest['files']
        if manifest.get('version')!='v18.48h' or len(expected)+1!=info['expected_member_count']:
            raise RuntimeError('v18.48h count/version mismatch')
        if set(members)!={prefix+n for n in expected}|{mf}:
            raise RuntimeError('Unmanifested / missing parent entry')
        for name,digest in expected.items():
            parts=Path(name).parts
            if not parts or any(x in ('..','.','') for x in parts) or name.startswith('/'):
                raise RuntimeError('Unsafe path in parent zip')
            if sha(z.read(prefix+name))!=digest:raise RuntimeError('Parent SHA drift '+name)
        dest=ROOT/'parent_authenticated'/next(iter(roots))
        dest.mkdir(parents=True,exist_ok=True)
        for name in expected:
            q=dest/name;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(z.read(prefix+name))
        (dest/'SHA256SUMS_OUTPUT.json').write_bytes(z.read(mf))
    return dest,dict(zip_sha256=info['sha256'],verified_members=len(expected)+1,archive_root=next(iter(roots)))
if __name__=='__main__':
    p,d=extract();print('AUTHENTICATED',p, json.dumps(d))
