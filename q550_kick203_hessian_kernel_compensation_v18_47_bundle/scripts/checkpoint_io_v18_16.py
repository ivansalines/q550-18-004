#!/usr/bin/env python3
"""Read and authenticate exact v18.15 sendback, then nested original v18.7 sources.

No fresh ODE integration. This loader retains the original genealogy, including the
original v18.15 manifest and the unchanged nested v18.14 source checkpoint chain.
"""
import hashlib,json,zipfile
from pathlib import Path
import checkpoint_io_v18_15 as previous
ROOT=Path(__file__).resolve().parents[1]
INPUT=ROOT/'inputs'/'v18_15_original_sendback.zip'

def read():
    with zipfile.ZipFile(INPUT) as z:
        bad=z.testzip()
        if bad:raise RuntimeError(f'v18.15 CRC failed: {bad}')
        roots={p.split('/')[0] for p in z.namelist() if '/' in p}
        if roots!={'q550_kick203_age_slab_source_interference_v18_15_bundle'}:
            raise RuntimeError(f'Unexpected v18.15 ZIP root: {roots}')
        root=next(iter(roots))+'/'
        checks=json.loads(z.read(root+'SHA256SUMS_OUTPUT.json'))['files']
        for name,expected in checks.items():
            actual=hashlib.sha256(z.read(root+name)).hexdigest()
            if actual!=expected:raise RuntimeError('v18.15 SHA256 failed: '+name)
        older=z.read(root+'inputs/v18_14_original_sendback.zip')
        summary=json.loads(z.read(root+'results/summary.json'))
    cache=ROOT/'.runtime';cache.mkdir(exist_ok=True)
    path=cache/'v18_14_original_sendback.zip'
    if path.exists() and hashlib.sha256(path.read_bytes()).digest()!=hashlib.sha256(older).digest():
        raise RuntimeError('Stale v18.14 nested transient; delete .runtime, not source files')
    path.write_bytes(older)
    previous.SOURCE=path
    data,prov=previous.read()
    prov.update({'v18_15_original_sha256':hashlib.sha256(INPUT.read_bytes()).hexdigest(),
                 'v18_15_verified_members':len(checks),
                 'v18_15_status':summary['status'],
                 'v18_15_original_root':root})
    return data,summary,prov
