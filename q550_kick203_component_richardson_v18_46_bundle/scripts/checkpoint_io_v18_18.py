"""Authenticate v18.17 original server sendback and nested chain to v18.7.

Keep immutable original input under inputs/; unpack predecessor to ignored .runtime/ only.
"""
import hashlib, json, zipfile
from pathlib import Path
import checkpoint_io_v18_17 as predecessor
ROOT=Path(__file__).resolve().parents[1]
INPUT=ROOT/'inputs/v18_17_original_sendback.zip'
ZIPROOT='q550_kick203_moving_antagonism_ridge_v18_17_bundle/'

def read():
    with zipfile.ZipFile(INPUT) as z:
        bad=z.testzip()
        if bad: raise RuntimeError('v18.17 CRC failure: '+bad)
        roots={n.split('/')[0] for n in z.namelist()}
        if roots != {ZIPROOT.rstrip('/')}:raise RuntimeError(f'Unexpected v18.17 ZIP root: {roots}')
        checks=json.loads(z.read(ZIPROOT+'SHA256SUMS_OUTPUT.json'))['files']
        for rel,hashval in checks.items():
            if hashlib.sha256(z.read(ZIPROOT+rel)).hexdigest()!=hashval:
                raise RuntimeError('v18.17 SHA256 mismatch: '+rel)
        inner=z.read(ZIPROOT+'inputs/v18_16_original_sendback.zip')
        status=json.loads(z.read(ZIPROOT+'results/summary.json'))['status']
    cache=ROOT/'.runtime';cache.mkdir(exist_ok=True)
    path=cache/'v18_16_original_sendback.zip'
    if path.exists() and hashlib.sha256(path.read_bytes()).digest()!=hashlib.sha256(inner).digest():
        raise RuntimeError('Conflicting .runtime v18.16 copy: remove stale transient')
    path.write_bytes(inner)
    predecessor.INPUT=path
    data,previous,provenance=predecessor.read()
    provenance.update({'v18_17_original_sha256':hashlib.sha256(INPUT.read_bytes()).hexdigest(),
                      'v18_17_verified_immutable_members':len(checks),
                      'v18_17_original_status':status})
    return data, previous, provenance
