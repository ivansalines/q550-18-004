#!/usr/bin/env python3
"""Authenticate original v18.16 server sendback and recover its v18.15 source genealogy.

Never silently substitute locally rebuilt data for authenticated original inputs.
"""
import hashlib,json,zipfile
from pathlib import Path
import checkpoint_io_v18_16 as predecessor
ROOT=Path(__file__).resolve().parents[1]
INPUT=ROOT/'inputs/v18_16_original_sendback.zip'
EXPECTED_ROOT='q550_kick203_source_lag_interference_v18_16_bundle/'

def read():
    with zipfile.ZipFile(INPUT) as z:
        bad=z.testzip()
        if bad:raise RuntimeError(f'Original v18.16 CRC failed: {bad}')
        roots={x.split('/')[0] for x in z.namelist()}
        if roots!={EXPECTED_ROOT.rstrip('/')}:
            raise RuntimeError(f'Unexpected original v18.16 archive root: {roots}')
        checks=json.loads(z.read(EXPECTED_ROOT+'SHA256SUMS_OUTPUT.json'))['files']
        for rel,dig in checks.items():
            if hashlib.sha256(z.read(EXPECTED_ROOT+rel)).hexdigest()!=dig:
                raise RuntimeError('Original v18.16 manifest mismatch: '+rel)
        previous=z.read(EXPECTED_ROOT+'inputs/v18_15_original_sendback.zip')
        summary=json.loads(z.read(EXPECTED_ROOT+'results/summary.json'))
    cache=ROOT/'.runtime';cache.mkdir(exist_ok=True)
    path=cache/'v18_15_original_sendback.zip'
    if path.exists() and hashlib.sha256(path.read_bytes()).digest()!=hashlib.sha256(previous).digest():
        raise RuntimeError('Existing transient input differs from authenticated archive; clear .runtime/')
    path.write_bytes(previous)
    predecessor.INPUT=path
    data,prior,provenance=predecessor.read()
    provenance.update({
        'v18_16_original_sha256':hashlib.sha256(INPUT.read_bytes()).hexdigest(),
        'v18_16_manifest_members_verified':len(checks),
        'v18_16_original_status':summary['status']})
    return data,summary,provenance
