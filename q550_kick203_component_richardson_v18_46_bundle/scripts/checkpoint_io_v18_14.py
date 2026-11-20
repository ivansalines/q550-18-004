#!/usr/bin/env python3
"""Validate full v18.13 sendback and load the original v18.7/6 frozen ODE material source arrays."""
from __future__ import annotations
import hashlib, json, zipfile
from pathlib import Path
import checkpoint_io_v18_13 as prior
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'inputs/v18_13_original_sendback.zip'

def read():
    with zipfile.ZipFile(SOURCE) as z:
        failed=z.testzip()
        if failed: raise RuntimeError('v18.13 archive CRC mismatch: '+failed)
        checks=json.loads(z.read('SHA256SUMS_OUTPUT.json'))['files']
        for k,v in checks.items():
            if hashlib.sha256(z.read(k)).hexdigest()!=v:
                raise RuntimeError('v18.13 archive SHA256 mismatch: '+k)
        prior_source=z.read('inputs/v18_12_original_sendback.zip')
        prior_summary=json.loads(z.read('results/summary.json'))
    tmp=ROOT/'.runtime';tmp.mkdir(exist_ok=True)
    nested=tmp/'v18_12_original_sendback.zip'
    if nested.exists() and nested.read_bytes()!=prior_source:
        raise RuntimeError('Existing transient v18.12 conflicts with authenticated v18.13 input')
    nested.write_bytes(prior_source)
    prior.SOURCE=nested
    data,prov=prior.read()
    prov.update({'v18_13_original_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        'v18_13_verified_members':len(checks),
        'v18_13_status':prior_summary['status'],
        'v18_13_nested_v18_12_sha256':hashlib.sha256(prior_source).hexdigest()})
    return data,prov
