#!/usr/bin/env python3
"""Authenticated v18.14 sendback -> original v18.7 ODE-derived checkpoint arrays."""
import hashlib,json,zipfile
from pathlib import Path
import checkpoint_io_v18_14 as prior
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'inputs/v18_14_original_sendback.zip'

def read():
    with zipfile.ZipFile(SOURCE) as z:
        bad=z.testzip()
        if bad:raise RuntimeError('v18.14 ZIP CRC mismatch: '+bad)
        checks=json.loads(z.read('SHA256SUMS_OUTPUT.json'))['files']
        for file,expected in checks.items():
            if hashlib.sha256(z.read(file)).hexdigest()!=expected:
                raise RuntimeError('v18.14 SHA256 failed: '+file)
        nested=z.read('inputs/v18_13_original_sendback.zip')
        old_report=json.loads(z.read('results/summary.json'))
    runtime=ROOT/'.runtime';runtime.mkdir(exist_ok=True)
    dest=runtime/'v18_13_original_sendback.zip'
    if dest.exists() and hashlib.sha256(dest.read_bytes()).digest()!=hashlib.sha256(nested).digest():
        raise RuntimeError('v18.13 cached file conflicts with authenticated v18.14 input')
    dest.write_bytes(nested)
    prior.SOURCE=dest
    data,prov=prior.read()
    prov.update({'v18_14_original_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                 'v18_14_verified_members':len(checks),
                 'v18_14_status':old_report['status'],
                 'v18_14_events':[float(v) for v in old_report['by_tolerance'][0]['events_xi']]})
    return data,prov
