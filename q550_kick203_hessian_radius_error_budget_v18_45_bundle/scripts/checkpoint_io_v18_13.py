#!/usr/bin/env python3
"""Read original v18.12 sendback, validate hashes, follow nested v18.11 checkpoint loader."""
from __future__ import annotations
import hashlib,json,zipfile
from pathlib import Path
import checkpoint_io_v18_11 as old_reader
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'inputs/v18_12_original_sendback.zip'

def read():
    with zipfile.ZipFile(SOURCE) as z:
        bad=z.testzip()
        if bad: raise RuntimeError('v18.12 CRC failure: '+bad)
        checks=json.loads(z.read('SHA256SUMS_OUTPUT.json'))['files']
        for file,sha in checks.items():
            got=hashlib.sha256(z.read(file)).hexdigest()
            if got!=sha:raise RuntimeError('v18.12 SHA256 failure '+file)
        nested=z.read('inputs/v18_11_original_sendback.zip')
        old_results=json.loads(z.read('results/summary.json'))
    transient=ROOT/'.runtime'
    transient.mkdir(exist_ok=True)
    path=transient/'v18_11_original_sendback.zip'
    if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest()!=hashlib.sha256(nested).hexdigest():
        raise RuntimeError('Local v18.11 input differs from v18.12 supplied source')
    path.write_bytes(nested)
    old_reader.INPUT=path
    data,p=old_reader.load()
    prov={**p,'source_v18_12_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
         'v18_12_members_verified':len(checks),'v18_12_status':old_results['status'],
         'v18_12_input_sha256':hashlib.sha256(nested).hexdigest()}
    return data,prov
