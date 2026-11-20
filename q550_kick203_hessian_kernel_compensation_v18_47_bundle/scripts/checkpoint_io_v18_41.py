#!/usr/bin/env python3
"""Authenticate the complete v18.40 ORIGINAL server sendback, stage v18.39 ancestry."""
import csv,hashlib,io,json,zipfile
from pathlib import Path
R=Path(__file__).resolve().parents[1]
PREFIX='q550_kick203_gauge_robust_temporal_v18_40_bundle/'
def sha(b):return hashlib.sha256(b).hexdigest()
def read_parent_v41():
    env=json.loads((R/'ENVIRONMENT.json').read_text())
    blob=(R/env['v18_40_parent_file']).read_bytes()
    if sha(blob)!=env['v18_40_parent_sha256']:raise RuntimeError('v18.40 ORIGINAL ZIP SHA mismatch')
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        if z.testzip() is not None:raise RuntimeError('Original v18.40 ZIP CRC mismatch')
        if {n.split('/')[0] for n in z.namelist()}!={PREFIX[:-1]}:raise RuntimeError('Unexpected v18.40 root')
        hashes=json.loads(z.read(PREFIX+'SHA256SUMS_OUTPUT.json'))['files']
        if len(hashes)<110:raise RuntimeError('Too few original authenticated files')
        for n,digest in hashes.items():
            if sha(z.read(PREFIX+n))!=digest:raise RuntimeError('Original v18.40 member SHA mismatch: '+n)
        proof=json.loads(z.read(PREFIX+'results/original_v18_40_proof.json'))
        if not (proof.get('status')=='ORIGINAL_Q550_GAUGE_ROBUST_SOURCE_TEMPORAL_COMPLETED'
                and proof.get('original_integrations')==61 and proof.get('original_cases')==12
                and proof.get('temporal_samples')==168 and proof.get('resolution_rows')==48
                and proof.get('source_kappa0_unchanged') and proof.get('original_ode_sha256_verified')
                and proof.get('no_material_metric_certified') and proof.get('no_physical_constraint_certified')):
            raise RuntimeError('Original v18.40 proof invalid')
        def rows(s):return list(csv.DictReader(io.StringIO(z.read(PREFIX+'results/'+s).decode())))
        temporal=rows('original_v18_40_temporal_sources.csv')
        families=rows('original_v18_40_family_summary.csv')
        resolution=rows('original_v18_40_resolution.csv')
        if not (len(temporal)==168 and len(families)==12 and len(resolution)==48):raise RuntimeError('Original v18.40 CSV size mismatch')
        nested=z.read(PREFIX+'inputs/v18_39_original_server_sendback.zip')
        if sha(nested)!=env['v18_39_parent_sha256']:raise RuntimeError('Nested v18.39 parent SHA mismatch')
    d=R/'.runtime';d.mkdir(exist_ok=True)
    (d/'v18_39_original_server_sendback.zip').write_bytes(nested)
    return proof,temporal,families,resolution,dict(parent_sha256=sha(blob),verified_members=len(hashes))
