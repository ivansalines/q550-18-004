#!/usr/bin/env python3
"""Authenticate ORIGINAL v18.41 server sendback and stage v18.40 archive."""
from pathlib import Path
import csv,hashlib,io,json,zipfile
R=Path(__file__).resolve().parents[1]
PREFIX='q550_kick203_odd_source_plane_curvature_v18_41_bundle/'
def digest(raw):return hashlib.sha256(raw).hexdigest()
def read_parent_v42():
    env=json.loads((R/'ENVIRONMENT.json').read_text())
    blob=(R/env['v18_41_parent_file']).read_bytes()
    if digest(blob)!=env['v18_41_parent_sha256']:raise RuntimeError('Authentic v18.41 ZIP SHA256 mismatch')
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        if z.testzip() is not None:raise RuntimeError('v18.41 original ZIP CRC failure')
        if {n.split('/')[0] for n in z.namelist()}!={PREFIX.rstrip('/')}:raise RuntimeError('Unexpected original ZIP root')
        hashes=json.loads(z.read(PREFIX+'SHA256SUMS_OUTPUT.json'))['files']
        if len(hashes)<130:raise RuntimeError('Incomplete original v18.41 manifest')
        for fn,expected in hashes.items():
            if digest(z.read(PREFIX+fn))!=expected:raise RuntimeError('Original v18.41 member invalid: '+fn)
        proof=json.loads(z.read(PREFIX+'results/original_v18_41_proof.json'))
        if not(proof.get('status')=='ORIGINAL_Q550_ODD_SOURCE_RESPONSE_PLANE_CURVATURE_COMPLETED'
               and proof.get('original_integrations')==61 and proof.get('original_cases')==12
               and proof.get('original_samples')==1548 and proof.get('resolution_rows')==48
               and proof.get('original_ode_sha256_verified') and proof.get('source_kappa0_unchanged')
               and proof.get('no_material_metric_certified') and proof.get('no_physical_constraint_certified')):
            raise RuntimeError('Original v18.41 proof not valid')
        def rows(name):return list(csv.DictReader(io.StringIO(z.read(PREFIX+'results/'+name).decode())))
        family=rows('original_v18_41_family_summary.csv')
        profile=rows('original_v18_41_plane_profiles.csv')
        resolution=rows('original_v18_41_curvature_resolution.csv')
        if (len(family),len(profile),len(resolution))!=(12,1548,48):raise RuntimeError('Incomplete original v18.41 CSVs')
        prev=z.read(PREFIX+'inputs/v18_40_original_server_sendback.zip')
        if digest(prev)!=env['v18_40_parent_sha256']:raise RuntimeError('Nested original v18.40 SHA mismatch')
    fp=R/'.runtime/v18_40_original_server_sendback.zip';fp.parent.mkdir(exist_ok=True)
    if not fp.exists() or digest(fp.read_bytes())!=digest(prev):fp.write_bytes(prev)
    return proof,family,profile,resolution,dict(parent_sha256=digest(blob),verified_members=len(hashes))
