#!/usr/bin/env python3
"""Strict original Q550 v18.42 archive genealogy authentication."""
from pathlib import Path
import csv,hashlib,io,json,zipfile
R=Path(__file__).resolve().parents[1]
ROOT='q550_kick203_second_derivative_noise_window_v18_42_bundle/'
def sha(x):return hashlib.sha256(x).hexdigest()
def read_parent_v43():
    env=json.loads((R/'ENVIRONMENT.json').read_text())
    p=R/env['v18_42_parent_file']
    if not p.is_file():raise RuntimeError('Original v18.42 sendback absent')
    blob=p.read_bytes()
    if sha(blob)!=env['v18_42_parent_sha256']:raise RuntimeError('Original v18.42 ZIP SHA mismatch')
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        if z.testzip() is not None:raise RuntimeError('Original v18.42 ZIP CRC failed')
        if {n.split('/')[0] for n in z.namelist()}!={ROOT.rstrip('/')}:raise RuntimeError('Unexpected v18.42 ZIP root')
        sums=json.loads(z.read(ROOT+'SHA256SUMS_OUTPUT.json'))['files']
        if len(sums)<145:raise RuntimeError('Incomplete v18.42 parent manifest')
        for k,v in sums.items():
            if sha(z.read(ROOT+k))!=v:raise RuntimeError('Parent original member SHA invalid: '+k)
        proof=json.loads(z.read(ROOT+'results/original_v18_42_proof.json'))
        if not(proof.get('status')=='ORIGINAL_Q550_SECOND_DERIVATIVE_NOISE_WINDOW_SCALING_COMPLETED'
               and proof.get('original_integrations')==73 and proof.get('original_cases')==12
               and proof.get('vector_temporal_samples')==1548 and proof.get('window_rows')==48
               and proof.get('original_ode_sha256_verified') and proof.get('source_kappa0_unchanged')
               and proof.get('no_certified_curvature')):
            raise RuntimeError('v18.42 original Q550 proof invalid')
        def rows(fn):return list(csv.DictReader(io.StringIO(z.read(ROOT+'results/'+fn).decode())))
        cases=rows('original_v18_42_family_summary.csv')
        profiles=rows('original_v18_42_odd_vector_profiles.csv')
        windows=rows('original_v18_42_window_scaling.csv')
        stencils=rows('original_v18_42_second_derivative_stencils.csv')
        tolerance=rows('original_v18_42_tolerance_noise.csv')
        if tuple(map(len,(cases,profiles,windows,stencils,tolerance)))!=(12,1548,48,144,16):raise RuntimeError('Incomplete original v18.42 results')
        old=z.read(ROOT+'inputs/v18_41_original_server_sendback.zip')
        if sha(old)!=env['v18_41_parent_sha256']:raise RuntimeError('Nested v18.41 original ZIP SHA mismatch')
    staged=R/'.runtime/v18_41_original_server_sendback.zip';staged.parent.mkdir(exist_ok=True)
    if not staged.exists() or sha(staged.read_bytes())!=sha(old):staged.write_bytes(old)
    return proof,cases,profiles,windows,stencils,tolerance,dict(parent_sha256=sha(blob),verified_members=len(sums))
