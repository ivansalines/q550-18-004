#!/usr/bin/env python3
"""Strict original v18.43 sendback and complete published result authentication."""
import hashlib,zipfile,json,io,csv
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PREFIX='q550_kick203_source_jet_precision_v18_43_bundle/'
def sha(data):return hashlib.sha256(data).hexdigest()
def read_parent_v44():
    env=json.loads((ROOT/'ENVIRONMENT.json').read_text())
    fn=ROOT/env['v18_43_parent_file']
    if not fn.is_file():raise RuntimeError('Missing original v18.43 parent')
    blob=fn.read_bytes()
    if sha(blob)!=env['v18_43_parent_sha256']:raise RuntimeError('Original v18.43 SHA256 mismatch')
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        if z.testzip() is not None:raise RuntimeError('Original v18.43 ZIP CRC failure')
        if {n.split('/')[0] for n in z.namelist()}!={PREFIX[:-1]}:raise RuntimeError('Unexpected v18.43 root')
        sums=json.loads(z.read(PREFIX+'SHA256SUMS_OUTPUT.json'))['files']
        if len(sums)!=147:raise RuntimeError('Parent manifest count differs from authenticated 147')
        for rel,expect in sums.items():
            if sha(z.read(PREFIX+rel))!=expect:raise RuntimeError('Bad original v18.43 file '+rel)
        proof=json.loads(z.read(PREFIX+'results/original_v18_43_proof.json'))
        if not(proof.get('status')=='ORIGINAL_Q550_SOURCE_JET_PRECISION_AUDIT_COMPLETED' and
            proof.get('original_integrations')==49 and proof.get('original_cases')==12 and
            proof.get('first_jet_rows')==60 and proof.get('second_window_rows')==144 and
            proof.get('directional_step_rows')==120 and proof.get('solver_step_rows')==48 and
            proof.get('original_ode_sha256_verified') and proof.get('source_kappa0_unchanged') and
            proof.get('no_certified_curvature')):
            raise RuntimeError('Invalid original v18.43 proof')
        def rows(name):return list(csv.DictReader(io.StringIO(z.read(PREFIX+'results/'+name).decode())))
        first=rows('original_v18_43_source_first_jets.csv')
        second=rows('original_v18_43_second_jet_windows.csv')
        directional=rows('original_v18_43_directional_step_stability.csv')
        families=rows('original_v18_43_family_summary.csv')
        if tuple(map(len,(first,second,directional,families)))!=(60,144,120,12):raise RuntimeError('Missing v18.43 original observations')
        old=z.read(PREFIX+'inputs/v18_42_original_server_sendback.zip')
        if sha(old)!=env['v18_42_parent_sha256']:raise RuntimeError('Nested original v18.42 ZIP mismatch')
    dst=ROOT/'.runtime/v18_42_original_server_sendback.zip'
    dst.parent.mkdir(exist_ok=True)
    if not dst.exists() or sha(dst.read_bytes())!=sha(old):dst.write_bytes(old)
    return proof,first,second,directional,families,dict(parent_sha256=sha(blob),verified_members=len(sums))
