#!/usr/bin/env python3
"""Authenticate the ORIGINAL v18.44 sendback and stage its v18.43 parent.
No scientific claims are made without this check and original Q550 server replay.
"""
import io,zipfile,json,csv,hashlib
from pathlib import Path
R=Path(__file__).resolve().parents[1]
P='q550_kick203_directional_hessian_transported_jet_v18_44_bundle/'
def sha(x):return hashlib.sha256(x).hexdigest()
def read_parent_v45():
    env=json.loads((R/'ENVIRONMENT.json').read_text())
    src=R/env['v18_44_parent_file'];data=src.read_bytes()
    if sha(data)!=env['v18_44_parent_sha256']:raise RuntimeError('V18.44 parent SHA mismatch')
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        if z.testzip() is not None:raise RuntimeError('Parent CRC failure')
        if {x.split('/')[0] for x in z.namelist()}!={P[:-1]}:raise RuntimeError('Unknown parent root')
        sums=json.loads(z.read(P+'SHA256SUMS_OUTPUT.json'))['files']
        if len(sums)!=147:raise RuntimeError('Unexpected v18.44 member count')
        for rel,digest in sums.items():
            if sha(z.read(P+rel))!=digest:raise RuntimeError('Invalid original parent member: '+rel)
        proof=json.loads(z.read(P+'results/original_v18_44_proof.json'))
        if not(proof['status']=='ORIGINAL_Q550_DIRECTIONAL_HESSIAN_TRANSPORTED_SOURCE_JET_COMPLETED'
            and proof['original_integrations']==49 and proof['hessian_comparison_rows']==324
            and proof['transported_jet_rows']==36 and proof['original_ode_sha256_verified']
            and proof['source_kappa0_unchanged']):raise RuntimeError('Incomplete original v18.44 proof')
        def rows(filename):return list(csv.DictReader(io.StringIO(z.read(P+'results/'+filename).decode())))
        hrows=rows('original_v18_44_hessian_dual_contractions.csv')
        jets=rows('original_v18_44_transported_source_jets.csv')
        families=rows('original_v18_44_family_summary.csv')
        if tuple(map(len,(hrows,jets,families)))!=(324,36,12):raise RuntimeError('Bad parent row counts')
        nested=z.read(P+'inputs/v18_43_original_server_sendback.zip')
        if sha(nested)!=env['v18_43_parent_sha256']:raise RuntimeError('Nested v18.43 mismatch')
    dst=R/env['v18_43_parent_file'];dst.parent.mkdir(parents=True,exist_ok=True)
    if not dst.exists() or sha(dst.read_bytes())!=sha(nested):dst.write_bytes(nested)
    return proof,hrows,jets,families,dict(parent_sha256=sha(data),verified_members=len(sums))
