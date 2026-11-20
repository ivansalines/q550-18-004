"""Verify immutable v18.34 server sendback and stage the genuine v18.33 ancestor.

This verifies all v18.34 member hashes. Older genealogy is nested unchanged and
can be fully traversed by checkpoint_io_v18_34.read_parent at original run time.
"""
from pathlib import Path
import hashlib,zipfile,json,io
ROOT=Path(__file__).resolve().parents[1]
PREFIX='q550_kick203_transport_defect_overlap_gauge_v18_34_bundle/'
P33='q550_kick203_cofactor_selection_surface_return_v18_33_bundle/'
def sha(b):return hashlib.sha256(b).hexdigest()
def read_parent_v35():
    env=json.loads((ROOT/'ENVIRONMENT.json').read_text())
    b=(ROOT/env['v18_34_parent_file']).read_bytes()
    if sha(b)!=env['v18_34_parent_sha256']:raise RuntimeError('Original v18.34 SHA mismatch')
    with zipfile.ZipFile(io.BytesIO(b)) as z:
        if z.testzip():raise RuntimeError('Original v18.34 ZIP CRC failed')
        if {p.split('/')[0] for p in z.namelist()}!={PREFIX[:-1]}:raise RuntimeError('Original v18.34 root mismatch')
        manifest=json.loads(z.read(PREFIX+'SHA256SUMS_OUTPUT.json'))['files']
        for name,h in manifest.items():
            if sha(z.read(PREFIX+name))!=h:raise RuntimeError('Original v18.34 member hash mismatch: '+name)
        proof=json.loads(z.read(PREFIX+'results/original_v18_34_proof.json'))
        endpoints=z.read(PREFIX+'results/original_v18_34_budget_samples.csv').decode()
        sub=z.read(PREFIX+'inputs/v18_33_original_server_sendback.zip')
    if proof.get('status')!='ORIGINAL_Q550_TRANSPORT_DEFECT_BUDGET_OVERLAP_GAUGE_COMPLETED' or proof.get('original_ode_integrations')!=13 or proof.get('total_sample_count')!=972:
        raise RuntimeError('Original v18.34 proof incompatible')
    if sha(sub)!=env['v18_33_parent_sha256']:raise RuntimeError('Nested v18.33 digest mismatch')
    with zipfile.ZipFile(io.BytesIO(sub)) as z:
        if {p.split('/')[0] for p in z.namelist()}!={P33[:-1]}:raise RuntimeError('Nested v18.33 root mismatch')
        model=json.loads(z.read(P33+'results/original_v18_33_surface_retests.json'))
        oldproof=json.loads(z.read(P33+'results/original_v18_33_proof.json'))
    if oldproof.get('original_ode_integrations')!=43 or len(model['candidates'])!=7:
        raise RuntimeError('Nested v18.33 original certificate incompatible')
    rt=ROOT/'.runtime';rt.mkdir(exist_ok=True)
    (rt/'v18_33_original_server_sendback.zip').write_bytes(sub)
    return proof,model,endpoints,dict(parent_sha=env['v18_34_parent_sha256'],verified_members=len(manifest))
