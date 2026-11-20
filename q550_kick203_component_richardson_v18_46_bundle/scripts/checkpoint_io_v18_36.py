"""Cryptographically validate original v18.35 and stage its authenticated genealogy."""
from pathlib import Path
import json,hashlib,zipfile,io
R=Path(__file__).resolve().parents[1]
P='q550_kick203_near_surface_boundary_layer_v18_35_bundle/'
def sha(b):return hashlib.sha256(b).hexdigest()
def read_parent_v36():
    env=json.loads((R/'ENVIRONMENT.json').read_text())
    raw=(R/env['v18_35_parent_file']).read_bytes()
    if sha(raw)!=env['v18_35_parent_sha256']:raise RuntimeError('Original v18.35 parent SHA mismatch')
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        if z.testzip():raise RuntimeError('Original v18.35 ZIP CRC failure')
        if {x.split('/')[0] for x in z.namelist()}!={P[:-1]}:raise RuntimeError('Unexpected v18.35 root')
        manifest=json.loads(z.read(P+'SHA256SUMS_OUTPUT.json'))['files']
        for fn,expected in manifest.items():
            if sha(z.read(P+fn))!=expected:raise RuntimeError('Original v18.35 member hash mismatch: '+fn)
        proof=json.loads(z.read(P+'results/original_v18_35_proof.json'))
        oldparent=z.read(P+'inputs/v18_34_original_server_sendback.zip')
        cases=z.read(P+'results/original_v18_35_case_summary.csv').decode()
    if proof.get('status')!='ORIGINAL_Q550_NEAR_SURFACE_MOVING_FRAME_AND_DEFECT_BUDGET_COMPLETED' or proof.get('original_ode_integrations')!=15 or proof.get('case_count')!=12:
        raise RuntimeError('v18.35 original validation absent or incompatible')
    if sha(oldparent)!=env['v18_34_parent_sha256']:raise RuntimeError('v18.34 embedded SHA mismatch')
    rt=R/'.runtime';rt.mkdir(exist_ok=True)
    (rt/'v18_34_original_server_sendback.zip').write_bytes(oldparent)
    return proof,cases,dict(parent_sha256=sha(raw),verified_members=len(manifest))
