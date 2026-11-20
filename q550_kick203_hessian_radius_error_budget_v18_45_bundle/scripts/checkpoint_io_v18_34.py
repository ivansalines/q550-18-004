"""Authenticate v18.33 original sendback, and stage its v18.32 checkpoint safely."""
from pathlib import Path
import hashlib,json,zipfile,io
ROOT=Path(__file__).resolve().parents[1]
P33='q550_kick203_cofactor_selection_surface_return_v18_33_bundle/'
P32='q550_kick203_invariant_manifold_phase_covariant_v18_32_bundle/'
def sha(b):return hashlib.sha256(b).hexdigest()
def read_parent(check_deep=True):
    env=json.loads((ROOT/'ENVIRONMENT.json').read_text())
    blob=(ROOT/env['v18_33_parent_file']).read_bytes()
    if sha(blob)!=env['v18_33_parent_sha256']:raise RuntimeError('v18.33 parent SHA mismatch')
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        if z.testzip():raise RuntimeError('v18.33 parent ZIP CRC failed')
        if {n.split('/')[0] for n in z.namelist()}!={P33[:-1]}:raise RuntimeError('v18.33 ZIP root mismatch')
        manifest=json.loads(z.read(P33+'SHA256SUMS_OUTPUT.json'))['files']
        for n,h in manifest.items():
            if sha(z.read(P33+n))!=h:raise RuntimeError('v18.33 hash mismatch '+n)
        proof=json.loads(z.read(P33+'results/original_v18_33_proof.json'))
        data=json.loads(z.read(P33+'results/original_v18_33_surface_retests.json'))
        csv_returns=z.read(P33+'results/original_v18_33_clock_return.csv').decode()
        csv_overlap=z.read(P33+'results/original_v18_33_posture_overlap.csv').decode()
        prior=z.read(P33+'inputs/v18_32_original_server_sendback.zip')
    if proof.get('status')!='ORIGINAL_Q550_COFACTOR_SELECTION_REPAIR_SURFACE_RETURN_COMPLETED' or proof.get('original_ode_integrations')!=43 or len(data['candidates'])!=7 or proof.get('numerically_passing')!=0:
        raise RuntimeError('v18.33 original proof unexpected')
    if sha(prior)!=env['v18_32_parent_sha256']:raise RuntimeError('v18.32 nested SHA mismatch')
    rt=ROOT/'.runtime';rt.mkdir(exist_ok=True)
    (rt/'v18_32_original_server_sendback.zip').write_bytes(prior)
    # Historical loaders authenticate earlier genealogical checkpoints independently.
    from checkpoint_io_v18_33 import read
    original,provenance,prior_proof=read()
    return original,proof,data,csv_returns,csv_overlap,dict(parent_sha=env['v18_33_parent_sha256'],parent_verified_members=len(manifest),historical=provenance)
