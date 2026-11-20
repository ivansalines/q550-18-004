#!/usr/bin/env python3
"""Authenticate entire original v18.39 sendback; stage unmodified v18.38 ancestry."""
from pathlib import Path
import zipfile,io,json,csv,hashlib
R=Path(__file__).resolve().parents[1]
PREFIX='q550_kick203_full_tangent_source_rotation_v18_39_bundle/'
def sha(b):return hashlib.sha256(b).hexdigest()
def read_parent_v40():
    env=json.loads((R/'ENVIRONMENT.json').read_text())
    blob=(R/env['v18_39_parent_file']).read_bytes()
    if sha(blob)!=env['v18_39_parent_sha256']:raise RuntimeError('v18.39 parent original ZIP hash mismatch')
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        if z.testzip() is not None:raise RuntimeError('v18.39 parent ZIP CRC error')
        if {n.split('/')[0] for n in z.namelist()}!={PREFIX[:-1]}:raise RuntimeError('Unrecognized v18.39 parent root')
        manifest=json.loads(z.read(PREFIX+'SHA256SUMS_OUTPUT.json'))['files']
        for n,h in manifest.items():
            if sha(z.read(PREFIX+n))!=h:raise RuntimeError('v18.39 parent member hash mismatch: '+n)
        proof=json.loads(z.read(PREFIX+'results/original_v18_39_proof.json'))
        if (proof.get('status')!='ORIGINAL_Q550_FULL_TANGENT_SOURCE_ROTATION_COMPLETED' or
            proof.get('original_cases')!=12 or proof.get('original_integrations')!=61 or
            proof.get('vector_samples')!=120 or not proof.get('source_kappa0_unchanged') or
            not proof.get('original_ode_sha256_verified') or not proof.get('no_physical_constraint_certified')):
            raise RuntimeError('Original v18.39 original Q550 proof fails')
        frame=list(csv.DictReader(io.StringIO(z.read(PREFIX+'results/original_v18_39_frame_transport.csv').decode())))
        vec=list(csv.DictReader(io.StringIO(z.read(PREFIX+'results/original_v18_39_vector_source_9d.csv').decode())))
        if len(frame)!=12 or len(vec)!=120:raise RuntimeError('Authentic v18.39 output row count mismatch')
        parent38=z.read(PREFIX+'inputs/v18_38_original_server_sendback.zip')
        if sha(parent38)!=env['v18_38_parent_sha256']:raise RuntimeError('Nested original v18.38 mismatch')
    rt=R/'.runtime';rt.mkdir(exist_ok=True)
    (rt/'v18_38_original_server_sendback.zip').write_bytes(parent38)
    return proof,frame,vec,dict(parent_sha256=sha(blob),verified_members=len(manifest))
