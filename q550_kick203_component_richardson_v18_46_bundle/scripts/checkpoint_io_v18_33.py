"""Authenticate the unmodified original v18.32 server archive and its genealogy."""
from pathlib import Path
import hashlib, json, zipfile
from checkpoint_io_v18_32 import read as read_v18_31
ROOT=Path(__file__).resolve().parents[1]
PREFIX='q550_kick203_invariant_manifold_phase_covariant_v18_32_bundle/'
def sha(data):return hashlib.sha256(data).hexdigest()
def read():
    env=json.loads((ROOT/'ENVIRONMENT.json').read_text())
    blob=(ROOT/env['v18_32_parent_file']).read_bytes()
    if sha(blob)!=env['v18_32_parent_sha256']:raise RuntimeError('Original v18.32 archive SHA256 mismatch')
    with zipfile.ZipFile(ROOT/env['v18_32_parent_file']) as z:
        if z.testzip():raise RuntimeError('Original v18.32 CRC failure')
        if {name.split('/')[0] for name in z.namelist()}!={PREFIX.rstrip('/')}:raise RuntimeError('Unexpected original v18.32 ZIP root')
        manifest=json.loads(z.read(PREFIX+'SHA256SUMS_OUTPUT.json'))['files']
        for path,expected in manifest.items():
            if sha(z.read(PREFIX+path))!=expected:raise RuntimeError('Original v18.32 manifest mismatch: '+path)
        proof=json.loads(z.read(PREFIX+'results/original_invariant_manifold_proof_v18_32.json'))
        if (proof.get('status')!='ORIGINAL_Q550_PHASE_COVARIANT_MANIFOLD_DIAGNOSTIC_COMPLETED'
            or proof.get('original_ODE_source_SHA256_verified') is not True
            or proof.get('candidate_grid_count')!=40 or proof.get('train_states')!=392
            or proof.get('holdout_states')!=224 or len(proof.get('full_candidate_scan',[]))!=40):
            raise RuntimeError('Authentic original v18.32 server proof missing or incompatible')
        sub=z.read(PREFIX+'inputs/v18_31_original_server_sendback.zip')
    if sha(sub)!=env['parent_sha256']:raise RuntimeError('Embedded authentic v18.31 SHA mismatch')
    rt=ROOT/'.runtime';rt.mkdir(exist_ok=True)
    (rt/'v18_31_original_server_sendback.zip').write_bytes(sub)
    data,provenance,oldproof=read_v18_31()
    provenance.update(v18_32_server_sha256=env['v18_32_parent_sha256'],
        v18_32_verified_members=len(manifest),v18_32_original_proof=proof['status'])
    return data,provenance,proof
