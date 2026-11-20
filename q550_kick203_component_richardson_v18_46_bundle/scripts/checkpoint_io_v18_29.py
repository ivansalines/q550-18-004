"""Authenticate the complete original server v18.28 nested executable lineage."""
from __future__ import annotations
from pathlib import Path
import json,zipfile,hashlib
import checkpoint_io_v18_28 as parent
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'inputs/v18_28_original_server_sendback.zip'
PREFIX='q550_kick203_second_variational_source_floor_v18_28_bundle/'

def sha(b):return hashlib.sha256(b).hexdigest()

def read():
    actual=sha(SOURCE.read_bytes()); env=json.loads((ROOT/'ENVIRONMENT.json').read_text())
    if actual!=env['parent_sha256']:raise RuntimeError('Pinned authentic v18.28 sendback SHA mismatch')
    with zipfile.ZipFile(SOURCE) as z:
        if z.testzip():raise RuntimeError('CRC mismatch in v18.28 sendback')
        if {n.split('/')[0] for n in z.namelist()}!={PREFIX.rstrip('/')}:raise RuntimeError('v18.28 ZIP root mismatch')
        manifest=json.loads(z.read(PREFIX+'SHA256SUMS_OUTPUT.json'))['files']
        for key,expected in manifest.items():
            if sha(z.read(PREFIX+key))!=expected:raise RuntimeError('Parent member hash mismatch '+key)
        proof=json.loads(z.read(PREFIX+'results/original_second_variational_proof_v18_28.json'))
        if proof.get('status')!='ORIGINAL_Q550_SECOND_VARIATIONAL_SOURCE_AND_NONLINEAR_PAIR_AUDIT_COMPLETED' or proof.get('new_original_ode_integrations')!=67:
            raise RuntimeError('Not the completed original Q550 v18.28 proof')
        original_csv=z.read(PREFIX+'results/original_second_variational_pairs_v18_28.csv').decode()
        nested=z.read(PREFIX+'inputs/v18_27_original_server_sendback.zip')
    scratch=ROOT/'.runtime';scratch.mkdir(exist_ok=True)
    child=scratch/'v18_27_original_server_sendback.zip';child.write_bytes(nested)
    parent.SOURCE=child
    D, provenance,_,_=parent.read()
    provenance.update({'v18_28_parent_sha256':actual,'v18_28_verified_members':len(manifest)})
    return D,provenance,proof,original_csv
