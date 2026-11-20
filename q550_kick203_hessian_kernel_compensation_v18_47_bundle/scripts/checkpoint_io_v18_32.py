"""Authenticate original v18.31 sendback and extract immutable v18.30 ancestor.

No regenerated trajectories or reconstructed prior scripts are used as inputs.
"""
from pathlib import Path
import zipfile,hashlib,json,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from checkpoint_io_v18_31 import read as load_ancestor
PREFIX='q550_kick203_material_invariant_constraint_curvature_v18_31_bundle/'
def sha(b):return hashlib.sha256(b).hexdigest()
def read():
    env=json.loads((ROOT/'ENVIRONMENT.json').read_text())
    data=(ROOT/env['parent_file']).read_bytes()
    if sha(data)!=env['parent_sha256']:raise RuntimeError('v18.31 original SHA mismatch')
    with zipfile.ZipFile(ROOT/env['parent_file']) as z:
        if z.testzip():raise RuntimeError('v18.31 CRC failure')
        if {p.split('/')[0] for p in z.namelist()}!={PREFIX.rstrip('/')}:raise RuntimeError('v18.31 root mismatch')
        manifest=json.loads(z.read(PREFIX+'SHA256SUMS_OUTPUT.json'))['files']
        for p,s in manifest.items():
            if sha(z.read(PREFIX+p))!=s:raise RuntimeError('v18.31 member SHA mismatch '+p)
        report=json.loads(z.read(PREFIX+'results/original_invariant_discovery_proof_v18_31.json'))
        if (report.get('status')!='ORIGINAL_Q550_MATERIAL_INVARIANT_DISCOVERY_AND_CONDITIONAL_CURVATURE_COMPLETED'
            or report.get('material_constraint_status')!='NO_PHYSICAL_MATERIAL_INVARIANT_CERTIFIED'
            or not report.get('pinned_original_ODE_SHA256_verified')):
            raise RuntimeError('v18.31 server certificate/status mismatch')
        old=z.read(PREFIX+'inputs/v18_30_original_server_sendback.zip')
    if sha(old)!=env['grandparent_v18_30_sha256']:
        raise RuntimeError('v18.30 grandparent SHA mismatch')
    rt=ROOT/'.runtime';rt.mkdir(exist_ok=True)
    (rt/'v18_30_original_server_sendback.zip').write_bytes(old)
    checkpoint,provenance,prior,*_ = load_ancestor()
    provenance.update(v18_31_server_SHA256=env['parent_sha256'],v18_31_verified_members=len(manifest),v18_31_report=report['status'])
    return checkpoint,provenance,report
