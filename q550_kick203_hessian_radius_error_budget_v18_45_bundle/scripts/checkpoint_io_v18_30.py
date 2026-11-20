"""Authenticate genuine v18.29 and the embedded historical Q550 executable chain."""
from pathlib import Path
import hashlib,zipfile,json,io
import checkpoint_io_v18_28 as older
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'inputs/v18_29_original_server_sendback.zip'
PREFIX='q550_kick203_second_variational_compensated_v18_29_bundle/'
def sha(data):return hashlib.sha256(data).hexdigest()
def read():
    raw=SOURCE.read_bytes();env=json.loads((ROOT/'ENVIRONMENT.json').read_text())
    if sha(raw)!=env['parent_sha256']:raise RuntimeError('v18.29 pinned ZIP SHA mismatch')
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        bad=z.testzip()
        if bad:raise RuntimeError('v18.29 ZIP CRC failed: '+bad)
        if {n.split('/')[0] for n in z.namelist()}!={PREFIX.rstrip('/')}:
            raise RuntimeError('v18.29 ZIP unexpected root')
        m=json.loads(z.read(PREFIX+'SHA256SUMS_OUTPUT.json'))['files']
        for rel,expected in m.items():
            if sha(z.read(PREFIX+rel))!=expected:raise RuntimeError('v18.29 member SHA mismatch '+rel)
        proof=json.loads(z.read(PREFIX+'results/original_compensated_proof_v18_29.json'))
        if proof.get('status')!='ORIGINAL_Q550_COMPENSATED_PAIR_SECOND_VARIATIONAL_CONVERGENCE_COMPLETED' or proof.get('pair_comparisons')!=6:
            raise RuntimeError('Missing completed original-Q550 v18.29 server proof')
        prev=z.read(PREFIX+'inputs/v18_28_original_server_sendback.zip')
        csv=z.read(PREFIX+'results/original_compensated_profiles_v18_29.csv').decode()
        checks=z.read(PREFIX+'results/original_compensated_pairs_v18_29.csv').decode()
    rt=ROOT/'.runtime';rt.mkdir(exist_ok=True)
    p=rt/'v18_28_original_server_sendback.zip';p.write_bytes(prev)
    with zipfile.ZipFile(io.BytesIO(prev)) as z:
        if z.testzip():raise RuntimeError('v18.28 embedded CRC failure')
        nested=z.read('q550_kick203_second_variational_source_floor_v18_28_bundle/inputs/v18_27_original_server_sendback.zip')
    p27=rt/'v18_27_original_server_sendback.zip';p27.write_bytes(nested)
    older.SOURCE=p27
    data,provenance,proof27,floor=older.read()
    provenance.update(v18_29_original_SHA256=sha(raw),v18_29_verified_members=len(m),v18_29_proof_status=proof['status'])
    return data,provenance,proof,csv,checks
