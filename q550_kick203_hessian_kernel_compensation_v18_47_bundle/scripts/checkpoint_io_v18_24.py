"""Authenticate original v18.23 sendback and all nested original Q550 lineage."""
from pathlib import Path
import zipfile,json,hashlib
import checkpoint_io_v18_23 as previous
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'inputs/v18_23_original_server_sendback.zip'
PREFIX='q550_kick203_model_blind_predictive_closure_v18_23_bundle/'
def read():
    with zipfile.ZipFile(SOURCE) as z:
        if z.testzip():raise RuntimeError('Parent v18.23 ZIP CRC failure')
        if {n.split('/')[0] for n in z.namelist()}!={PREFIX.rstrip('/')}:raise RuntimeError('Incorrect v18.23 root')
        sums=json.loads(z.read(PREFIX+'SHA256SUMS_OUTPUT.json'))['files']
        for rel,digest in sums.items():
            if hashlib.sha256(z.read(PREFIX+rel)).hexdigest()!=digest:raise RuntimeError('Original v18.23 SHA mismatch '+rel)
        proof=json.loads(z.read(PREFIX+'results/original_ode_blind_summary.json'))
        if proof.get('status')!='ORIGINAL_Q550_MODEL_BLIND_NONLINEAR_TEST_COMPLETED':raise RuntimeError('Parent never nonlinear certified')
        nested=z.read(PREFIX+'inputs/v18_22_original_server_sendback.zip')
    cache=ROOT/'.runtime';cache.mkdir(exist_ok=True)
    path=cache/'v18_22_original_server_sendback.zip';path.write_bytes(nested)
    previous.SOURCE=path
    D,prov,prior=previous.read()
    prov.update(v18_23_server_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
      v18_23_verified_files=len(sums),v18_23_server_status=proof['status'])
    return D,prov,proof
