"""Authenticate original nonlinear v18.24 sendback and inherited entire Q550 chain."""
from pathlib import Path
import zipfile,hashlib,json
import checkpoint_io_v18_24 as previous
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'inputs/v18_24_original_server_sendback.zip'
PREFIX='q550_kick203_endpoint_history_latent_robustness_v18_24_bundle/'

def read():
    with zipfile.ZipFile(SOURCE) as z:
        if z.testzip():raise RuntimeError('v18.24 ZIP failed CRC')
        if {n.split('/')[0] for n in z.namelist()}!={PREFIX[:-1]}:raise RuntimeError('Wrong original v18.24 root')
        sums=json.loads(z.read(PREFIX+'SHA256SUMS_OUTPUT.json'))['files']
        for rel,digest in sums.items():
            if hashlib.sha256(z.read(PREFIX+rel)).hexdigest()!=digest:raise RuntimeError('v18.24 digest mismatch '+rel)
        proof=json.loads(z.read(PREFIX+'results/original_ode_robustness_proof.json'))
        if proof.get('status')!='ORIGINAL_Q550_ENDPOINT_PRESERVED_BLIND_NONLINEAR_VALIDATION_COMPLETED':
            raise RuntimeError('Missing original v18.24 nonlinear certification')
        nested=z.read(PREFIX+'inputs/v18_23_original_server_sendback.zip')
    runtime=ROOT/'.runtime';runtime.mkdir(exist_ok=True)
    path=runtime/'v18_23_original_server_sendback.zip';path.write_bytes(nested)
    previous.SOURCE=path
    D,provenance,prevproof=previous.read()
    provenance.update(v18_24_server_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
       v18_24_verified_files=len(sums),v18_24_server_status=proof['status'])
    return D,provenance,proof
