"""Authenticate the original server v18.22 sendback and full nested Q550 lineage."""
from pathlib import Path
import zipfile, hashlib, json
import checkpoint_io_v18_22 as previous
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'inputs/v18_22_original_server_sendback.zip'
PREFIX='q550_kick203_raw_sample_jet_survival_v18_22_bundle/'

def read():
    with zipfile.ZipFile(SOURCE) as z:
        if z.testzip():raise RuntimeError('Original v18.22 zip CRC failure')
        if {n.split('/')[0] for n in z.namelist()}!={PREFIX.rstrip('/')}:raise RuntimeError('Unexpected v18.22 root')
        sums=json.loads(z.read(PREFIX+'SHA256SUMS_OUTPUT.json'))['files']
        for rel,digest in sums.items():
            if hashlib.sha256(z.read(PREFIX+rel)).hexdigest()!=digest:raise RuntimeError('v18.22 SHA256 failure: '+rel)
        proof=json.loads(z.read(PREFIX+'results/original_ode_raw_summary.json'))
        if proof.get('status')!='ORIGINAL_Q550_RAW_SAMPLE_NONLINEAR_VALIDATION_COMPLETED':raise RuntimeError('Uncertified previous server run')
        previous_data=z.read(PREFIX+'inputs/v18_21_original_server_sendback.zip')
    cache=ROOT/'.runtime';cache.mkdir(exist_ok=True)
    path=cache/'v18_21_original_server_sendback.zip';path.write_bytes(previous_data)
    previous.INPUT=path
    D,prov,_,olderproof=previous.read()
    prov.update(v18_22_server_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        v18_22_verified_files=len(sums),v18_22_server_status=proof['status'])
    return D,prov,proof
