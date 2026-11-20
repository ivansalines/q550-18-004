"""Authenticate original v18.19 nonlinear SERVER sendback and nested genealogy."""
from pathlib import Path
import zipfile,hashlib,json
import checkpoint_io_v18_19 as prior
ROOT=Path(__file__).resolve().parents[1]
INPUT=ROOT/'inputs/v18_19_original_server_sendback.zip'
PREFIX='q550_kick203_paired_nonlinear_history_v18_19_bundle/'
def read():
    with zipfile.ZipFile(INPUT) as z:
        bad=z.testzip()
        if bad:raise RuntimeError('v18.19 ZIP CRC failure: '+bad)
        roots={name.split('/')[0] for name in z.namelist()}
        if roots!={PREFIX.rstrip('/')}:raise RuntimeError('Wrong v18.19 ZIP roots: '+str(roots))
        checks=json.loads(z.read(PREFIX+'SHA256SUMS_OUTPUT.json'))['files']
        for rel,exp in checks.items():
            if hashlib.sha256(z.read(PREFIX+rel)).hexdigest()!=exp:raise RuntimeError('v18.19 SHA mismatch '+rel)
        pair=json.loads(z.read(PREFIX+'results/nonlinear_pair_summary.json'))
        if pair['status']!='NONLINEAR_PAIRED_REDUCED_OBSERVER_DIAGNOSTIC_COMPLETED':
            raise RuntimeError('v18.19 nonlinear server proof not completed')
        predecessor=z.read(PREFIX+'inputs/v18_18_original_server_sendback.zip')
    cache=ROOT/'.runtime';cache.mkdir(exist_ok=True)
    p=cache/'v18_18_original_server_sendback.zip';p.write_bytes(predecessor)
    prior.INPUT=p
    data,provenance=prior.read()
    provenance['v18_19_original_server_sha256']=hashlib.sha256(INPUT.read_bytes()).hexdigest()
    provenance['v18_19_verified_immutable_files']=len(checks)
    provenance['v18_19_server_status']=pair['status']
    return data,provenance,pair
