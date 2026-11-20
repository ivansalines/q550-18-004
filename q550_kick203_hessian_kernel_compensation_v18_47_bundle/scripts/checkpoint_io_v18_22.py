"""Authenticate original v18.21 server sendback and its embedded v18.20 nonlinear genealogy."""
from pathlib import Path
import zipfile,hashlib,json
import checkpoint_io_v18_21 as previous
ROOT=Path(__file__).resolve().parents[1]
INPUT=ROOT/'inputs/v18_21_original_server_sendback.zip'
PREFIX='q550_kick203_jet_noise_gauge_v18_21_bundle/'

def read():
    with zipfile.ZipFile(INPUT) as z:
        bad=z.testzip()
        if bad:raise RuntimeError('v18.21 archive CRC failed: '+bad)
        if {n.split('/')[0] for n in z.namelist()}!={PREFIX[:-1]}:
            raise RuntimeError('v18.21 source ZIP has unexpected root')
        manifest=json.loads(z.read(PREFIX+'SHA256SUMS_OUTPUT.json'))['files']
        for rel,digest in manifest.items():
            if hashlib.sha256(z.read(PREFIX+rel)).hexdigest()!=digest:
                raise RuntimeError('v18.21 SHA mismatch: '+rel)
        proof=json.loads(z.read(PREFIX+'results/original_ode_check_summary.json'))
        if proof.get('status')!='ORIGINAL_Q550_INDEPENDENT_NONLINEAR_SPOT_CHECK_COMPLETED':
            raise RuntimeError('v18.21 real ODE proof missing')
        raw=z.read(PREFIX+'inputs/v18_20_original_server_sendback.zip')
    cache=ROOT/'.runtime';cache.mkdir(exist_ok=True)
    old=cache/'v18_20_original_server_sendback.zip';old.write_bytes(raw)
    previous.INPUT=old
    data,prov,oldproof,nonlinear=previous.read()
    prov.update({'v18_21_original_server_sha256':hashlib.sha256(INPUT.read_bytes()).hexdigest(),
                 'v18_21_members_verified':len(manifest),'v18_21_original_ode_check':proof['status']})
    return data,prov,oldproof,proof
