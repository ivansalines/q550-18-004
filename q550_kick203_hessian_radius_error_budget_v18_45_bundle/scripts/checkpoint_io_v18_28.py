"""Authenticate v18.27 original nonlinear Q550 run and its nested v18.26 input."""
import hashlib,json,zipfile
from pathlib import Path
import checkpoint_io_v18_26
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'inputs/v18_27_original_server_sendback.zip'
PREFIX='q550_kick203_frozen_latent_singular_floor_v18_27_bundle/'
OLDER='q550_kick203_vanishing_amplitude_latent_transport_v18_26_bundle/'

def sha(data):return hashlib.sha256(data).hexdigest()

def read():
    with zipfile.ZipFile(SOURCE) as z:
        bad=z.testzip()
        if bad:raise RuntimeError('Parent v18.27 CRC mismatch: '+bad)
        if {n.split('/')[0] for n in z.namelist()}!={PREFIX.rstrip('/')}:raise RuntimeError('Parent root mismatch')
        manifest=json.loads(z.read(PREFIX+'SHA256SUMS_OUTPUT.json'))['files']
        for rel,exp in manifest.items():
            if sha(z.read(PREFIX+rel))!=exp:raise RuntimeError('Parent SHA mismatch: '+rel)
        proof=json.loads(z.read(PREFIX+'results/original_high_precision_floor_proof_v18_27.json'))
        if proof.get('status')!='ORIGINAL_Q550_HIGH_PRECISION_NUMERICAL_FLOOR_SCAN_COMPLETED' or proof.get('rows')!=45:
            raise RuntimeError('The genuine nonlinear v18.27 proof was not found')
        nested=z.read(PREFIX+'inputs/v18_26_original_server_sendback.zip')
        floor=z.read(PREFIX+'results/original_tight_floor_comparison_v18_27.csv').decode()
    runtime=ROOT/'.runtime';runtime.mkdir(exist_ok=True)
    inp=runtime/'v18_26_original_server_sendback.zip';inp.write_bytes(nested)
    with zipfile.ZipFile(inp) as z:
        if z.testzip():raise RuntimeError('Embedded v18.26 corrupt')
        manifest26=json.loads(z.read(OLDER+'SHA256SUMS_OUTPUT.json'))['files']
        for rel,exp in manifest26.items():
            if sha(z.read(OLDER+rel))!=exp:raise RuntimeError('v18.26 SHA mismatch: '+rel)
        nested25=z.read(OLDER+'inputs/v18_25_original_server_sendback.zip')
    oldpath=runtime/'v18_25_original_server_sendback.zip';oldpath.write_bytes(nested25)
    checkpoint_io_v18_26.SOURCE=oldpath
    data,provenance,oldproof=checkpoint_io_v18_26.read()
    provenance.update({'v18_27_sendback_sha256':sha(SOURCE.read_bytes()),
      'v18_27_immutable_files_verified':len(manifest),
      'v18_27_status':proof['status']})
    return data,provenance,proof,floor
