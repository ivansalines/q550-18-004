"""Authenticate original v18.25 sendback before unpacking its v18.24 ancestor.

All Q550 provenance is read from original server archives. No guesswork or
replacement of the pinned source scripts is permitted.
"""
from pathlib import Path
import hashlib,json,zipfile
import checkpoint_io_v18_25 as parent
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'inputs/v18_25_original_server_sendback.zip'
PREFIX='q550_kick203_delay_frame_cross_posture_v18_25_bundle/'

def read():
    with zipfile.ZipFile(SOURCE) as z:
        if z.testzip():raise RuntimeError('v18.25 CRC failure')
        if {name.split('/')[0] for name in z.namelist()} != {PREFIX[:-1]}:
            raise RuntimeError('Wrong v18.25 archive root')
        manifest=json.loads(z.read(PREFIX+'SHA256SUMS_OUTPUT.json'))['files']
        for rel,sha in manifest.items():
            if hashlib.sha256(z.read(PREFIX+rel)).hexdigest()!=sha:
                raise RuntimeError('v18.25 SHA256 mismatch '+rel)
        proof=json.loads(z.read(PREFIX+'results/original_ode_transport_proof.json'))
        if proof.get('status')!='ORIGINAL_Q550_NONLINEAR_PAIRED_DELAY_FRAME_TRANSPORT_COMPLETED':
            raise RuntimeError('Missing original nonlinear v18.25 proof')
        nested=z.read(PREFIX+'inputs/v18_24_original_server_sendback.zip')
    cache=ROOT/'.runtime';cache.mkdir(exist_ok=True)
    nested_path=cache/'v18_24_original_server_sendback.zip'
    nested_path.write_bytes(nested)
    parent.SOURCE=nested_path
    data, provenance, previous=parent.read()
    provenance.update(v18_25_server_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                      v18_25_verified_files=len(manifest),v18_25_original_status=proof['status'])
    return data,provenance,proof
