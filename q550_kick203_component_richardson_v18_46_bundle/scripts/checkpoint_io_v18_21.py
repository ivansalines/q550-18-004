"""Read authentic v18.20 server sendback and verify its SHA and complete older genealogy.
Unpacks ONLY the previous v18.19 input into .runtime for prior checkpoint loaders.
"""
from pathlib import Path
import hashlib,json,zipfile
import checkpoint_io_v18_20 as prior
ROOT=Path(__file__).resolve().parents[1]
INPUT=ROOT/'inputs/v18_20_original_server_sendback.zip'
PREFIX='q550_kick203_reduced_jet_hidden_observability_v18_20_bundle/'

def read():
    with zipfile.ZipFile(INPUT) as z:
        bad=z.testzip()
        if bad: raise RuntimeError('Source ZIP CRC mismatch: '+bad)
        roots={n.split('/')[0] for n in z.namelist()}
        if roots!={PREFIX.rstrip('/')}: raise RuntimeError('Source ZIP unexpected roots: '+str(roots))
        checks=json.loads(z.read(PREFIX+'SHA256SUMS_OUTPUT.json'))['files']
        for rel,digest in checks.items():
            if hashlib.sha256(z.read(PREFIX+rel)).hexdigest()!=digest:raise RuntimeError('Source SHA mismatch: '+rel)
        summary=json.loads(z.read(PREFIX+'results/nonlinear_jet_summary.json'))
        if summary.get('status')!='ORIGINAL_Q550_NONLINEAR_JET_VALIDATION_COMPLETED':
            raise RuntimeError('Original nonlinear v18.20 proof missing')
        predecessor=z.read(PREFIX+'inputs/v18_19_original_server_sendback.zip')
        nonlinear_csv=z.read(PREFIX+'results/nonlinear_jet_validation.csv').decode()
    runtime=ROOT/'.runtime';runtime.mkdir(exist_ok=True)
    p=runtime/'v18_19_original_server_sendback.zip';p.write_bytes(predecessor)
    prior.INPUT=p
    D,provenance,old=prior.read()
    provenance.update(dict(v18_20_server_sendback_sha256=hashlib.sha256(INPUT.read_bytes()).hexdigest(),
        v18_20_verified_immutable_members=len(checks),v18_20_status=summary['status']))
    return D,provenance,summary,nonlinear_csv
