"""Verify original v18.37 sendback, stage authentic nested original v18.36 genealogy."""
from pathlib import Path
import hashlib,json,zipfile,io,csv
R=Path(__file__).resolve().parents[1]
PREFIX='q550_kick203_residual_gradient_normal_tilt_v18_37_bundle/'
def sha(b):return hashlib.sha256(b).hexdigest()
def read_parent_v38():
    env=json.loads((R/'ENVIRONMENT.json').read_text());blob=(R/env['v18_37_parent_file']).read_bytes()
    if sha(blob)!=env['v18_37_parent_sha256']:raise RuntimeError('Original v18.37 ZIP SHA256 mismatch')
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        if z.testzip() is not None:raise RuntimeError('Parent ZIP CRC failed')
        if {n.split('/')[0] for n in z.namelist()}!={PREFIX[:-1]}:raise RuntimeError('Parent ZIP unexpected root')
        manifest=json.loads(z.read(PREFIX+'SHA256SUMS_OUTPUT.json'))['files']
        for n,h in manifest.items():
            if sha(z.read(PREFIX+n))!=h:raise RuntimeError('Original v18.37 member SHA256 mismatch: '+n)
        proof=json.loads(z.read(PREFIX+'results/original_v18_37_proof.json'))
        cases=list(csv.DictReader(io.StringIO(z.read(PREFIX+'results/original_v18_37_case_summary.csv').decode())))
        profiles=list(csv.DictReader(io.StringIO(z.read(PREFIX+'results/original_v18_37_residual_gradient_profiles.csv').decode())))
        raw36=z.read(PREFIX+'inputs/v18_36_original_server_sendback.zip')
    if proof.get('status')!='ORIGINAL_Q550_RESIDUAL_GRADIENT_NORMAL_TILT_MOVING_LEVEL_COMPLETED' or proof.get('original_cases')!=12 or len(cases)!=12 or len(profiles)!=1920:
        raise RuntimeError('Original v18.37 proof or data incomplete')
    if sha(raw36)!=env['v18_36_parent_sha256']:raise RuntimeError('Nested v18.36 genealogical SHA256 mismatch')
    rt=R/'.runtime';rt.mkdir(exist_ok=True)
    (rt/'v18_36_original_server_sendback.zip').write_bytes(raw36)
    return proof,cases,profiles,dict(parent_sha256=sha(blob),verified_members=len(manifest))
