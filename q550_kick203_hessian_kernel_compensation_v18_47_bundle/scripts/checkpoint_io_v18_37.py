"""Authenticate the ORIGINAL Q550 v18.36 server sendback and stage its nested genealogy."""
from pathlib import Path
import hashlib,json,zipfile,io,csv
R=Path(__file__).resolve().parents[1]
PREFIX='q550_kick203_flow_aligned_support_normal_transport_v18_36_bundle/'

def digest(data):return hashlib.sha256(data).hexdigest()

def read_parent_v37():
    env=json.loads((R/'ENVIRONMENT.json').read_text())
    blob=(R/env['v18_36_parent_file']).read_bytes()
    if digest(blob)!=env['v18_36_parent_sha256']:raise RuntimeError('v18.36 original parent hash mismatch')
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        if z.testzip():raise RuntimeError('Original v18.36 ZIP CRC mismatch')
        if {n.split('/')[0] for n in z.namelist()}!={PREFIX.rstrip('/')}:raise RuntimeError('v18.36 root mismatch')
        manifest=json.loads(z.read(PREFIX+'SHA256SUMS_OUTPUT.json'))['files']
        for n,h in manifest.items():
            if digest(z.read(PREFIX+n))!=h:raise RuntimeError('Parent member hash failed: '+n)
        proof=json.loads(z.read(PREFIX+'results/original_v18_36_proof.json'))
        cases=list(csv.DictReader(io.StringIO(z.read(PREFIX+'results/original_v18_36_joint_support_cases.csv').decode())))
        profiles=list(csv.DictReader(io.StringIO(z.read(PREFIX+'results/original_v18_36_normal_profiles.csv').decode())))
        attempts=list(csv.DictReader(io.StringIO(z.read(PREFIX+'results/original_v18_36_seed_attempts.csv').decode())))
        p35=z.read(PREFIX+'inputs/v18_35_original_server_sendback.zip')
    if proof.get('status')!='ORIGINAL_Q550_FLOW_ALIGNED_SUPPORT_NORMAL_TRANSPORT_COMPLETED' or proof.get('original_integrations')!=13 or proof.get('case_count')!=12 or len(profiles)!=1740:
        raise RuntimeError('Original Q550 v18.36 server results missing/incompatible')
    if digest(p35)!=env['v18_35_parent_sha256']:raise RuntimeError('Nested v18.35 parent mismatch')
    rt=R/'.runtime';rt.mkdir(exist_ok=True)
    (rt/'v18_35_original_server_sendback.zip').write_bytes(p35)
    return proof,cases,profiles,attempts,dict(parent_sha256=digest(blob),verified_members=len(manifest))
