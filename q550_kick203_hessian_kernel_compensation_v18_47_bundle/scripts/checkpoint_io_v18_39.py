#!/usr/bin/env python3
"""SHA-pinned authentic v18.38 original server parent, including both executed hotfixes."""
from pathlib import Path
import zipfile,io,json,csv,hashlib
R=Path(__file__).resolve().parents[1]
PREFIX='q550_kick203_coupled_level_normal_scaling_v18_38_bundle/'
def sha(b):return hashlib.sha256(b).hexdigest()
def read_parent_v39():
    env=json.loads((R/'ENVIRONMENT.json').read_text())
    blob=(R/env['v18_38_parent_file']).read_bytes()
    if sha(blob)!=env['v18_38_parent_sha256']:raise RuntimeError('Original v18.38 parent ZIP hash mismatch')
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        if z.testzip() is not None:raise RuntimeError('Original v18.38 ZIP CRC failure')
        if {n.split('/')[0] for n in z.namelist()}!={PREFIX[:-1]}:raise RuntimeError('Unrecognized original v18.38 root')
        manifest=json.loads(z.read(PREFIX+'SHA256SUMS_OUTPUT.json'))['files']
        if 'HOTFIX_v18_38_2.md' not in manifest or 'scripts/support_censor_v18_38.py' not in manifest:raise RuntimeError('Original v18.38.2 hotfix provenance missing')
        for name,h in manifest.items():
            if sha(z.read(PREFIX+name))!=h:raise RuntimeError('Original v18.38 member SHA256 mismatch: '+name)
        proof=json.loads(z.read(PREFIX+'results/original_v18_38_proof.json'))
        summaries=list(csv.DictReader(io.StringIO(z.read(PREFIX+'results/original_v18_38_coupled_summary.csv').decode())))
        profiles=list(csv.DictReader(io.StringIO(z.read(PREFIX+'results/original_v18_38_coupled_profiles.csv').decode())))
        fits=list(csv.DictReader(io.StringIO(z.read(PREFIX+'results/original_v18_38_coupled_fits.csv').decode())))
        p37=z.read(PREFIX+'inputs/v18_37_original_server_sendback.zip')
    if proof.get('status')!='ORIGINAL_Q550_COUPLED_LEVEL_NORMAL_SCALING_COMPLETED' or proof.get('original_integrations')!=61 or proof.get('original_cases')!=12 or len(profiles)!=120 or len(fits)!=96 or len(summaries)!=24:
        raise RuntimeError('Original v18.38 Q550 proof/data are incomplete')
    if not proof.get('source_kappa0_unchanged') or not proof.get('original_ode_sha256_verified') or not proof.get('no_physical_constraint_certified'):
        raise RuntimeError('Original parent physical provenance flags failed')
    if sha(p37)!=env['v18_37_parent_sha256']:raise RuntimeError('Nested authentic v18.37 SHA mismatch')
    rt=R/'.runtime';rt.mkdir(exist_ok=True)
    (rt/'v18_37_original_server_sendback.zip').write_bytes(p37)
    return proof,summaries,profiles,fits,dict(parent_sha256=sha(blob),verified_members=len(manifest))
