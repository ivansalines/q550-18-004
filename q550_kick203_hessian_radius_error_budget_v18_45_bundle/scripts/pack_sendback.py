#!/usr/bin/env python3
"""Fail closed without a verified genuine original Q550 v18.38 experiment."""
from pathlib import Path
import hashlib,json,zipfile
R=Path(__file__).resolve().parents[1];proof=R/'results/original_v18_38_proof.json'
if not proof.is_file():raise RuntimeError('Missing ORIGINAL v18.38 Q550 integrations; no sendback from preflight')
p=json.loads(proof.read_text())
if p.get('status')!='ORIGINAL_Q550_COUPLED_LEVEL_NORMAL_SCALING_COMPLETED' or p.get('original_integrations')!=61 or p.get('original_cases')!=12 or p.get('samples')!=120 or p.get('fit_rows')!=96 or not p.get('original_ode_sha256_verified') or not p.get('source_kappa0_unchanged') or not p.get('no_physical_constraint_certified'):
    raise RuntimeError('Missing or incompatible original v18.38 Q550 proof')
for name in ('original_v18_38_coupled_profiles.csv','original_v18_38_coupled_fits.csv','original_v18_38_coupled_summary.csv','original_v18_38_coupled_scaling.png'):
    if not (R/'results'/name).is_file():raise RuntimeError('Missing original Q550 output: '+name)
files=json.loads((R/'SHA256SUMS_OUTPUT.json').read_text())['files']
for name,sha in files.items():
    if not (R/name).is_file() or hashlib.sha256((R/name).read_bytes()).hexdigest()!=sha:raise RuntimeError('SHA mismatch: '+name)
output=R.parent/(R.name.replace('_bundle','_sendback')+'.zip')
with zipfile.ZipFile(output,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3,allowZip64=True) as z:
    for name in sorted(files):z.write(R/name,R.name+'/'+name,compress_type=zipfile.ZIP_STORED if name.endswith('.zip') else zipfile.ZIP_DEFLATED)
    z.write(R/'SHA256SUMS_OUTPUT.json',R.name+'/SHA256SUMS_OUTPUT.json')
with zipfile.ZipFile(output) as z:
    if z.testzip() is not None or {x.split('/')[0] for x in z.namelist()}!={R.name}:raise RuntimeError('Final ZIP invalid')
print('SEND_BACK',output)
