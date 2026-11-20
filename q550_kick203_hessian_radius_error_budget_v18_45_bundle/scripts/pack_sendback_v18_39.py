#!/usr/bin/env python3
"""Fail-closed; includes code, frozen authentic inputs and original outputs."""
from pathlib import Path
import hashlib,json,zipfile,csv
R=Path(__file__).resolve().parents[1]
p=R/'results/original_v18_39_proof.json'
if not p.is_file():raise RuntimeError('No ORIGINAL Q550 v18.39 proof; preflight cannot make a sendback')
pr=json.loads(p.read_text())
if pr.get('status')!='ORIGINAL_Q550_FULL_TANGENT_SOURCE_ROTATION_COMPLETED' or pr.get('original_integrations')!=61 or pr.get('original_cases')!=12 or pr.get('samples')!=120 or pr.get('vector_samples')!=120 or pr.get('vector_fit_rows')!=48 or pr.get('fit_rows')!=96 or not pr.get('source_kappa0_unchanged') or not pr.get('original_ode_sha256_verified') or not pr.get('no_physical_constraint_certified'):
    raise RuntimeError('v18.39 original proof incomplete or inconsistent')
needed={'original_v18_39_scalar_replay.csv':120,'original_v18_39_vector_source_9d.csv':120,'original_v18_39_vector_fits.csv':48,'original_v18_39_family_summary.csv':24,'original_v18_39_frame_transport.csv':12}
for filename,count in needed.items():
    file=R/'results'/filename
    if not file.is_file() or sum(1 for _ in csv.DictReader(file.open()))!=count:raise RuntimeError('Missing/wrong original Q550 output: '+filename)
for name in ('original_v18_39_full_vs_scalar_holdout.png','original_v18_39_full_source_rotation.png','original_v18_39_observer_capture.png'):
    if not (R/'results'/name).is_file():raise RuntimeError('Missing genuine original figure: '+name)
manifest=json.loads((R/'SHA256SUMS_OUTPUT.json').read_text())['files']
for fn,h in manifest.items():
    if not (R/fn).is_file() or hashlib.sha256((R/fn).read_bytes()).hexdigest()!=h:raise RuntimeError('Member hash failure: '+fn)
out=R.parent/(R.name.replace('_bundle','_sendback')+'.zip')
with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3,allowZip64=True) as z:
    for name in sorted(manifest):z.write(R/name,R.name+'/'+name,compress_type=zipfile.ZIP_STORED if name.endswith('.zip') else zipfile.ZIP_DEFLATED)
    z.write(R/'SHA256SUMS_OUTPUT.json',R.name+'/SHA256SUMS_OUTPUT.json')
with zipfile.ZipFile(out) as z:
    if z.testzip() is not None:raise RuntimeError('Sendback CRC failure')
print('SEND_BACK',out)
