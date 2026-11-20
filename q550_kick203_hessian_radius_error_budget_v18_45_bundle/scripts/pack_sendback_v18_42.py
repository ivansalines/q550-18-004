#!/usr/bin/env python3
"""Fail closed without fresh ORIGINAL Q550 v18.42 integration and results."""
from pathlib import Path
import zipfile,hashlib,json,csv
R=Path(__file__).resolve().parents[1]
p=R/'results/original_v18_42_proof.json'
if not p.exists():raise RuntimeError('NO ORIGINAL v18.42 Q550 proof: preflight cannot package a sendback')
r=json.loads(p.read_text())
if not(r.get('status')=='ORIGINAL_Q550_SECOND_DERIVATIVE_NOISE_WINDOW_SCALING_COMPLETED'
       and r.get('original_integrations')==73 and r.get('baseline_original_integrations')==61
       and r.get('tolerance_original_integrations')==12 and r.get('original_cases')==12
       and r.get('vector_temporal_samples')==1548 and r.get('window_rows')==48
       and r.get('second_derivative_rows')==144 and r.get('tolerance_comparison_rows')==16
       and r.get('source_kappa0_unchanged') and r.get('original_ode_sha256_verified')
       and r.get('no_analytic_second_variation_claim') and r.get('no_certified_curvature')
       and r.get('no_material_metric_certified') and r.get('no_physical_constraint_certified')):
    raise RuntimeError('ORIGINAL v18.42 proof invalid')
for fn,rows in [('original_v18_42_odd_vector_profiles.csv',1548),('original_v18_42_window_scaling.csv',48),
                ('original_v18_42_second_derivative_stencils.csv',144),('original_v18_42_family_summary.csv',12),
                ('original_v18_42_tolerance_noise.csv',16),('original_v18_41_plane_profiles.csv',1548)]:
    f=R/'results'/fn
    if not f.exists() or sum(1 for _ in csv.DictReader(f.open()))!=rows:
        raise RuntimeError('Missing ORIGINAL v18.42 CSV: '+fn)
for fn in ('original_v18_42_window_scaling.png','original_v18_42_second_derivative_stencil_disagreement.png',
           'original_v18_42_solver_tolerance_floor.png'):
    if not (R/'results'/fn).exists():raise RuntimeError('Missing original diagnostic figure: '+fn)
hashes=json.loads((R/'SHA256SUMS_OUTPUT.json').read_text())['files']
for fn,digest in hashes.items():
    file=R/fn
    if not file.exists() or hashlib.sha256(file.read_bytes()).hexdigest()!=digest:
        raise RuntimeError('Failed SHA256 manifest entry '+fn)
out=R.parent/(R.name.replace('_bundle','_sendback')+'.zip')
with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3,allowZip64=True) as z:
    for fn in sorted(hashes):z.write(R/fn,R.name+'/'+fn,compress_type=zipfile.ZIP_STORED if fn.endswith('.zip') else zipfile.ZIP_DEFLATED)
    z.write(R/'SHA256SUMS_OUTPUT.json',R.name+'/SHA256SUMS_OUTPUT.json')
with zipfile.ZipFile(out) as z:
    if z.testzip() is not None:raise RuntimeError('Original Q550 v18.42 ZIP CRC mismatch')
print('SEND_BACK',out,flush=True)
