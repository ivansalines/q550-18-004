#!/usr/bin/env python3
"""Fail closed if original Q550 v18.43 has not run and been hashed."""
from pathlib import Path
import zipfile,hashlib,json,csv
R=Path(__file__).resolve().parents[1]
p=R/'results/original_v18_43_proof.json'
if not p.is_file():raise RuntimeError('NO ORIGINAL v18.43 Q550 run; preflight cannot package')
a=json.loads(p.read_text())
if not(a.get('status')=='ORIGINAL_Q550_SOURCE_JET_PRECISION_AUDIT_COMPLETED'
       and a.get('original_integrations')==49 and a.get('original_cases')==12
       and a.get('first_jet_rows')==60 and a.get('second_window_rows')==144
       and a.get('directional_step_rows')==120 and a.get('solver_step_rows')==48
       and a.get('original_ode_sha256_verified') and a.get('source_kappa0_unchanged')
       and a.get('no_certified_analytic_second_jet') and a.get('no_certified_curvature')
       and a.get('no_material_metric_certified') and a.get('no_physical_constraint_certified')):
    raise RuntimeError('Original v18.43 proof invalid/incomplete')
for fn,num in [('original_v18_43_source_first_jets.csv',60),('original_v18_43_second_jet_windows.csv',144),
               ('original_v18_43_directional_step_stability.csv',120),('original_v18_43_family_summary.csv',12),('original_v18_43_solver_step_diagnostics.csv',48)]:
    if not (R/'results'/fn).exists() or sum(1 for _ in csv.DictReader((R/'results'/fn).open()))!=num:
        raise RuntimeError('Missing original v18.43 rows: '+fn)
for fn in ('original_v18_43_precision_separation.png','original_v18_43_directional_step_floor.png'):
    if not (R/'results'/fn).exists():raise RuntimeError('Missing original v18.43 plot '+fn)
hashes=json.loads((R/'SHA256SUMS_OUTPUT.json').read_text())['files']
for fn,expected in hashes.items():
    f=R/fn
    if not f.is_file() or hashlib.sha256(f.read_bytes()).hexdigest()!=expected:
        raise RuntimeError('Manifest integrity failure: '+fn)
out=R.parent/(R.name.replace('_bundle','_sendback')+'.zip')
with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3,allowZip64=True) as z:
    for fn in sorted(hashes):z.write(R/fn,R.name+'/'+fn,compress_type=zipfile.ZIP_STORED if fn.endswith('.zip') else zipfile.ZIP_DEFLATED)
    z.write(R/'SHA256SUMS_OUTPUT.json',R.name+'/SHA256SUMS_OUTPUT.json')
with zipfile.ZipFile(out) as z:
    if z.testzip():raise RuntimeError('Packed original ZIP CRC failure')
print('ORIGINAL_V18_43_SENDBACK',out)
