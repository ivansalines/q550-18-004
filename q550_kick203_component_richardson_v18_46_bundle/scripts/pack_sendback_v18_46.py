#!/usr/bin/env python3
"""Fail-closed original Q550 v18.46 packaging, SHA authenticated genealogy."""
import csv,hashlib,json,zipfile
from pathlib import Path
R=Path(__file__).resolve().parents[1]
proof=R/'results/original_v18_46_proof.json'
if not proof.is_file():raise RuntimeError('Missing ORIGINAL v18.46 Q550 run; preflight cannot be packaged')
p=json.loads(proof.read_text())
expected={'original_integrations':25,'original_cases':12,'original_snapshots':36,
    'rhs_radius_rows':432,'jac_radius_rows':288,'rhs_ladder_rows':288,
    'jac_ladder_rows':216,'component_ladder_rows':2880,'solver_rows':24}
if p.get('status')!='ORIGINAL_Q550_COMPONENT_RICHARDSON_CONSISTENCY_COMPLETED':raise RuntimeError('Wrong original v18.46 status')
for key,num in expected.items():
    if p.get(key)!=num:raise RuntimeError('Incomplete original v18.46 proof: '+key)
for key in ('original_ode_sha256_verified','source_kappa0_unchanged',
    'no_symbolic_hessian_claimed','no_certified_hessian','no_certified_curvature'):
    if not p.get(key):raise RuntimeError('Missing provenance/disclaimer flag: '+key)
for name,num in [('original_v18_46_snapshot_summary.csv',36),('original_v18_46_rhs_radii.csv',432),
    ('original_v18_46_jac_radii.csv',288),('original_v18_46_rhs_richardson_ladders.csv',288),
    ('original_v18_46_jac_richardson_ladders.csv',216),('original_v18_46_component_ladders.csv',2880),
    ('original_v18_46_solver_steps.csv',24)]:
    dst=R/'results'/name
    if not dst.is_file() or sum(1 for _ in csv.DictReader(dst.open()))!=num:
        raise RuntimeError('Missing original Q550 results: '+name)
for name in ('original_v18_46_richardson_order.png',
    'original_v18_46_richardson_reference_sensitivity.png',
    'original_v18_46_component_signed_budget.png',
    'original_v18_46_jacobian_consistency.png'):
    if not (R/'results'/name).is_file():raise RuntimeError('Missing original plot: '+name)
man=json.loads((R/'SHA256SUMS_OUTPUT.json').read_text())['files']
for rel,digest in man.items():
    path=R/rel
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
        raise RuntimeError('SHA256 manifest mismatch: '+rel)
out=R.parent/(R.name.replace('_bundle','_sendback')+'.zip')
with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3,allowZip64=True) as z:
    for rel in sorted(man):
        z.write(R/rel,R.name+'/'+rel,compress_type=zipfile.ZIP_STORED if rel.endswith('.zip') else zipfile.ZIP_DEFLATED)
    z.write(R/'SHA256SUMS_OUTPUT.json',R.name+'/SHA256SUMS_OUTPUT.json')
with zipfile.ZipFile(out) as z:
    if z.testzip() is not None or {n.split('/')[0] for n in z.namelist()}!={R.name}:
        raise RuntimeError('Invalid packaged archive')
print('ORIGINAL_V18_46_SENDBACK',out)
