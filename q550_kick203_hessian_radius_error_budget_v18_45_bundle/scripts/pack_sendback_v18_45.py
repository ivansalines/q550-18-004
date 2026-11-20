#!/usr/bin/env python3
"""Fail-closed original server Q550 v18.45 packaging."""
import json,csv,hashlib,zipfile
from pathlib import Path
R=Path(__file__).resolve().parents[1]
p=R/'results/original_v18_45_proof.json'
if not p.exists():raise RuntimeError('NO ORIGINAL v18.45 Q550 run; preflight cannot be packaged')
a=json.loads(p.read_text())
if not(a.get('status')=='ORIGINAL_Q550_HESSIAN_RADIUS_ERROR_BUDGET_COMPLETED'
    and a.get('original_integrations')==25 and a.get('original_snapshots')==36
    and a.get('rhs_radius_rows')==396 and a.get('jac_target_rows')==216
    and a.get('component_budget_rows')==360 and a.get('solver_rows')==24
    and a.get('original_ode_sha256_verified') and a.get('source_kappa0_unchanged')
    and a.get('no_symbolic_hessian_claimed') and a.get('no_certified_curvature')
    and a.get('no_material_metric_certified')):raise RuntimeError('Incomplete v18.45 original proof')
for filename,expected in [('original_v18_45_rhs_radius_scan.csv',396),('original_v18_45_jac_target_scan.csv',216),
                          ('original_v18_45_component_error_budget.csv',360),('original_v18_45_snapshot_summary.csv',36),
                          ('original_v18_45_solver_steps.csv',24)]:
    p=R/'results'/filename
    if not p.exists() or sum(1 for _ in csv.DictReader(p.open()))!=expected:raise RuntimeError('Missing original rows: '+filename)
for filename in ('original_v18_45_radius_convergence.png','original_v18_45_component_error_budget.png',
                 'original_v18_45_crosspath_error_budget.png'):
    if not (R/'results'/filename).is_file():raise RuntimeError('Missing original graph: '+filename)
manifest=json.loads((R/'SHA256SUMS_OUTPUT.json').read_text())['files']
for filename,expect in manifest.items():
    if not (R/filename).is_file() or hashlib.sha256((R/filename).read_bytes()).hexdigest()!=expect:
        raise RuntimeError('Manifest mismatch: '+filename)
out=R.parent/(R.name.replace('_bundle','_sendback')+'.zip')
with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3,allowZip64=True) as z:
    for filename in sorted(manifest):
        z.write(R/filename,R.name+'/'+filename,compress_type=zipfile.ZIP_STORED if filename.endswith('.zip') else zipfile.ZIP_DEFLATED)
    z.write(R/'SHA256SUMS_OUTPUT.json',R.name+'/SHA256SUMS_OUTPUT.json')
with zipfile.ZipFile(out) as z:
    if z.testzip() is not None or {n.split('/')[0] for n in z.namelist()}!={R.name}:
        raise RuntimeError('Sendback ZIP corrupted')
print('ORIGINAL_V18_45_SENDBACK',out)
