#!/usr/bin/env python3
"""No original Q550 integrations = no sendback. Hash every output."""
from pathlib import Path
import json,csv,hashlib,zipfile
R=Path(__file__).resolve().parents[1]
pfile=R/'results/original_v18_47_proof.json'
if not pfile.is_file():raise RuntimeError('Missing ORIGINAL v18.47 Q550 run; preflight cannot create sendback')
p=json.loads(pfile.read_text())
required={'original_integrations':25,'original_cases':12,'original_snapshots':36,'kernel_vector_rows':432,'component_budget_rows':2160,'solver_rows':24}
if p.get('status')!='ORIGINAL_Q550_HESSIAN_KERNEL_COMPENSATION_COMPLETED':raise RuntimeError('Wrong original status')
for k,v in required.items():
    if p.get(k)!=v:raise RuntimeError('Original proof incomplete '+k)
for k in ('original_ode_sha256_verified','source_kappa0_unchanged','no_exact_hessian','not_full_precision_RHS','no_certified_curvature'):
    if p.get(k) is not True:raise RuntimeError('Provenance/disclaimer failed '+k)
for filename,num in [('original_v18_47_kernel_vectors.csv',432),('original_v18_47_component_budget.csv',2160),
                     ('original_v18_47_snapshot_summary.csv',36),('original_v18_47_solver_steps.csv',24)]:
    x=R/'results'/filename
    if not x.is_file() or sum(1 for _ in csv.DictReader(x.open()))!=num:raise RuntimeError('Incomplete original CSV '+filename)
for filename in ('original_v18_47_cross_kernel.png','original_v18_47_component_signature.png'):
    if not (R/'results'/filename).is_file():raise RuntimeError('Missing original figure '+filename)
man=json.loads((R/'SHA256SUMS_OUTPUT.json').read_text())['files']
for rel,d in man.items():
    x=R/rel
    if not x.is_file() or hashlib.sha256(x.read_bytes()).hexdigest()!=d:raise RuntimeError('Manifest mismatch '+rel)
out=R.parent/(R.name.replace('_bundle','_sendback')+'.zip')
with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3,allowZip64=True) as z:
    for rel in sorted(man):z.write(R/rel,R.name+'/'+rel,compress_type=zipfile.ZIP_STORED if rel.endswith('.zip') else zipfile.ZIP_DEFLATED)
    z.write(R/'SHA256SUMS_OUTPUT.json',R.name+'/SHA256SUMS_OUTPUT.json')
with zipfile.ZipFile(out) as z:
    if z.testzip() is not None or {x.split('/')[0] for x in z.namelist()}!={R.name}:raise RuntimeError('Bad sendback ZIP')
print('ORIGINAL_V18_47_SENDBACK',out)
