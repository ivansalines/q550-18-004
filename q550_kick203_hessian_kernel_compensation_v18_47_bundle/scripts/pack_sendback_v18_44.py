#!/usr/bin/env python3
"""Fail-closed original Q550 v18.44 scientific sendback packager."""
from pathlib import Path
import hashlib,json,zipfile,csv
R=Path(__file__).resolve().parents[1]
p=R/'results/original_v18_44_proof.json'
if not p.is_file():raise RuntimeError('NO ORIGINAL Q550 v18.44 integrations; preflight alone cannot package')
proof=json.loads(p.read_text())
if not(proof.get('status')=='ORIGINAL_Q550_DIRECTIONAL_HESSIAN_TRANSPORTED_SOURCE_JET_COMPLETED'
    and proof.get('original_integrations')==49 and proof.get('original_cases')==12
    and proof.get('hessian_comparison_rows')==324 and proof.get('transported_jet_rows')==36
    and proof.get('solver_rows')==48 and proof.get('original_ode_sha256_verified')
    and proof.get('source_kappa0_unchanged') and proof.get('no_symbolic_hessian_claimed')
    and proof.get('no_certified_second_jet') and proof.get('no_certified_curvature')
    and proof.get('no_material_metric_certified') and proof.get('no_physical_constraint_certified')):
    raise RuntimeError('Invalid/incomplete original Q550 v18.44 scientific proof')
for name,count in [('original_v18_44_hessian_dual_contractions.csv',324),
                   ('original_v18_44_transported_source_jets.csv',36),
                   ('original_v18_44_family_summary.csv',12),
                   ('original_v18_44_solver_diagnostics.csv',48)]:
    file=R/'results'/name
    if not file.is_file() or sum(1 for _ in csv.DictReader(file.open()))!=count:
        raise RuntimeError('Incomplete original v18.44 output: '+name)
for name in ('original_v18_44_hessian_probe_comparison.png','original_v18_44_jet_separation.png','original_v18_44_source_transport_terms.png'):
    if not (R/'results'/name).is_file():raise RuntimeError('Missing ORIGINAL Q550 plot '+name)
manifest=json.loads((R/'SHA256SUMS_OUTPUT.json').read_text())['files']
for name,sha in manifest.items():
    if not (R/name).is_file() or hashlib.sha256((R/name).read_bytes()).hexdigest()!=sha:
        raise RuntimeError('Original v18.44 SHA mismatch for '+name)
out=R.parent/(R.name.replace('_bundle','_sendback')+'.zip')
with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3,allowZip64=True) as z:
    for name in sorted(manifest):
        z.write(R/name,R.name+'/'+name,compress_type=zipfile.ZIP_STORED if name.endswith('.zip') else zipfile.ZIP_DEFLATED)
    z.write(R/'SHA256SUMS_OUTPUT.json',R.name+'/SHA256SUMS_OUTPUT.json')
with zipfile.ZipFile(out) as z:
    if z.testzip() is not None or {f.split('/')[0] for f in z.namelist()}!={R.name}:
        raise RuntimeError('Original v18.44 sendback corrupted')
print('ORIGINAL_V18_44_SENDBACK',out)
