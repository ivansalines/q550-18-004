#!/usr/bin/env python3
"""Fail closed without 61 independently rerun ORIGINAL Q550 trajectories."""
from pathlib import Path
import zipfile,hashlib,json,csv
R=Path(__file__).resolve().parents[1]
proofpath=R/'results/original_v18_41_proof.json'
if not proofpath.exists():raise RuntimeError('Original v18.41 Q550 proof missing; preflight cannot pack')
p=json.loads(proofpath.read_text())
if not (p.get('status')=='ORIGINAL_Q550_ODD_SOURCE_RESPONSE_PLANE_CURVATURE_COMPLETED'
        and p.get('original_integrations')==61 and p.get('original_cases')==12
        and p.get('original_samples')==1548 and p.get('resolution_rows')==48
        and p.get('source_kappa0_unchanged') and p.get('original_ode_sha256_verified')
        and p.get('no_material_metric_certified') and p.get('no_physical_constraint_certified')):
    raise RuntimeError('Original Q550 v18.41 proof incomplete')
for file,num in [('original_v18_41_plane_profiles.csv',1548),('original_v18_41_family_summary.csv',12),
                 ('original_v18_41_curvature_resolution.csv',48),('original_v18_40_temporal_sources.csv',168)]:
    f=R/'results'/file
    if not f.exists() or sum(1 for _ in csv.DictReader(f.open()))!=num:
        raise RuntimeError('Original Q550 v18.41 output missing: '+file)
for fig in ('original_v18_41_off_plane_departure.png','original_v18_41_raw_chord_departure.png',
            'original_v18_41_direction_turn.png'):
    if not (R/'results'/fig).exists():raise RuntimeError('Missing figure '+fig)
m=json.loads((R/'SHA256SUMS_OUTPUT.json').read_text())['files']
for n,h in m.items():
    f=R/n
    if not f.exists() or hashlib.sha256(f.read_bytes()).hexdigest()!=h:raise RuntimeError('Invalid manifest: '+n)
out=R.parent/(R.name.replace('_bundle','_sendback')+'.zip')
with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3,allowZip64=True) as z:
    for n in sorted(m):z.write(R/n,R.name+'/'+n,compress_type=zipfile.ZIP_STORED if n.endswith('.zip') else zipfile.ZIP_DEFLATED)
    z.write(R/'SHA256SUMS_OUTPUT.json',R.name+'/SHA256SUMS_OUTPUT.json')
with zipfile.ZipFile(out) as z:
    if z.testzip() is not None:raise RuntimeError('v18.41 sendback CRC failed')
print('SEND_BACK',out)
