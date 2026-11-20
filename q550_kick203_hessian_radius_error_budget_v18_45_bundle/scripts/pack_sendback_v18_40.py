#!/usr/bin/env python3
"""Fail-closed sendback containing original Q550 outputs and the runnable ancestry."""
from pathlib import Path
import hashlib,json,zipfile,csv
R=Path(__file__).resolve().parents[1]
p=R/'results/original_v18_40_proof.json'
if not p.is_file():raise RuntimeError('No ORIGINAL Q550 v18.40 proof; preflight cannot make a sendback')
pr=json.loads(p.read_text())
if (pr.get('status')!='ORIGINAL_Q550_GAUGE_ROBUST_SOURCE_TEMPORAL_COMPLETED' or
    pr.get('original_integrations')!=61 or pr.get('original_cases')!=12 or
    pr.get('temporal_samples')!=168 or pr.get('gauge_samples')!=168 or
    pr.get('resolution_rows')!=48 or not pr.get('source_kappa0_unchanged') or
    not pr.get('original_ode_sha256_verified') or not pr.get('no_physical_constraint_certified')):
    raise RuntimeError('v18.40 original Q550 proof incomplete')
needed={'original_v18_40_temporal_sources.csv':168,'original_v18_40_gauge_comparison.csv':168,
        'original_v18_40_family_summary.csv':12,'original_v18_40_resolution.csv':48}
for file,count in needed.items():
    p=R/'results'/file
    if not p.is_file() or sum(1 for _ in csv.DictReader(p.open()))!=count:raise RuntimeError('Missing/wrong Q550 original output: '+file)
for f in ('original_v18_40_temporal_rotation.png','original_v18_40_transport_comparison.png',
          'original_v18_40_temporal_resolution.png'):
    if not (R/'results'/f).is_file():raise RuntimeError('Original Q550 figure missing: '+f)
m=json.loads((R/'SHA256SUMS_OUTPUT.json').read_text())['files']
for name,h in m.items():
    if not (R/name).is_file() or hashlib.sha256((R/name).read_bytes()).hexdigest()!=h:raise RuntimeError('Manifest member invalid: '+name)
out=R.parent/(R.name.replace('_bundle','_sendback')+'.zip')
with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3,allowZip64=True) as z:
    for n in sorted(m):z.write(R/n,R.name+'/'+n,compress_type=zipfile.ZIP_STORED if n.endswith('.zip') else zipfile.ZIP_DEFLATED)
    z.write(R/'SHA256SUMS_OUTPUT.json',R.name+'/SHA256SUMS_OUTPUT.json')
with zipfile.ZipFile(out) as z:
    if z.testzip() is not None:raise RuntimeError('Original v18.40 sendback ZIP CRC failed')
print('SEND_BACK',out)
