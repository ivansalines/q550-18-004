#!/usr/bin/env python3
"""Read full SHA-authenticated ORIGINAL v18.45 and stage original v18.44 ancestor.
Never treat frozen-only data as original Q550 results.
"""
import hashlib,json,zipfile,io,csv
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PREFIX='q550_kick203_hessian_radius_error_budget_v18_45_bundle/'
def sha(data):return hashlib.sha256(data).hexdigest()
def read_parent_v46():
    env=json.loads((ROOT/'ENVIRONMENT.json').read_text())
    blob=(ROOT/env['v18_45_parent_file']).read_bytes()
    if sha(blob)!=env['v18_45_parent_sha256']:raise RuntimeError('v18.45 parent archive digest mismatch')
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        if z.testzip() is not None:raise RuntimeError('v18.45 parent CRC failure')
        if {x.split('/')[0] for x in z.namelist()}!={PREFIX[:-1]}:raise RuntimeError('Wrong parent ZIP root')
        sums=json.loads(z.read(PREFIX+'SHA256SUMS_OUTPUT.json'))['files']
        if len(sums)!=156:raise RuntimeError('v18.45 manifest cardinality mismatch')
        for path,digest in sums.items():
            if sha(z.read(PREFIX+path))!=digest:raise RuntimeError('v18.45 member SHA mismatch: '+path)
        proof=json.loads(z.read(PREFIX+'results/original_v18_45_proof.json'))
        if not (proof.get('status')=='ORIGINAL_Q550_HESSIAN_RADIUS_ERROR_BUDGET_COMPLETED'
           and proof.get('original_integrations')==25 and proof.get('original_snapshots')==36
           and proof.get('rhs_radius_rows')==396 and proof.get('jac_target_rows')==216
           and proof.get('component_budget_rows')==360 and proof.get('original_ode_sha256_verified')
           and proof.get('source_kappa0_unchanged') and proof.get('no_certified_curvature')):
            raise RuntimeError('Original v18.45 scientific proof missing or malformed')
        def rows(fn):return list(csv.DictReader(io.StringIO(z.read(PREFIX+'results/'+fn).decode())))
        rhs=rows('original_v18_45_rhs_radius_scan.csv')
        jac=rows('original_v18_45_jac_target_scan.csv')
        comps=rows('original_v18_45_component_error_budget.csv')
        snaps=rows('original_v18_45_snapshot_summary.csv')
        if tuple(map(len,(rhs,jac,comps,snaps)))!=(396,216,360,36):raise RuntimeError('Original v18.45 CSV counts inconsistent')
        nested=z.read(PREFIX+'inputs/v18_44_original_server_sendback.zip')
        if sha(nested)!=env['v18_44_parent_sha256']:raise RuntimeError('Nested original v18.44 hash mismatch')
    dest=ROOT/env['v18_44_parent_file'];dest.parent.mkdir(exist_ok=True,parents=True)
    if not dest.exists() or sha(dest.read_bytes())!=sha(nested):dest.write_bytes(nested)
    return proof,rhs,jac,comps,snaps,dict(parent_sha256=sha(blob),verified_members=len(sums))
