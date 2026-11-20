#!/usr/bin/env python3
"""SHA/CRC authenticated v18.46 parent; stage its v18.45 ancestor."""
from pathlib import Path
import zipfile,hashlib,json,csv,io
R=Path(__file__).resolve().parents[1]
PREFIX='q550_kick203_component_richardson_v18_46_bundle/'
def sha(b):return hashlib.sha256(b).hexdigest()
def read_parent_v47():
    env=json.loads((R/'ENVIRONMENT.json').read_text())
    blob=(R/env['v18_46_parent_file']).read_bytes()
    if sha(blob)!=env['v18_46_parent_sha256']:raise RuntimeError('v18.46 original SHA mismatch')
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        if z.testzip() is not None:raise RuntimeError('v18.46 ZIP CRC invalid')
        if {x.split('/')[0] for x in z.namelist()}!={PREFIX[:-1]}:raise RuntimeError('Unexpected v18.46 parent root')
        sums=json.loads(z.read(PREFIX+'SHA256SUMS_OUTPUT.json'))['files']
        if len(sums)!=166:raise RuntimeError('Unexpected v18.46 parent manifest size')
        for name,digest in sums.items():
            if sha(z.read(PREFIX+name))!=digest:raise RuntimeError('v18.46 member mismatch '+name)
        proof=json.loads(z.read(PREFIX+'results/original_v18_46_proof.json'))
        if not(proof.get('status')=='ORIGINAL_Q550_COMPONENT_RICHARDSON_CONSISTENCY_COMPLETED'
          and proof.get('original_integrations')==25 and proof.get('original_snapshots')==36
          and proof.get('original_ode_sha256_verified') and proof.get('source_kappa0_unchanged')
          and proof.get('no_certified_hessian') and proof.get('rhs_radius_rows')==432):
            raise RuntimeError('Original v18.46 scientific provenance malformed')
        def rows(fn):return list(csv.DictReader(io.StringIO(z.read(PREFIX+'results/'+fn).decode())))
        oldr=rows('original_v18_46_rhs_radii.csv')
        oldj=rows('original_v18_46_jac_radii.csv')
        olds=rows('original_v18_46_snapshot_summary.csv')
        assert tuple(map(len,(oldr,oldj,olds)))==(432,288,36)
        nested=z.read(PREFIX+'inputs/v18_45_original_server_sendback.zip')
        if sha(nested)!=env['v18_45_parent_sha256']:raise RuntimeError('v18.45 nested hash mismatch')
    # Recover v18.43 from the authenticated nested v18.44 original archive.
    # This avoids shipping an otherwise redundant extra ~75 MB while retaining
    # identical archived parent bytes, verified by the original SHA pin.
    with zipfile.ZipFile(io.BytesIO(nested)) as z45:
        v44=z45.read('q550_kick203_hessian_radius_error_budget_v18_45_bundle/inputs/v18_44_original_server_sendback.zip')
    if sha(v44)!=env['v18_44_parent_sha256']:raise RuntimeError('v18.44 nested digest mismatch')
    with zipfile.ZipFile(io.BytesIO(v44)) as z44:
        v43=z44.read('q550_kick203_directional_hessian_transported_jet_v18_44_bundle/inputs/v18_43_original_server_sendback.zip')
    if sha(v43)!=env['v18_43_parent_sha256']:raise RuntimeError('Recovered v18.43 digest mismatch')
    dest43=R/env['v18_43_parent_file'];dest43.parent.mkdir(parents=True,exist_ok=True)
    if not dest43.is_file() or sha(dest43.read_bytes())!=sha(v43):dest43.write_bytes(v43)
    dest=R/env['v18_45_parent_file'];dest.parent.mkdir(parents=True,exist_ok=True)
    if not dest.is_file() or sha(dest.read_bytes())!=sha(nested):dest.write_bytes(nested)
    return proof,oldr,oldj,olds,{'parent_sha256':sha(blob),'verified_members':len(sums)}
