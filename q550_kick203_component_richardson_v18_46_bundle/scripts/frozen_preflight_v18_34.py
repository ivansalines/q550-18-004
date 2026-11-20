#!/usr/bin/env python3
"""Authenticated v18.33 endpoint-only preflight; no NEW original Q550 integration."""
import csv,io,json,sys,numpy as np
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'scripts'))
from checkpoint_io_v18_34 import read_parent
_,p,d,rows,overlap,prov=read_parent()
arr=list(csv.DictReader(io.StringIO(rows)))
if len(arr)!=126 or len(list(csv.DictReader(io.StringIO(overlap))))!=84:raise RuntimeError('Original v18.33 CSV row-count mismatch')
max_identity=max(abs(float(r['C_end'])-float(r['predicted_C'])-float(r['signed_transport_residual'])) for r in arr)
if max_identity>1e-10:raise RuntimeError('v18.33 archived endpoint identity mismatch')
report=dict(version='v18.34',status='AUTHENTICATED_V18_33_ENDPOINT_ONLY_NO_NEW_Q550_RUN',
            parent_sha256=prov['parent_sha'],parent_manifest_members=prov['parent_verified_members'],
            original_v18_33_return_rows=len(arr),archive_endpoint_identity_max_abs=max_identity,
            old_certified_material_manifolds=p['numerically_passing'],new_ODE_integrations=0)
(R/'results/frozen_preflight_v18_34.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
