#!/usr/bin/env python3
"""Validate v18.35 original, no new Q550 integration is performed."""
import json,csv,io,sys
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'scripts'))
from checkpoint_io_v18_36 import read_parent_v36
p,data,meta=read_parent_v36()
rows=list(csv.DictReader(io.StringIO(data)))
if len(rows)!=12:raise RuntimeError('Parent did not contain exactly 12 archived cases')
inside=sum(r['inside_joint_calibrated_region_initial']=='True' for r in rows)
if inside!=4:raise RuntimeError('Parent v18.35 in-domain count changed')
report=dict(version='v18.36',status='PREFLIGHT_AUTHENTICATED_NO_NEW_Q550_RUN',parent_sha256=meta['parent_sha256'],parent_verified_members=meta['verified_members'],parent_inside_joint_support=inside,parent_case_count=len(rows),no_new_original_ODE=True,no_material_manifold_certified=True)
(R/'results/frozen_preflight_v18_36.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
