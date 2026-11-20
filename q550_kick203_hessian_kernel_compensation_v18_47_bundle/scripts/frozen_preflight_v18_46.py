#!/usr/bin/env python3
"""Authentic original v18.45 reanalysis without any new ODE integration."""
import sys,csv,json
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R/'scripts'),str(R/'reference')]
from checkpoint_io_v18_46 import read_parent_v46
from checkpoint_io_v18_45 import read_parent_v45
from checkpoint_io_v18_44 import read_parent_v44
from checkpoint_io_v18_43 import read_parent_v43
from checkpoint_io_v18_42 import read_parent_v42
from checkpoint_io_v18_41 import read_parent_v41
from checkpoint_io_v18_40 import read_parent_v40
from checkpoint_io_v18_39 import read_parent_v39
from checkpoint_io_v18_38 import read_parent_v38
from checkpoint_io_v18_37 import read_parent_v37
from checkpoint_io_v18_36 import read_parent_v36
from checkpoint_io_v18_35 import read_parent_v35
from checkpoint_io_v18_34 import read_parent
proof,rhs,jac,comps,snaps,auth=read_parent_v46()
read_parent_v45();read_parent_v44();read_parent_v43();read_parent_v42()
read_parent_v41();read_parent_v40();read_parent_v39();read_parent_v38()
read_parent_v37();read_parent_v36();read_parent_v35()
cp,*_=read_parent()
if not {'standard','tight'}<=set(cp):raise RuntimeError('Original invariant checkpoint tiers missing')
radii=sorted({float(r['rhs_normalized_h']) for r in rhs})
record=dict(version='v18.46',status='PREFLIGHT_ONLY_V18_45_AUTHENTICATED',
    parent_sha256=auth['parent_sha256'],verified_members=auth['verified_members'],
    prior_rhs_radii=radii,original_v18_45_snapshots=len(snaps),
    original_v18_45_component_rows=len(comps),checkpoint_tiers=sorted(cp),
    no_new_original_integrations=True,no_certified_hessian=True)
(R/'results/frozen_preflight_v18_46.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
