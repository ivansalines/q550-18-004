#!/usr/bin/env python3
"""Authenticate v18.46 ORIGINAL archive and full checkpoint ancestry. No new ODE."""
from pathlib import Path
import sys,json,statistics
R=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(R/'scripts'),str(R/'reference')]
from checkpoint_io_v18_47 import read_parent_v47
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
p,oldr,oldj,oldsn,auth=read_parent_v47()
read_parent_v46();read_parent_v45();read_parent_v44();read_parent_v43()
read_parent_v42();read_parent_v41();read_parent_v40();read_parent_v39()
read_parent_v38();read_parent_v37();read_parent_v36();read_parent_v35()
c,*_=read_parent()
if set(c)<{'standard','tight'}:raise RuntimeError('Original checkpoint tiers missing')
rr=[r for r in oldr if float(r['rhs_h'])==.008]
record=dict(version='v18.47',status='PREFLIGHT_ONLY_PARENT_V18_46_AUTHENTICATED',
    parent_sha256=auth['parent_sha256'],verified_members=auth['verified_members'],
    parent_original_integrations=p['original_integrations'],parent_snapshots=len(oldsn),
    parent_fixed_r=.008,parent_median_jac_disagreement_at_fixed_r=float(statistics.median(float(v['relative_to_jac_parent']) for v in rr)),
    checkpoint_tiers=sorted(c),no_new_original_integrations=True)
(R/'results/frozen_preflight_v18_47.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
