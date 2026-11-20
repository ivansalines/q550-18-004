#!/usr/bin/env python3
"""Frozen authentic v18.44 reanalysis. NO v18.45 original Q550 simulations."""
import sys,json,csv
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(R/'scripts'),str(R/'reference')]
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
p,h,j,f,auth=read_parent_v45()
read_parent_v44();read_parent_v43();read_parent_v42();read_parent_v41();read_parent_v40()
read_parent_v39();read_parent_v38();read_parent_v37();read_parent_v36();read_parent_v35()
cp,*_=read_parent()
if not {'standard','tight'}<=set(cp):raise RuntimeError('Missing historical checkpoint tiers')
rows=[]
for radius in (.006,.018,.036):
    r=[float(x['relative_rhs_jac_disagreement']) for x in h if float(x['rhs_normalized_h'])==radius]
    if len(r)!=108:raise RuntimeError('Incomplete v18.44 radius parent grid')
    rows.append(dict(radius=radius,comparison_count=len(r),median_relative_disagreement=float(np.median(r)),
                     maximum_relative_disagreement=max(r)))
pout=R/'results';pout.mkdir(exist_ok=True)
with (pout/'frozen_v18_44_radius_baseline.csv').open('w',newline='') as ff:
    w=csv.DictWriter(ff,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
record=dict(version='v18.45',status='PREFLIGHT_ONLY_ORIGINAL_V18_44_AUTHENTICATED',
    parent_sha256=auth['parent_sha256'],parent_verified_members=auth['verified_members'],
    historical_checkpoint_tiers=sorted(cp),parent_hessian_rows=len(h),
    parent_jet_rows=len(j),parent_case_rows=len(f),parent_radius_summaries=rows,
    no_new_original_integrations=True,no_certified_hessian=True)
(pout/'frozen_preflight_v18_45.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
print(json.dumps(record,indent=2))
