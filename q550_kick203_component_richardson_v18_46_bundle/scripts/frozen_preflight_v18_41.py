#!/usr/bin/env python3
"""Authenticate v18.40 + historical checkpoints. Never integrate new Q550 ODE."""
from pathlib import Path
import sys,json
R=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(R/'scripts'),str(R/'reference')]
from checkpoint_io_v18_41 import read_parent_v41
from checkpoint_io_v18_40 import read_parent_v40
from checkpoint_io_v18_39 import read_parent_v39
from checkpoint_io_v18_38 import read_parent_v38
from checkpoint_io_v18_37 import read_parent_v37
from checkpoint_io_v18_36 import read_parent_v36
from checkpoint_io_v18_35 import read_parent_v35
from checkpoint_io_v18_34 import read_parent
p40,t40,f40,r40,a=read_parent_v41()
read_parent_v40();read_parent_v39();read_parent_v38();read_parent_v37();read_parent_v36();read_parent_v35()
cp,*_=read_parent()
if not {'standard','tight'}<=set(cp):raise RuntimeError('Original checkpoint tiers missing')
report=dict(version='v18.41',status='PREFLIGHT_ONLY_V18_40_PARENT_AUTHENTICATED',
    parent_sha256=a['parent_sha256'],verified_members=a['verified_members'],
    parent_original_integrations=p40['original_integrations'],parent_temporal_samples=len(t40),
    parent_families=len(f40),parent_resolution_rows=len(r40),
    checkpoint_tiers=sorted(cp),no_new_original_integrations=True)
(R/'results').mkdir(exist_ok=True)
(R/'results/frozen_preflight_v18_41.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
