#!/usr/bin/env python3
"""Authenticate v18.41 original Q550 parent + all historical checkpoints only."""
from pathlib import Path
import json,sys
R=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(R/'scripts'),str(R/'reference')]
from checkpoint_io_v18_42 import read_parent_v42
from checkpoint_io_v18_41 import read_parent_v41
from checkpoint_io_v18_40 import read_parent_v40
from checkpoint_io_v18_39 import read_parent_v39
from checkpoint_io_v18_38 import read_parent_v38
from checkpoint_io_v18_37 import read_parent_v37
from checkpoint_io_v18_36 import read_parent_v36
from checkpoint_io_v18_35 import read_parent_v35
from checkpoint_io_v18_34 import read_parent
p,fa,pr,re,auth=read_parent_v42()
read_parent_v41();read_parent_v40();read_parent_v39();read_parent_v38()
read_parent_v37();read_parent_v36();read_parent_v35()
cp,*_=read_parent()
if not {'standard','tight'}<=set(cp):raise RuntimeError('Original standard/tight checkpoint tiers missing')
record=dict(version='v18.42',status='PREFLIGHT_ONLY_V18_41_PARENT_AUTHENTICATED',
    parent_sha256=auth['parent_sha256'],verified_members=auth['verified_members'],
    original_parent_integrations=p['original_integrations'],parent_families=len(fa),
    parent_profiles=len(pr),parent_resolution_rows=len(re),checkpoint_tiers=sorted(cp),
    no_new_original_integrations=True)
(R/'results').mkdir(exist_ok=True)
(R/'results/frozen_preflight_v18_42.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
