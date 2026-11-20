#!/usr/bin/env python3
"""Historical SHA-verified inputs only; no new original Q550 integrations."""
from pathlib import Path
import json,sys
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R/'scripts'),str(R/'reference')]
from checkpoint_io_v18_40 import read_parent_v40
from checkpoint_io_v18_39 import read_parent_v39
from checkpoint_io_v18_38 import read_parent_v38
from checkpoint_io_v18_37 import read_parent_v37
from checkpoint_io_v18_36 import read_parent_v36
from checkpoint_io_v18_35 import read_parent_v35
from checkpoint_io_v18_34 import read_parent
p39,frame,vec,auth40=read_parent_v40()
p38,*_=read_parent_v39()
p37,*_=read_parent_v38()
p36,*_=read_parent_v37()
read_parent_v36();read_parent_v35()
checkpoints,_,model,_,_,_=read_parent()
if not {'standard','tight'}<=set(checkpoints):raise RuntimeError('Missing original standard/tight checkpoints')
report=dict(version='v18.40',status='PREFLIGHT_ONLY_V18_39_PARENT_AUTHENTICATED',
            parent_sha256=auth40['parent_sha256'],parent_verified_members=auth40['verified_members'],
            parent_original_cases=p39['original_cases'],parent_original_integrations=p39['original_integrations'],
            parent_frames=len(frame),parent_vectors=len(vec),
            source_kappa0_unchanged=True,no_new_original_integrations=True)
(R/'results').mkdir(exist_ok=True)
(R/'results/frozen_preflight_v18_40.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
