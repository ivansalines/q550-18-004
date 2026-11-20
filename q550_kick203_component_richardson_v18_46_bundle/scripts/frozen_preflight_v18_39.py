#!/usr/bin/env python3
"""Frozen old Q550 data preflight only; NEVER substitute for original integration."""
from pathlib import Path
import sys,json
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R/'scripts'),str(R/'reference')]
from checkpoint_io_v18_39 import read_parent_v39
from checkpoint_io_v18_38 import read_parent_v38
from checkpoint_io_v18_37 import read_parent_v37
from checkpoint_io_v18_36 import read_parent_v36
from checkpoint_io_v18_35 import read_parent_v35
from checkpoint_io_v18_34 import read_parent
p38,cases38,profiles38,fits38,auth=read_parent_v39()
p37,_,_,_=read_parent_v38()
p36,_,_,_,_=read_parent_v37()
read_parent_v36();read_parent_v35()
checkpoints,proof,model,ret,ov,historical=read_parent()
if not {'standard','tight'}<=set(checkpoints):raise RuntimeError('Original standard/tight checkpoints unavailable')
report=dict(version='v18.39',status='PREFLIGHT_ONLY_V18_38_ORIGINAL_AUTHENTICATED',parent_sha256=auth['parent_sha256'],verified_members=auth['verified_members'],original_v18_38_cases=p38['original_cases'],old_profiles=len(profiles38),old_fits=len(fits38),previous_exit_observed=p38['observed_exit_count'],previous_right_censored=p38['right_censored_count'],no_new_original_integrations=True)
(R/'results').mkdir(exist_ok=True)
(R/'results/frozen_preflight_v18_39.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
