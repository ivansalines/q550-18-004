#!/usr/bin/env python3
"""Parent-only preflight: validates Q550 v18.37 results; no new ODE run."""
import json,sys
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'scripts'))
from checkpoint_io_v18_38 import read_parent_v38
p,cases,profiles,auth=read_parent_v38()
if not p['all_seeds_jointly_admissible']:raise RuntimeError('Not all prior seeds admitted')
result=dict(version='v18.38',status='PREFLIGHT_ONLY_V18_37_PARENT_AUTHENTICATED',parent_sha256=auth['parent_sha256'],verified_members=auth['verified_members'],parent_cases=len(cases),parent_profiles=len(profiles),no_new_original_integrations=True)
(R/'results/frozen_preflight_v18_38.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
