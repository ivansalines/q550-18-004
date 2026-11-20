#!/usr/bin/env python3
"""Read/authenticate genuine original Q550 v18.36; no new Q550 ODE simulation."""
from pathlib import Path
import json,sys
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'scripts'))
from checkpoint_io_v18_37 import read_parent_v37
p,cases,profiles,attempts,auth=read_parent_v37()
if len(cases)!=12 or len(attempts)!=12 or len(profiles)!=1740:raise RuntimeError('Original Q550 record count changed')
if any(c['joint_initial_admissibility']!='True' for c in cases):raise RuntimeError('Original Q550 admissibility not passed')
if any(c['first_exit_status']!='CROSSED_AFTER_START' for c in cases):raise RuntimeError('Original first-exit analysis incomplete')
report=dict(version='v18.37',status='PREFLIGHT_ONLY_ORIGINAL_V18_36_AUTHENTICATED',parent_sha256=auth['parent_sha256'],verified_members=auth['verified_members'],
 original_parent_cases=len(cases),original_parent_samples=len(profiles),parent_all_seeds_admissible=True,
 no_new_q550_original_run=True,physical_material_constraint_not_certified=True)
(R/'results/frozen_preflight_v18_37.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
