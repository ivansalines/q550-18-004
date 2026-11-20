#!/usr/bin/env python3
"""Authenticate v18.31 original result, perform no new Q550 integrations."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'reference')]
from checkpoint_io_v18_32 import read
from original_material_invariants_v18_31 import select_original_checkpoint
chk,prov,prior=read()
d=select_original_checkpoint(chk)
if prior['material_constraint_status']!='NO_PHYSICAL_MATERIAL_INVARIANT_CERTIFIED':raise RuntimeError('Incorrect ancestor material status')
out={'version':'v18.32','status':'AUTHENTICATED_PARENT_V18_31_ONLY_NO_NEW_Q550_RUN',
     'parent_SHA256':prov['v18_31_server_SHA256'],
     'parent_verified_members':prov['v18_31_verified_members'],
     'parent_numeric_survivors':prior['candidates_surviving_numeric'],
     'parent_material_status':prior['material_constraint_status'],
     'verified_source_slope':float(d['slope']),
     'phase_covariant_manifold_not_equal_first_integral':True,
     'no_new_original_integrations':True}
(ROOT/'results/frozen_preflight_v18_32.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
