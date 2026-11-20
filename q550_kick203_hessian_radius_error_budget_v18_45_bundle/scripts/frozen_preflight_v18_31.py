#!/usr/bin/env python3
"""Authenticate the genuine v18.30 parent; original field is NOT run locally."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from checkpoint_io_v18_31 import read
D,prov,proof,*_=read()
report=dict(version='v18.31',status='AUTHENTICATED_V18_30_ONLY_NO_NEW_Q550_RUN',parent_SHA256=prov['v18_30_server_SHA256'],parent_verified_members=prov['v18_30_verified_members'],source_model_parent_status=proof['status'],material_invariant_status='NOT_DISCOVERED_IN_PREFLIGHT',no_new_original_integrations=True)
(ROOT/'results/frozen_preflight_v18_31.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
