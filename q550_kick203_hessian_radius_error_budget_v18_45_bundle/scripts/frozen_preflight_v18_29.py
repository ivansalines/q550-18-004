#!/usr/bin/env python3
"""Offline authentic v18.28 data authentication and non-ODE math preflight."""
import csv,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from checkpoint_io_v18_29 import read
D, provenance, proof, old=read()
rows=list(csv.DictReader(old.splitlines()))
assert len(rows)==8 and proof['new_original_ode_integrations']==67
r6=[r for r in rows if r['observer']=='raw6' and float(r['epsilon'])==1e-5]
assert len(r6)==1
summary={'version':'v18.29','status':'VERIFIED_AUTHENTIC_PARENT_AND_SYNTHETIC_PREFLIGHT_ONLY',
 'parent_proof':proof['status'],'parent_sha256':provenance['v18_28_parent_sha256'],
 'parent_manifest_members_verified':provenance['v18_28_verified_members'],
 'parent_original_ode_integrations':proof['new_original_ode_integrations'],
 'parent_raw6_xi4p5_e1e5_even_rms':float(r6[0]['paired_even_normalized_rms']),
 'parent_raw6_xi4p5_e1e5_second_variational_predicted_rms':float(r6[0]['second_variational_even_predicted_rms']),
 'no_new_original_Q550_ODE_integrations':True}
(ROOT/'results/frozen_preflight_v18_29.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
