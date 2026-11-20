#!/usr/bin/env python3
"""Authenticate input lineage; no new Q550 ODE run here."""
from pathlib import Path
import json,sys,csv
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from checkpoint_io_v18_30 import read
D,prov,proof,profiles,summary=read()
rows=list(csv.DictReader(summary.splitlines()))
assert len(rows)==6 and proof['original_integrations']==34
out=dict(version='v18.30',status='VERIFIED_GENUINE_V18_29_LINEAGE_PREFLIGHT_ONLY',
    sha256=prov['v18_29_original_SHA256'],verified_files=prov['v18_29_verified_members'],
    parent_integrations=proof['original_integrations'],compensated_pairs=len(rows),
    no_new_original_Q550_integrations=True,
    constraints='NO_MATERIAL_INVARIANTS_DECLARED; common-phase gauge only unless plugin provided')
(ROOT/'results/frozen_preflight_v18_30.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
