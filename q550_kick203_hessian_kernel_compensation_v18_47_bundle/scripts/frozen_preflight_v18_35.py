#!/usr/bin/env python3
"""Authenticate v18.34/33 originals. No v18.35 Q550 integration."""
import sys,json,csv,io,collections
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'scripts'))
from checkpoint_io_v18_35 import read_parent_v35
from near_surface_math_v18_35 import microgrid
proof,model,data,prov=read_parent_v35()
rows=list(csv.DictReader(io.StringIO(data)))
if len(rows)!=972 or len(microgrid())<150:raise RuntimeError('Incomplete parent original or microgrid')
initial=[r for r in rows if abs(float(r['delta_xi']))<1e-15]
if len(initial)!=12:raise RuntimeError('Authentic v18.34 initial seed count mismatch')
with (R/'results/frozen_v18_35_initial_sources.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=['id','seed_anchor','seed_index','C_initial','source_r_initial']);writer.writeheader()
    for r in initial:writer.writerow(dict(id=r['id'],seed_anchor=r['seed_anchor'],seed_index=r['seed_index'],C_initial=r['C'],source_r_initial=r['source_r']))
report=dict(version='v18.35',status='AUTHENTICATED_PARENT_ONLY_NO_NEW_Q550_INTEGRATION',
 parent_sha256=prov['parent_sha'],parent_member_count=prov['verified_members'],
 original_v18_34_sample_rows=len(rows),initial_seed_count=len(initial),
 source_at_zero_is_not_identically_zero=any(abs(float(r['source_r']))>1e-6 for r in initial),
 microgrid_count=len(microgrid()),
 no_new_original_ODE_run=True,no_physical_invariant=True)
(R/'results/frozen_preflight_v18_35.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
