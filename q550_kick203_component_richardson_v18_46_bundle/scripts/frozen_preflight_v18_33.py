#!/usr/bin/env python3
"""Authentic v18.32 archival selection-only preflight; NO NEW Q550 RUN."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from checkpoint_io_v18_33 import read
from cofactor_return_math_v18_33 import training_rank,shortlist
_,prov,proof=read()
ranked=training_rank(proof['full_candidate_scan'])
picked=shortlist(proof['full_candidate_scan'])
report=dict(version='v18.33',status='AUTHENTICATED_V18_32_ONLY_NO_NEW_Q550_RUN',
 parent_v18_32_SHA256=prov['v18_32_server_sha256'],
 parent_v18_32_verified_members=prov['v18_32_verified_members'],
 old_finalists=[x['id'] for x in proof['finalists']],
 corrected_training_only_finalists=[x['id'] for x in picked],
 best_training_relative_residual=float(ranked[0]['train_relative_residual']),
 best_candidate=ranked[0]['id'],
 all_40_original_training_scores_immutable=True,
 no_new_original_integrations=True,no_new_manifold_certificate=True)
( ROOT/'results').mkdir(exist_ok=True)
(ROOT/'results/frozen_preflight_v18_33.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
