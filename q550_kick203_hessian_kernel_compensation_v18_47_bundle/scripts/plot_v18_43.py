#!/usr/bin/env python3
"""Plots only after verified original proof; no simulated result creation."""
from pathlib import Path
import json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1]
p=json.loads((R/'results/original_v18_43_proof.json').read_text())
if p.get('status')!='ORIGINAL_Q550_SOURCE_JET_PRECISION_AUDIT_COMPLETED':raise RuntimeError('No original v18.43 result')
with (R/'results/original_v18_43_second_jet_windows.csv').open() as f:rows=list(csv.DictReader(f))
fig,ax=plt.subplots(figsize=(9,5.7))
for cid in sorted({r['case_id'] for r in rows}):
 if 'phaseblind' not in cid or ('xi=3.5|branch=0' not in cid and 'xi=4|branch=0' not in cid):continue
 sel=[r for r in rows if r['case_id']==cid and float(r['window_fraction'])==1]
 ax.plot([int(r['stencil_subdivision']) for r in sel],
         [float(r['second_relative_disagreement']) for r in sel],marker='o',label=cid)
ax.set_xlabel('Subintervals per time window');ax.set_ylabel('Second jet vs double source differences (relative)')
ax.set_title('Q550 v18.43 | precision separation of second derivatives');ax.set_yscale('log');ax.grid(alpha=.3);ax.legend(fontsize=8)
fig.tight_layout();fig.savefig(R/'results/original_v18_43_precision_separation.png',dpi=170);plt.close(fig)
with (R/'results/original_v18_43_directional_step_stability.csv').open() as f:rows=list(csv.DictReader(f))
fig,ax=plt.subplots(figsize=(9,5.7))
for cid in sorted({r['case_id'] for r in rows}):
 if not ('phaseblind' in cid and 'branch=0' in cid and ('xi=3.5|' in cid or 'xi=4|' in cid)):continue
 sel=[r for r in rows if r['case_id']==cid and float(r['source_fraction'])==.5]
 ax.plot([float(r['scaled_directional_target']) for r in sel],
         [max(float(r['jet_disagreement_rel']),1e-18) for r in sel],marker='o',label=cid)
ax.set_xlabel('Scaled directional Jacobian step');ax.set_ylabel('First-jet disagreement relative to middle step');ax.set_xscale('log');ax.set_yscale('log')
ax.set_title('Q550 v18.43 | Jacobian directional-difference floor');ax.grid(alpha=.3);ax.legend(fontsize=8)
fig.tight_layout();fig.savefig(R/'results/original_v18_43_directional_step_floor.png',dpi=170);plt.close(fig)
print('V18_43_ORIGINAL_PLOTS_COMPLETED')
