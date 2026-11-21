#!/usr/bin/env python3
"""Optional reproducible plot of signed branch/cancellation histories."""
import csv
from collections import defaultdict
from pathlib import Path
root=Path(__file__).resolve().parents[1]
rows=list(csv.DictReader((root/'results'/'transported_kernel_ledger.csv').open()))
try:
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
except ImportError:
 print('matplotlib not installed; plots skipped (CSV data unaffected)')
 raise SystemExit(0)
focus=defaultdict(list)
for r in rows:
 if r['family'].startswith('phaseblind') and float(r['anchor_xi'])==3.5:
  for variant in ('reference','geometric'):
   focus[(int(r['branch']),variant)].append((float(r['relative_historical_T']),float(r[variant+'_cancellation_ratio'])))
fig,ax=plt.subplots(figsize=(8.7,4.7))
for (branch,variant),rr in sorted(focus.items()):
 rr.sort()
 ax.plot([x for x,y in rr],[y for x,y in rr],label=f'branch {branch} | {variant}')
ax.set_xlabel('time / historical T')
ax.set_ylabel('Cancellation ratio (sum of norms / norm of total)')
ax.set_title('Q550 v18.48f · phaseblind · xi = 3.5')
ax.legend(fontsize=8);ax.grid(alpha=.25)
fig.tight_layout()
fig.savefig(root/'results'/'phaseblind_xi35_cancellation_transport.png',dpi=160)
plt.close(fig)
# Signed branch localization across family and xi, never ranking hidden in a maximum.
rows2=list(csv.DictReader((root/'results'/'branch_pairs.csv').open()))
fig,ax=plt.subplots(figsize=(8.7,4.7))
for family in sorted({r['family'] for r in rows2}):
 rr=sorted((r for r in rows2 if r['family']==family),key=lambda r:float(r['anchor_xi']))
 for branch in (0,1):
  key='reference_branch'+str(branch)+'_cmax'
  ax.plot([float(r['anchor_xi']) for r in rr],[float(r[key]) for r in rr],marker='o',label=f'{family.split("_")[0]} | branch {branch}')
ax.set_xlabel('anchor xi')
ax.set_ylabel('Peak cancellation ratio')
ax.set_title('Q550 v18.48f · peak cancellation by branch')
ax.legend(fontsize=8);ax.grid(alpha=.25)
fig.tight_layout();fig.savefig(root/'results'/'branch_cancellation_localization.png',dpi=160)
plt.close(fig)
print('Created optional diagnostic plots')
