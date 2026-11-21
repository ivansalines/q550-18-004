#!/usr/bin/env python3
"""Optional depiction of genuine Q550 IVP continuation; no interpolation seeds."""
import csv
from pathlib import Path
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
source=ROOT/'results'/'physical_section_cancellation.csv'
if not source.exists():source=ROOT/'preflight'/'physical_section_cancellation.csv'
rows=list(csv.DictReader(source.open()))
fig,ax=plt.subplots(figsize=(10,5.6))
for branch in (0,1):
 for mode in ('historical_transport','activity_sin_lie_recomputed'):
  a=[r for r in rows if r['family'].startswith('phaseblind') and r['seed_branch']==str(branch) and r['kappa_kind']=='original' and r['observer_mode']==mode]
  a.sort(key=lambda r:float(r['xi']))
  ax.plot([float(r['xi']) for r in a],[float(r['ratio']) for r in a],label=f'Branch {branch} / {mode}',marker='.' if mode=='historical_transport' else None)
ax.set_yscale('log');ax.set_xlabel('Reference support time xi (same Theta-clock sections)')
ax.set_ylabel('Cancellation ratio, logarithmic')
ax.set_title('Q550 phaseblind: physical IVP continuation, not branch-anchor interpolation')
ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.tight_layout()
out=source.parent/'phaseblind_physical_section_cancellation.png'
fig.savefig(out,dpi=150);print('FIGURE',out)
