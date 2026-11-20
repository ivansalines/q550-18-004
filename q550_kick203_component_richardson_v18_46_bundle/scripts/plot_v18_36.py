#!/usr/bin/env python3
"""Scientific summaries of new original normal transport results."""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1];inp=R/'results/original_v18_36_normal_profiles.csv'
if not inp.is_file():raise RuntimeError('Original Q550 v18.36 absent')
rows=list(csv.DictReader(inp.open()))
fig,ax=plt.subplots(figsize=(9,5.3))
for ident in sorted(set(r['id'] for r in rows)):
    r=[x for x in rows if x['id']==ident and float(x['anchor_xi'])==4.0 and int(x['branch'])==0]
    ax.plot([float(x['delta_xi']) for x in r],[float(x['transported_candidate_normal_angle_deg']) for x in r],label=ident)
ax.set(xlabel='Elapsed xi',ylabel='Candidate normal vs transported cotangent (degrees)',title='Q550 v18.36 — Normal transport vs frozen candidate')
ax.legend(fontsize=8);ax.grid(alpha=.25);fig.tight_layout();fig.savefig(R/'results/original_v18_36_normal_transport.png',dpi=160);plt.close(fig)
