#!/usr/bin/env python3
"""Compare original nonlinear Q550 compensated even response with independent source."""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];r=ROOT/'results'
name=r/'original_compensated_pairs_v18_29.csv'
if not name.exists():raise RuntimeError('Server nonlinear pair output absent; plot not certified')
rows=list(csv.DictReader(name.open()))
fig,ax=plt.subplots(figsize=(9,5.2))
for case in sorted({x['observer'] for x in rows}):
 data=sorted((x for x in rows if x['observer']==case),key=lambda x:float(x['epsilon']),reverse=True)
 E=np.array([float(x['epsilon']) for x in data]);old=np.array([float(x['uncompensated_v18_28_vs_variation_rms']) for x in data]);new=np.array([float(x['compensated_vs_second_variational_rms']) for x in data]);
 pred=np.array([float(x['second_variational_even_rms']) for x in data]);tier=np.array([float(x['compensated_tier_difference_rms']) for x in data]);
 ax.loglog(E,new,'o-',label=f'{case}: compensated residual');ax.loglog(E,pred,'s--',label=f'{case}: predicted even');ax.loglog(E,tier,'.:',label=f'{case}: tier discrepancy')
 if np.any(np.isfinite(old)):ax.loglog(E,old,'x--',label=f'{case}: v18.28 conventional residual')
ax.set_xlabel('Perturbation amplitude ε');ax.set_ylabel('RMS of normalized even observables');ax.set_title('Q550 v18.29 — Compensated difference transport / independent second variation');ax.grid(alpha=.25,which='both');ax.legend(fontsize=7,ncol=2)
fig.tight_layout();fig.savefig(r/'original_compensated_floor_comparison_v18_29.png',dpi=165)
plt.close(fig)
P=list(csv.DictReader((r/'original_compensated_profiles_v18_29.csv').open()))
fig,ax=plt.subplots(figsize=(9,4.8))
for case in sorted({x['observer'] for x in P}):
 filtered=sorted((x for x in P if x['observer']==case and float(x['epsilon'])==1e-5 and x['tier']=='tight'),key=lambda x:float(x['xi']))
 x=np.array([float(t['delta_xi']) for t in filtered]);
 ax.plot(x,[float(t['compensated_even_norm']) for t in filtered],label=f'{case}: compensated pair')
 ax.plot(x,[float(t['second_variational_even_norm']) for t in filtered],ls='--',label=f'{case}: second variation')
ax.set_xlabel('Δξ from anchor');ax.set_ylabel('Norm of normalized even observed response')
ax.set_title('Q550 v18.29 — Time-localized even response, ε = 10⁻⁵')
ax.grid(alpha=.25);ax.legend();fig.tight_layout()
fig.savefig(r/'original_compensated_profile_v18_29.png',dpi=165);plt.close(fig)
print('CREATED original Q550 v18.29 figures')
