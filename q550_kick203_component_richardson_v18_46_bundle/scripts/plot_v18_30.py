#!/usr/bin/env python3
"""Source-level and observation-level comparisons from original-Q550 server data."""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'results'
def read(name):return list(csv.DictReader((R/name).open()))
source=read('original_independent_hessian_source_v18_30.csv')
profile=read('original_independent_hessian_profiles_v18_30.csv')
fig,ax=plt.subplots(figsize=(9,5.4))
for name in ('raw6','material3'):
 for h in (.02,.08):
  r=sorted([z for z in source if z['observer']==name and abs(float(z['h'])-h)<1e-10],key=lambda z:float(z['xi']))
  ax.semilogy([float(z['xi'])-float(z['anchor']) for z in r],
    [max(float(z['source_disagreement_rel']),1e-17) for z in r],
    label=f'{name}: RHS vs J (h={h})')
ax.set_xlabel('Δ ξ from anchor');ax.set_ylabel('Relative independent Hessian-source discrepancy');ax.grid(True,alpha=.3);ax.legend(fontsize=8)
fig.tight_layout();fig.savefig(R/'original_rhs_vs_jac_hessian_v18_30.png',dpi=175);plt.close(fig)
fig,ax=plt.subplots(figsize=(9,5.4))
for name in ('raw6','material3'):
 r=sorted([z for z in profile if z['observer']==name and abs(float(z['h'])-.02)<1e-10],key=lambda z:float(z['xi']))
 x=[float(z['xi'])-float(z['anchor']) for z in r]
 ax.plot(x,[float(z['new_rhs_even_norm']) for z in r],label=f'{name}: RHS Hessian')
 ax.plot(x,[float(z['archived_compensated_even_norm']) for z in r],ls='--',label=f'{name}: v18.29 compensated')
ax.set_xlabel('Δ ξ from anchor');ax.set_ylabel('Normalized even profile (ε=1e-5)');ax.grid(True,alpha=.3);ax.legend(fontsize=8)
fig.tight_layout();fig.savefig(R/'original_constraint_hessian_even_profiles_v18_30.png',dpi=175);plt.close(fig)
print('PLOTS_ORIGINAL_V18_30_CREATED')
