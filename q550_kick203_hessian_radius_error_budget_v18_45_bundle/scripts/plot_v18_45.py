#!/usr/bin/env python3
"""Original-only Q550 figures, no images from synthetic/frozen data."""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1]
def read(fn):
    with (R/'results'/fn).open() as f:return list(csv.DictReader(f))
rhs=read('original_v18_45_rhs_radius_scan.csv')
sumr=read('original_v18_45_snapshot_summary.csv')
com=read('original_v18_45_component_error_budget.csv')
if (len(rhs),len(sumr),len(com))!=(396,36,360):raise RuntimeError('Do not plot incomplete original result')
fig,ax=plt.subplots(figsize=(9,5))
for name in sorted({r['id'] for r in rhs}):
    vals=[]
    for h in sorted({float(r['rhs_normalized_h']) for r in rhs}):
        a=[float(r['relative_to_jac_median']) for r in rhs if r['id']==name and float(r['rhs_normalized_h'])==h]
        vals.append((h,float(np.median(a))))
    ax.loglog([x[0] for x in vals],[x[1] for x in vals],marker='o',label=name)
ax.set_xlabel('RHS normalized mixed-difference radius h')
ax.set_ylabel('Median relative disagreement to Jacobian consensus')
ax.set_title('Q550 v18.45: original numerical Hessian radius scan')
ax.grid(True,which='both',alpha=.2);ax.legend(fontsize=7);fig.tight_layout()
fig.savefig(R/'results/original_v18_45_radius_convergence.png',dpi=155);plt.close(fig)
fig,ax=plt.subplots(figsize=(9,5))
cases=sorted({r['case_id'] for r in com})
heat=np.zeros((len(cases),10))
for i,c in enumerate(cases):
    for j in range(10):
        heat[i,j]=np.median([float(r['squared_norm_error_share']) for r in com if r['case_id']==c and int(r['component'])==j])
im=ax.imshow(heat,aspect='auto',interpolation='nearest',vmin=0,vmax=1)
ax.set_xticks(range(10));ax.set_xlabel('Normalized material component j')
ax.set_yticks(range(len(cases)),[f'{i+1:02d} '+c.replace('phasecov_d2_lam-1.00_mod+0.00','phasecov').replace('phaseblind_d2_lam-1.00_mod+0.00','phaseblind') for i,c in enumerate(cases)],fontsize=7)
ax.set_title('Share of squared cross-path discrepancy by component')
fig.colorbar(im,ax=ax,label='Fraction of squared discrepancy')
fig.tight_layout();fig.savefig(R/'results/original_v18_45_component_error_budget.png',dpi=155);plt.close(fig)
fig,ax=plt.subplots(figsize=(9,5))
x=np.arange(len(sumr))
ax.semilogy(x,[max(float(r['baseline_0006_error']),1e-16) for r in sumr],'o',label='Historical h = 0.006')
ax.semilogy(x,[max(float(r['selected_cross_path_error']),1e-16) for r in sumr],'x',label='Minimum tested cross-path disagreement')
ax.semilogy(x,[max(float(r['jac_internal_reference_spread']),1e-16) for r in sumr],'.',label='Jacobians: consensus spread')
ax.set_xlabel('Matched snapshot index (12 families × 3 times)')
ax.set_ylabel('Relative discrepancy (reference-dependent)')
ax.set_title('Q550 v18.45: cross-path error versus internal reference spread')
ax.grid(alpha=.2);ax.legend(fontsize=9)
fig.tight_layout();fig.savefig(R/'results/original_v18_45_crosspath_error_budget.png',dpi=155);plt.close(fig)
print('V18_45_ORIGINAL_PLOTS_COMPLETED')
