#!/usr/bin/env python3
"""Plots of original Q550 v18.46 only. Never plot preflight as Q550 results."""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1]
def read(name):
    with (R/'results'/name).open() as f:return list(csv.DictReader(f))
rhs=read('original_v18_46_rhs_richardson_ladders.csv')
jac=read('original_v18_46_jac_richardson_ladders.csv')
comp=read('original_v18_46_component_ladders.csv')
summ=read('original_v18_46_snapshot_summary.csv')
if (len(rhs),len(jac),len(comp),len(summ))!=(288,216,2880,36):raise RuntimeError('Original Q550 rows missing: refuse graphs')
# 1: Empirical order measured, no assumed fourth-order
fig,ax=plt.subplots(figsize=(9,5))
for name in sorted({r['id'] for r in rhs}):
    hs=sorted({float(r['fine_h']) for r in rhs})
    med=[];valid=[]
    for h in hs:
        rr=[float(r['observed_order']) for r in rhs if r['id']==name and float(r['fine_h'])==h and r['observed_order']!='']
        med.append(np.median(rr) if rr else np.nan)
        valid.append(len(rr))
    ax.semilogx(hs,med,'o-',label=name.replace('_d2_lam-1.00_mod+0.00',''))
ax.axhline(4,color='gray',ls='--',lw=1,label='Nominal fourth order')
ax.set_xlabel('Finest RHS normalized radius in 4:2:1 ladder')
ax.set_ylabel('Observed vector difference order (not error order proof)')
ax.set_title('Original Q550 v18.46: empirical Richardson order')
ax.grid(alpha=.3);ax.legend(fontsize=8)
fig.tight_layout();fig.savefig(R/'results/original_v18_46_richardson_order.png',dpi=160);plt.close(fig)
# 2: Pair-to-pair disagreement and J reference sensitivity
fig,ax=plt.subplots(figsize=(9,5))
hs=sorted({float(r['fine_h']) for r in rhs})
for key,lab in (('rich_pair_disagreement','Successive Richardson extrapolations'),
                ('relative_rich_fine_to_jac_parent','Richardson vs J parent consensus'),
                ('relative_rich_fine_to_jac_small','Richardson vs J small consensus'),
                ('relative_rich_fine_to_jac_large','Richardson vs J large consensus')):
    vals=[np.median([float(r[key]) for r in rhs if float(r['fine_h'])==h]) for h in hs]
    ax.loglog(hs,vals,'o-',label=lab)
ax.set_xlabel('Finest RHS normalized radius')
ax.set_ylabel('Median relative discrepancy (diagnostic)')
ax.set_title('Original Q550 v18.46: Richardson and reference sensitivity')
ax.grid(which='both',alpha=.2);ax.legend(fontsize=8)
fig.tight_layout();fig.savefig(R/'results/original_v18_46_richardson_reference_sensitivity.png',dpi=160);plt.close(fig)
# 3: signed component differences — never hide signed cancellation
fig,axes=plt.subplots(3,1,figsize=(10,9),sharex=True)
for ax,j in zip(axes,(0,2,8)):
    xs=sorted({float(r['fine_h']) for r in comp})
    for key,lab in (('signed_error_parent','J parent'),('signed_error_small','J small'),('signed_error_large','J large')):
        med=[np.median([float(r[key]) for r in comp if int(r['component'])==j and float(r['fine_h'])==x]) for x in xs]
        ax.semilogx(xs,med,'o-',label=lab)
    ax.axhline(0,ls='--',color='gray',lw=.8)
    ax.set_ylabel(f'component {j}\nsigned residual')
    ax.grid(alpha=.25)
axes[0].legend(fontsize=8,ncol=3)
axes[-1].set_xlabel('Finest RHS radius in Richardson ladder')
fig.suptitle('Original Q550 v18.46: signed component-wise disagreements (0,2,8)')
fig.tight_layout();fig.savefig(R/'results/original_v18_46_component_signed_budget.png',dpi=160);plt.close(fig)
# 4: Compare behavior of Jacobian-only ladders
fig,ax=plt.subplots(figsize=(9,5))
hj=sorted({float(r['fine_h']) for r in jac})
for key,lab in (('rich_pair_disagreement','J extrapolation self-disagreement'),
                ('relative_rich_fine_to_jac_parent','J extrapolation vs J baseline')):
    vals=[np.median([float(r[key]) for r in jac if float(r['fine_h'])==h]) for h in hj]
    ax.loglog(hj,vals,'o-',label=lab)
ax.set_xlabel('Finest Jacobian directional target')
ax.set_ylabel('Median relative difference')
ax.set_title('Original Q550 v18.46: Jacobian internal radius consistency')
ax.grid(which='both',alpha=.2);ax.legend(fontsize=9)
fig.tight_layout();fig.savefig(R/'results/original_v18_46_jacobian_consistency.png',dpi=160);plt.close(fig)
print('ORIGINAL_V18_46_PLOTS_COMPLETED')
