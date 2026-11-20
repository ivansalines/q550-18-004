#!/usr/bin/env python3
"""Two ORIGINAL-only comparison plots; never produced by frozen preflight."""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1]
p=R/'results'
a=list(csv.DictReader((p/'original_v18_47_kernel_vectors.csv').open()))
b=list(csv.DictReader((p/'original_v18_47_component_budget.csv').open()))
fig,ax=plt.subplots(figsize=(10,5))
for name in ('mixed_compensated','mixed_reversed','polarized_rhs','acceleration_gradient'):
    x=sorted({float(r['h']) for r in a if r['kernel']==name})
    y=[float(np.median([float(r['relative_to_jac']) for r in a if r['kernel']==name and float(r['h'])==h])) for h in x]
    ax.plot(x,y,marker='o',label=name)
ax.set(xlabel='Normalized probe radius',ylabel='Median relative difference to numerical J reference',title='Q550 v18.47: alternative Hessian kernels (diagnostic)')
ax.set_yscale('symlog',linthresh=1e-8);ax.legend(fontsize=8);ax.grid(alpha=.2)
fig.tight_layout();fig.savefig(p/'original_v18_47_cross_kernel.png',dpi=160);plt.close(fig)
fig,ax=plt.subplots(figsize=(10,5))
chosen=[r for r in b if r['kernel']=='mixed_compensated' and abs(float(r['h'])-.008)<1e-10]
sig={j:[float(r['signed_kernel_minus_jac']) for r in chosen if int(r['component'])==j] for j in range(10)}
ax.boxplot([sig[j] for j in range(10)],tick_labels=[str(j) for j in range(10)],showfliers=False)
ax.axhline(0,color='black',linewidth=.7)
ax.set(xlabel='Normalized material component',ylabel='Signed compensated-minus-numerical-J',title='Q550 v18.47: component-resolved residual at h=0.008')
ax.grid(alpha=.2);fig.tight_layout();fig.savefig(p/'original_v18_47_component_signature.png',dpi=160);plt.close(fig)
print('V18_47_PLOTS completed')
