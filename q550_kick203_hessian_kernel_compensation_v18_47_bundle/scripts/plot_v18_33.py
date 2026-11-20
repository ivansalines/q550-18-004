#!/usr/bin/env python3
"""Plot v18.33 original validation scores and clock-return residuals."""
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1]/'results'
proof=json.loads((R/'original_v18_33_proof.json').read_text())
rows=proof['candidate_diagnostics']
fig,ax=plt.subplots(figsize=(9.5,5.6))
labels=[x['id'].replace('phaseblind_','B ').replace('phasecov_','P ') for x in rows]
y=np.arange(len(rows))
ax.barh(y-.18,[max(x['training_score'],1e-16) for x in rows],height=.34,label='training: normalized cofactor residual')
ax.barh(y+.18,[max(x['holdout']['relative_residual'],1e-16) for x in rows],height=.34,label='held-out states: normalized cofactor residual')
ax.set_yticks(y,labels,fontsize=8);ax.invert_yaxis();ax.set_xscale('log');ax.set_xlabel('Relative Lie-cofactor defect (lower is better)')
ax.set_title('Q550 v18.33 · Repaired ranking and independent residuals')
ax.legend(fontsize=9);ax.grid(axis='x',alpha=.2);fig.tight_layout();fig.savefig(R/'original_v18_33_selection_vs_holdout.png',dpi=150);plt.close(fig)
with (R/'original_v18_33_clock_return.csv').open() as fp:entries=list(csv.DictReader(fp))
fig,ax=plt.subplots(figsize=(9.5,5.6))
for k,x in enumerate(rows):
    points=[e for e in entries if e['id']==x['id']]
    timeline=sorted(set(float(e['delta_xi']) for e in points))
    median=[np.median([float(e['abs_transport_residual']) for e in points if float(e['delta_xi'])==t]) for t in timeline]
    if timeline:
        ax.plot(timeline,np.maximum(median,1e-18),marker='o',lw=1.5,label=x['id'])
if not entries:
    ax.text(.5,.5,'No successful independent surface transports: unresolved',ha='center',va='center',transform=ax.transAxes)
ax.set_yscale('log');ax.set_xlabel('Elapsed ξ');ax.set_ylabel('Median |C(t) - exp(∫μ)C(0)|')
ax.set_title('Q550 v18.33 · Independent surface transport, phase-clock cycle at Δξ=1')
ax.grid(alpha=.2);ax.legend(fontsize=6,ncol=2,loc='best');fig.tight_layout();fig.savefig(R/'original_v18_33_surface_return.png',dpi=150);plt.close(fig)
print('V18_33_ORIGINAL_PLOTS_COMPLETED')
