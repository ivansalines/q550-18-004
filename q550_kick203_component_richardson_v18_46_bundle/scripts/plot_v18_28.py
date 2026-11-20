"""Plot genuine results only; do not create a misleading server figure preflight."""
import csv,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'results'
if not (R/'original_second_variational_pairs_v18_28.csv').exists():
    print('No original v18.28 server result: skip plot');sys.exit(0)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
with (R/'original_second_variational_pairs_v18_28.csv').open() as f:rows=list(csv.DictReader(f))
labels=[f"{r['observer']} ξ={r['anchor']} ε={float(r['epsilon']):g}" for r in rows]
x=np.arange(len(rows))
y1=[float(r['paired_even_normalized_rms']) for r in rows]
y2=[float(r['second_variational_even_predicted_rms']) for r in rows]
y3=[float(r['second_variation_stencil_difference_rms']) for r in rows]
fig,ax=plt.subplots(figsize=(12,5));ax.semilogy(x,y1,'o-',label='Independent nonlinear even (measured)')
ax.semilogy(x,y2,'s-',label='Second-variational even (predicted)')
ax.semilogy(x,y3,'^--',label='Hessian stencil change (indicator, NOT bound)')
ax.set_xticks(x,labels,rotation=42,ha='right',fontsize=8)
ax.set_ylabel('Observed RMS in reduced coordinate / ε');ax.set_title('Q550 v18.28 | second variation versus numerical floor')
ax.grid(True,alpha=.25);ax.legend();fig.tight_layout()
fig.savefig(R/'original_second_variational_vs_pairs_v18_28.png',dpi=180);plt.close(fig)
with (R/'original_second_variational_profiles_v18_28.csv').open() as f:prof=list(csv.DictReader(f))
q=[r for r in prof if r['observer']=='material3' and float(r['anchor'])==4.5 and float(r['epsilon'])==1e-5]
if q:
 xx=np.array([float(r['delta_xi']) for r in q]);p=np.array([float(r['paired_even_normalized_norm']) for r in q]);
 v=np.array([float(r['predicted_even_normalized_norm']) for r in q])
 fig,ax=plt.subplots(figsize=(9,4));ax.plot(xx,p,'o-',label='Nonlinear +/- measured');ax.plot(xx,v,'s-',label='Second variational source')
 ax.set_xlabel('ξ - anchor');ax.set_ylabel('Normalized even magnitude');ax.legend();ax.grid(True,alpha=.25)
 fig.tight_layout();fig.savefig(R/'original_second_variational_profile_v18_28.png',dpi=180);plt.close(fig)
print('v18.28 genuine server plots saved')
