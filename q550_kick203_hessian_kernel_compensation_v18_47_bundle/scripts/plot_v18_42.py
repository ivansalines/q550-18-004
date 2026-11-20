#!/usr/bin/env python3
"""Plot only ORIGINAL v18.42 Q550 rows; no synthetic original results."""
import csv
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1];out=R/'results'
def rows(file):return list(csv.DictReader((out/file).open()))
w=rows('original_v18_42_window_scaling.csv')
d=rows('original_v18_42_second_derivative_stencils.csv')
n=rows('original_v18_42_tolerance_noise.csv')
focus=sorted({r['case_id'] for r in n})
if len(w)!=48 or len(d)!=144 or len(n)!=16 or len(focus)!=2:raise RuntimeError('Incomplete ORIGINAL v18.42 plot inputs')
# Window dependence of a geometric observable without differentiating twice.
fig,ax=plt.subplots(figsize=(9,5.4))
for case in focus:
    a=sorted([r for r in w if r['case_id']==case],key=lambda x:float(x['window_fraction']))
    x=np.array([float(r['window_fraction']) for r in a]);y=np.array([float(r['chord_max_rel']) for r in a]);
    ax.loglog(x,y,'o-',label=case.split('|')[1]+' '+case.split('|')[2])
ax.set(xlabel='Window fraction of common within-support horizon',ylabel='Maximum full-vector chord departure (fixed normalization)',title='Q550 v18.42 — window scaling of diagnostic chord departure')
ax.grid(alpha=.25);ax.legend(fontsize=8)
fig.tight_layout();fig.savefig(out/'original_v18_42_window_scaling.png',dpi=160);plt.close(fig)
# Five and three point numerical second derivatives can diverge at small steps.
fig,axs=plt.subplots(1,2,figsize=(13,4.9))
for idx,case in enumerate(focus):
    for frac in (1.,.125):
        a=sorted([r for r in d if r['case_id']==case and float(r['window_fraction'])==frac],key=lambda r:float(r['step']))
        x=[float(r['step']) for r in a]
        axs[idx].loglog(x,[max(float(r['accel3_5_disagreement_rel']),1e-18) for r in a],'.-',label=f'3/5 disagreement, window={frac}')
        axs[idx].loglog(x,[max(float(r['accel5_poly4_disagreement_rel']),1e-18) for r in a],'x--',label=f'5/poly disagreement, window={frac}')
    axs[idx].set(xlabel='Derivative step / common time',ylabel='Relative acceleration-stencil disagreement',title=case.split('|')[1])
    axs[idx].grid(alpha=.25);axs[idx].legend(fontsize=7)
fig.tight_layout();fig.savefig(out/'original_v18_42_second_derivative_stencil_disagreement.png',dpi=160);plt.close(fig)
# Absolute empirical tolerance effects, never equated to formal enclosures.
fig,axs=plt.subplots(1,2,figsize=(13,5))
for idx,case in enumerate(focus):
    for tier in ('coarse','tight'):
        a=sorted([r for r in n if r['case_id']==case and r['solver_tier']==tier],key=lambda r:float(r['window_fraction']))
        x=[float(r['window_fraction']) for r in a]
        y=[max(float(r['max_source_vector_disagreement']),1e-30) for r in a]
        axs[idx].loglog(x,y,'o-',label=f'{tier} original-ODE rerun')
    a=sorted([r for r in n if r['case_id']==case and r['solver_tier']=='tight'],key=lambda r:float(r['window_fraction']))
    axs[idx].loglog([float(r['window_fraction']) for r in a],[max(float(r['baseline_raw_chord_abs']),1e-30) for r in a],'k:',label='baseline chord signal')
    axs[idx].set(xlabel='Window fraction',ylabel='Absolute diagnostic 9D norm',title=case.split('|')[1])
    axs[idx].legend(fontsize=8);axs[idx].grid(alpha=.25)
fig.tight_layout();fig.savefig(out/'original_v18_42_solver_tolerance_floor.png',dpi=160);plt.close(fig)
print('PLOTS_V18_42',len(focus),flush=True)
