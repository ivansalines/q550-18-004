#!/usr/bin/env python3
"""Optional plots from measured 11D-IVP counterfactuals (no interpolated physics)."""
import csv
from pathlib import Path
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'results'
with open(out/'time_state_observer_factorial.csv',newline='') as f:rows=list(csv.DictReader(f))
q=[r for r in rows if r['family'].startswith('phasecov') and r['seed_branch']=='0' and r['kappa_kind']=='original' and r['mode']=='historical_transport' and r['interval_kind']=='crest_4_0']
q=sorted(q,key=lambda r:float(r['xi_end']))
fig,ax=plt.subplots(figsize=(8.5,4.6))
for key,label in [('cancellation_ratio_state_only','Only state changed'),('cancellation_ratio_observer_only','Only covector changed'),('cancellation_ratio_end','State and covector together')]:
 ax.plot([float(r['xi_end']) for r in q],[float(r[key]) for r in q],marker='.',markersize=3,label=label)
ax.set_yscale('log');ax.set_xlabel('Common lifted phase section ξ');ax.set_ylabel('Projected cancellation ratio (log axis)')
ax.set_title('Q550 phasecov · initial branch 0 · historical transported observer')
ax.axvline(4.25,linewidth=.9,linestyle=':',color='gray');ax.legend();fig.tight_layout()
fig.savefig(out/'crest_state_observer_counterfactual.png',dpi=180);plt.close(fig)
with open(out/'crest_attribution_xi400_to_425.csv',newline='') as f:ar=list(csv.DictReader(f))
r=next(r for r in ar if r['family'].startswith('phasecov') and r['seed_branch']=='0' and r['kappa_kind']=='original' and r['mode']=='historical_transport')
labels=['State 4.00 / Observer 4.00','State 4.25 / Observer 4.00','State 4.00 / Observer 4.25','State 4.25 / Observer 4.25']
vals=[float(r[x]) for x in ['cancellation_ratio_start','cancellation_ratio_state_only','cancellation_ratio_observer_only','cancellation_ratio_end']]
fig,ax=plt.subplots(figsize=(8,4.5));bars=ax.barh(labels,vals);ax.set_xlabel('Cancellation ratio C');ax.set_title('Four-corner frozen-chart diagnostic · physical endpoints only')
for b,v in zip(bars,vals):ax.text(v+max(vals)*.01,b.get_y()+b.get_height()/2,f'{v:.3f}',va='center')
ax.set_xlim(0,max(vals)*1.2);fig.tight_layout();fig.savefig(out/'crest_four_corner.png',dpi=180);plt.close(fig)
print('FIGURES',out/'crest_state_observer_counterfactual.png',out/'crest_four_corner.png')
