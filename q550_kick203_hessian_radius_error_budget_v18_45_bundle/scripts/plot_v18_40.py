#!/usr/bin/env python3
"""Plot original Q550 temporal observations; never plot synthetic preflight as original."""
import csv,sys
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1]
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
out=R/'results'
def read(n):
    with (out/n).open(newline='') as f:return list(csv.DictReader(f))
t=read('original_v18_40_temporal_sources.csv')
fams=read('original_v18_40_family_summary.csv')
res=read('original_v18_40_resolution.csv')
fig,ax=plt.subplots(figsize=(11.4,6.3))
for fid in [x['case_id'] for x in fams]:
    xx=[r for r in t if r['case_id']==fid]
    target='phaseblind' in fid and ('xi=3.5|branch=0' in fid or 'xi=4|branch=0' in fid)
    ax.plot([float(x['fraction_of_common_time']) for x in xx],
            [float(x['odd_source_turn_direct_deg']) for x in xx],
            lw=2.0 if target else 0.85,alpha=.95 if target else .37,
            label=fid.replace('phaseblind_d2_lam-1.00_mod+0.00','blind').replace('phasecov_d2_lam-1.00_mod+0.00','cov') if target else None)
ax.set(xlabel='Fraction of the shared within-support interval',ylabel='Direct-frame odd-source turning angle (degrees)',title='Q550 v18.40 — Temporal rotation of the full 9D odd source')
ax.grid(alpha=.2);ax.legend(loc='best',fontsize=8);fig.tight_layout();fig.savefig(out/'original_v18_40_temporal_rotation.png',dpi=160);plt.close(fig)
fig,ax=plt.subplots(figsize=(10.5,5.7))
for fld,style,label in [('endpoint_turn_direct_deg','o','Direct minimal'),('endpoint_turn_path8_deg','x','Incremental 8'),('endpoint_turn_path64_deg','+','Incremental 64')]:
    ax.scatter(np.arange(len(fams)),[float(x[fld]) for x in fams],marker=style,label=label)
ax.set(xlabel='Diagnostic family (original v18.39 order)',ylabel='Full odd source end-to-start angle (degrees)',title='Declared transport comparison (not intrinsic torsion)')
ax.grid(alpha=.2);ax.legend();fig.tight_layout();fig.savefig(out/'original_v18_40_transport_comparison.png',dpi=160);plt.close(fig)
fig,ax=plt.subplots(figsize=(10.3,5.6))
for fid in [x['case_id'] for x in fams]:
    s=sorted([r for r in res if r['case_id']==fid],key=lambda r:int(r['uniform_time_subintervals']))
    if 'phaseblind' in fid and ('xi=3.5|branch=0' in fid or 'xi=4|branch=0' in fid):
        ax.plot([int(x['uniform_time_subintervals']) for x in s],[float(x['total_spherical_angular_travel_deg']) for x in s],'-o',label=fid.replace('phaseblind_d2_lam-1.00_mod+0.00','blind'))
ax.set(xlabel='Number of temporal subintervals',ylabel='Total angular path (degrees)',title='Q550 v18.40 — Nested temporal sampling resolution')
ax.grid(alpha=.2);ax.legend(fontsize=8);fig.tight_layout();fig.savefig(out/'original_v18_40_temporal_resolution.png',dpi=160);plt.close(fig)
print('V18_40_ORIGINAL_PLOTS_COMPLETE')
