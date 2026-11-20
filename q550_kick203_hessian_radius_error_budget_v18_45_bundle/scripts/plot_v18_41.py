#!/usr/bin/env python3
"""Render only original Q550 v18.41 data; do not synthesize original results."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
R=Path(__file__).resolve().parents[1]
res=R/'results'
p=res/'original_v18_41_plane_profiles.csv'
if not p.exists():raise RuntimeError('No original server v18.41 CSV')
rows=list(csv.DictReader(p.open()))
if len(rows)!=1548:raise RuntimeError('Original Q550 temporal records incomplete')
by={}
for z in rows:by.setdefault(z['case_id'],[]).append(z)
for name,col,ylabel,figure in [
    ('plane','departure_from_endpoint_plane_deg','Off-plane departure (degrees)','original_v18_41_off_plane_departure.png'),
    ('chord','raw_chord_error_rel','Raw vector affine-chord residual / scale','original_v18_41_raw_chord_departure.png'),
    ('turn','initial_direction_turn_deg','Odd-source direction turn (degrees)','original_v18_41_direction_turn.png')]:
    fig,ax=plt.subplots(figsize=(10,5.2))
    for cid,rec in sorted(by.items()):
        x=[float(v['fraction_of_common_time']) for v in rec]
        y=[float(v[col]) for v in rec]
        ax.plot(x,y,lw=1.2,alpha=.85,label=cid.replace('phaseblind_','pb_').replace('phasecov_','pc_'))
    ax.set_xlabel('Fraction of common within-support interval')
    ax.set_ylabel(ylabel);ax.set_title('Q550 v18.41 — original source-response '+name)
    ax.grid(True,alpha=.2)
    ax.legend(loc='upper left',bbox_to_anchor=(1.,1.),fontsize=6,frameon=False)
    fig.tight_layout();fig.savefig(res/figure,dpi=170,bbox_inches='tight');plt.close(fig)
print('V18_41_PLOTS_CREATED',len(by))
