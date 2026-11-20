#!/usr/bin/env python3
"""Plots measured signed source coupling and the held-out amplitude response."""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1]
with (R/'results/original_v18_38_coupled_profiles.csv').open() as fh:rows=list(csv.DictReader(fh))
with (R/'results/original_v18_38_coupled_fits.csv').open() as fh:fits=list(csv.DictReader(fh))
fig,axs=plt.subplots(1,2,figsize=(12,4.5))
for name in sorted({r['id'] for r in rows}):
    z=[r for r in rows if r['id']==name and r['moment']=='initial']
    axs[0].scatter([float(v['level_rate']) for v in z],[float(v['signed_tilt_rate']) for v in z],s=15,alpha=.7,label=name.split('_')[0])
axs[0].set_xlabel('Signed normalized level-source rate')
axs[0].set_ylabel('Signed projected normal-tilt source rate')
axs[0].set_title('Initial coupled sources: original Q550')
axs[0].legend(fontsize=8)
for name in sorted({r['id'] for r in fits}):
    z=[r for r in fits if r['id']==name and r['moment']=='common_inside' and r['observable'] in ('level_normal_offset','signed_normal_tilt_rad')]
    for obs in ('level_normal_offset','signed_normal_tilt_rad'):
        zz=[float(r['holdout_relative_error']) for r in z if r['observable']==obs]
        axs[1].scatter([obs+'\n'+name.split('_')[0]]*len(zz),zz,s=20,alpha=.65)
axs[1].set_yscale('symlog',linthresh=1e-10)
axs[1].set_ylabel('Held-out outer-amplitude relative error')
axs[1].set_title('Local response transfer within diagnostic support')
axs[1].tick_params(axis='x',labelrotation=20)
fig.tight_layout()
fig.savefig(R/'results/original_v18_38_coupled_scaling.png',dpi=175)
plt.close(fig)
print('V18_38_PLOT_CREATED')
