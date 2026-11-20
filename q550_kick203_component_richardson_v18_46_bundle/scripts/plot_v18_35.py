#!/usr/bin/env python3
"""Q550 v18.35 genuine run charts: fixed vs co-moving and defect source."""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1];O=R/'results'
with (O/'original_v18_35_microprofiles.csv').open() as fh:rows=list(csv.DictReader(fh))
fig,axes=plt.subplots(1,2,figsize=(12,4.4))
for name,col in [('phasecov_d2_lam-1.00_mod+0.00','C0'),('phaseblind_d2_lam-1.00_mod+0.00','C1')]:
    subset=[r for r in rows if r['id']==name and float(r['anchor_xi'])==3.5 and int(r['seed_index'])==0]
    t=np.array([float(r['delta_xi']) for r in subset]);fixed=np.array([float(r['distance_fixed_anchor']) for r in subset]);moving=np.array([float(r['distance_co_moving']) for r in subset]);
    axes[0].plot(t[1:],fixed[1:],color=col,linestyle='--',label=name.split('_')[0]+' fixed')
    axes[0].plot(t[1:],moving[1:],color=col,label=name.split('_')[0]+' co-moving')
    pred=np.array([float(r['integrated_source_defect']) for r in subset]);obs=np.array([float(r['observed_transport_defect']) for r in subset]);
    axes[1].plot(t,obs,color=col,label=name.split('_')[0]+' observed')
    axes[1].plot(t,pred,color=col,linestyle='--',alpha=.6,label=name.split('_')[0]+' accumulated')
axes[0].set_xscale('symlog',linthresh=2e-7);axes[0].set_yscale('symlog',linthresh=.05)
axes[0].set_xlabel('delta xi');axes[0].set_ylabel('normalized diagnostic distance')
axes[0].set_title('Fixed anchor vs moving Q550 reference')
axes[1].set_xlabel('delta xi');axes[1].set_ylabel('signed C transport defect')
axes[1].set_title('Direct and integrated defect (micro-time)')
for ax in axes:ax.grid(True,alpha=.2);ax.legend(fontsize=7)
fig.tight_layout();fig.savefig(O/'original_v18_35_moving_frame_microdefect.png',dpi=170);plt.close(fig)
print('V18_35_PLOTS_COMPLETED')
