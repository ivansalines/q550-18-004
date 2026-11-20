#!/usr/bin/env python3
"""Plot conditional diagnostic normal tilt and moving level for original Q550."""
from pathlib import Path
import csv,numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1];p=R/'results/original_v18_37_residual_gradient_profiles.csv'
if not p.is_file():raise RuntimeError('Original v18.37 data absent; cannot plot')
with p.open() as fh:data=list(csv.DictReader(fh))
colors=plt.rcParams['axes.prop_cycle'].by_key()['color']
for field,ylab,fn in [('normal_angle_stable_deg','Principal angle (degrees)','original_v18_37_stable_normal_tilt.png'),
                      ('moving_level_normal_offset','Signed diagnostic level offset','original_v18_37_moving_level.png'),
                      ('residual_gradient_transverse_norm','Norm of transverse residual gradient','original_v18_37_transverse_source.png')]:
 fig,ax=plt.subplots(figsize=(9,5.2))
 for k,(name,anchor,branch) in enumerate([('phasecov_d2_lam-1.00_mod+0.00',3.5,0),('phaseblind_d2_lam-1.00_mod+0.00',3.5,0),('phasecov_d2_lam-1.00_mod+0.00',4.,0),('phaseblind_d2_lam-1.00_mod+0.00',4.,0)]):
  rr=[x for x in data if x['id']==name and float(x['anchor_xi'])==anchor and int(x['branch'])==branch]
  x=np.array([float(v['delta_xi']) for v in rr]);y=np.array([float(v[field]) for v in rr]);exit_t=float(rr[0]['exit_delta_xi'])
  ax.plot(x,y,label=('with phase' if 'phasecov' in name else 'no phase')+f' / xi={anchor}',color=colors[k])
  ax.axvline(exit_t,alpha=.3,ls='--',color=colors[k])
 ax.set_xlabel('Local elapsed xi');ax.set_ylabel(ylab);ax.grid(alpha=.2);ax.legend(fontsize=8)
 ax.set_title('Q550 v18.37 — diagnostic candidate only; dashed: local support exit')
 fig.tight_layout();fig.savefig(R/'results'/fn,dpi=160);plt.close(fig)
print('V18_37_PLOTS_COMPLETE')
