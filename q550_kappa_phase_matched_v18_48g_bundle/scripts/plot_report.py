#!/usr/bin/env python3
"""Plot exploratory phaseblind interpolation: alternatives remain explicitly separate."""
import csv,math
from pathlib import Path
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
folder=ROOT/'results'
if not (folder/'interpolation_controls.csv').is_file():folder=ROOT/'preflight'
with (folder/'interpolation_controls.csv').open(newline='') as f:
 rows=list(csv.DictReader(f))
fig,ax=plt.subplots(figsize=(8,4.8))
for strategy in ('identity','swap'):
 for track in (0,1):
  rr=sorted([r for r in rows if r['family'].startswith('phaseblind') and float(r['xi_start'])==3.5 and
        float(r['xi_end'])==4.0 and r['matching']==strategy and int(r['track'])==track],
        key=lambda r:float(r['xi_interpolated']))
  ax.plot([float(r['xi_interpolated']) for r in rr],
          [math.log(float(r['reference_C'])) for r in rr],
          linestyle='-' if strategy=='identity' else '--',marker='o',markersize=3,
          label=f'{strategy} / track {track}')
ax.set(xlabel='Interpolated ξ (not an authenticated orbit)',ylabel='log projected cancellation ratio',
 title='Q550 v18.48g — phaseblind 3.5 to 4.0, two matching hypotheses')
ax.grid(True,alpha=.3);ax.legend(fontsize=8)
fig.tight_layout();path=folder/'phaseblind_branch_matching_hypotheses.png';fig.savefig(path,dpi=170)
print(path)
