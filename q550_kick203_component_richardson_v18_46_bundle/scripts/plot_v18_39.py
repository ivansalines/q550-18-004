#!/usr/bin/env python3
"""Visualization only from ORIGINAL Q550 v18.39 server outputs (not synthetic)."""
from pathlib import Path
import csv,json,numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1];O=R/'results'
def read(name):return list(csv.DictReader((O/name).open()))
p=json.loads((O/'original_v18_39_proof.json').read_text())
if p.get('status')!='ORIGINAL_Q550_FULL_TANGENT_SOURCE_ROTATION_COMPLETED':raise RuntimeError('Missing genuine original server proof')
fits=read('original_v18_39_vector_fits.csv');summ=read('original_v18_39_family_summary.csv');rot=read('original_v18_39_frame_transport.csv')
fs=[x for x in fits if x['moment']=='common_inside' and x['observable']=='full9d_source_rate']
ss=[x for x in summ if x['moment']=='common_inside']
assert len(fs)==len(ss)==len(rot)==12
labels=[f"{x['anchor_xi']}/{x['branch']}/"+('phase' if x['id'].startswith('phasecov') else 'blind') for x in fs]
fig,ax=plt.subplots(figsize=(11,5))
idx=np.arange(len(fs));a=np.array([float(x['holdout_relative_error']) for x in fs]);b=np.array([float(x['holdout_relative_error']) for x in fs])
ax.plot(idx,a,'o-',label='Full 9D transverse source')
# Prior v18.38 old scalar heldout is loaded from pinned original archive's original fits
from sys import path
path.insert(0,str(R/'scripts'))
from checkpoint_io_v18_39 import read_parent_v39
_,_,_,oldfits,_=read_parent_v39()
old={(x['case_id'],x['moment'],x['observable']):float(x['holdout_relative_error']) for x in oldfits}
c=np.array([old[(x['case_id'],'common_inside','signed_tilt_rate')] for x in fs])
ax.plot(idx,c,'s--',label='Old one-direction projected source')
ax.set_xticks(idx,labels,rotation=70,ha='right');ax.set_ylabel('Held-out relative error');ax.set_title('v18.39 — full source vs single signed projection');ax.grid(alpha=.25);ax.legend();fig.tight_layout();fig.savefig(O/'original_v18_39_full_vs_scalar_holdout.png',dpi=165);plt.close(fig)
fig,ax=plt.subplots(figsize=(11,5))
angles=np.array([float(x['full_odd_source_transport_turn_deg']) if x['full_odd_source_transport_turn_deg'] else np.nan for x in rot])
ax.plot(idx,angles,'o-',label='Full 9D odd source turning (degrees)')
flips=[i for i,x in enumerate(rot) if x['old_projected_sign_reversal']=='True']
for i in flips:ax.axvspan(i-.25,i+.25,alpha=.15)
ax.set_xticks(idx,labels,rotation=70,ha='right');ax.set_ylabel('Moving-frame angle (degrees)');ax.set_title('v18.39 — full-vector rotation; shaded: old projection sign reversals');ax.grid(alpha=.25);fig.tight_layout();fig.savefig(O/'original_v18_39_full_source_rotation.png',dpi=165);plt.close(fig)
fig,ax=plt.subplots(figsize=(11,5))
ax.plot(idx,[float(x['full_source_median_observer_fraction']) for x in ss],'o-',label='Fraction seen by old observer')
ax.plot(idx,[float(x['full_source_level_coherence']) if x['full_source_level_coherence'] else np.nan for x in ss],'s--',label='Centered full-vector/level association')
ax.set_xticks(idx,labels,rotation=70,ha='right');ax.set_ylim(-.05,1.1);ax.set_ylabel('Diagnostic fraction (0–1)');ax.set_title('v18.39 — observer information fraction and coupled variation');ax.grid(alpha=.25);ax.legend();fig.tight_layout();fig.savefig(O/'original_v18_39_observer_capture.png',dpi=165);plt.close(fig)
print('ORIGINAL_V18_39_PLOTS_COMPLETED')
