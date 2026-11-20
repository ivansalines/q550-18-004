#!/usr/bin/env python3
"""Plot authenticated original v18.34 results only, never preflight as original."""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1];O=R/'results'
p=O/'original_v18_34_budget_samples.csv'
if not p.is_file():raise RuntimeError('Original Q550 budget absent')
with p.open() as fh:rows=list(csv.DictReader(fh))
chosen='phasecov_d2_lam-1.00_mod+0.00'
for a,seed in [(3.5,0),(3.5,1),(4.,0),(4.5,0)]:
    rr=[r for r in rows if r['id']==chosen and float(r['seed_anchor'])==a and int(r['seed_index'])==seed]
    if not rr:continue
    t=np.array([float(r['delta_xi']) for r in rr]);direct=np.array([float(r['direct_transport_defect']) for r in rr]);integ=np.array([float(r['integrated_defect_simpson']) for r in rr]);r=np.array([float(r['source_r']) for r in rr]);dist=np.array([float(r['material_displacement_scaled']) for r in rr])
    plt.figure(figsize=(9,5))
    plt.plot(t,direct,label='Direct C - transported C(0)',lw=2)
    plt.plot(t,integ,'--',label='Integrated source defect',lw=1.8)
    plt.xlabel('Delta xi');plt.ylabel('Signed cofactor transport defect')
    plt.title(f'Q550 v18.34 — source-budget closure; anchor {a}, seed {seed}')
    plt.axhline(0,color='gray',lw=.6);plt.legend();plt.grid(alpha=.25)
    plt.tight_layout();plt.savefig(O/f'original_v18_34_budget_closure_xi{str(a).replace(".","p")}_seed{seed}.png',dpi=160);plt.close()
    if a==3.5 and seed==0:
        fig=plt.figure(figsize=(9,5));plt.plot(t,r,label='Local Lie-cofactor defect r(xi)')
        plt.xlabel('Delta xi');plt.ylabel('Source r = dC/dxi - mu C');plt.grid(alpha=.25)
        plt.title('Q550 v18.34 — residual source along transported surface');plt.tight_layout()
        plt.savefig(O/'original_v18_34_source_profile.png',dpi=160);plt.close()
        plt.figure(figsize=(9,5));plt.semilogy(t,np.maximum(dist,1e-12),label='Distance from projected start (diagnostic scaled 10D)')
        for level in (1,3,10):plt.axhline(level,linestyle=':',label=f'{level} scale radii')
        plt.xlabel('Delta xi');plt.ylabel('Scaled material displacement (not a physical metric)')
        plt.title('Q550 v18.34 — when the trajectory leaves its local diagnostic neighborhood')
        plt.grid(alpha=.2);plt.legend();plt.tight_layout()
        plt.savefig(O/'original_v18_34_domain_exit.png',dpi=160);plt.close()
print('ORIGINAL_V18_34_PLOTS_COMPLETE')
