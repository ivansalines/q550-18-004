"""Charts use only saved v18.27 observations; no simulated 'findings'."""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'results'
def read(f):return list(csv.DictReader((R/f).open()))

def main():
    rows=read('original_principal_subspace_v18_27.csv')
    fig,ax=plt.subplots(figsize=(9,5.3))
    for obs in ('material3','raw4','raw6'):
      for k in (10,20):
        r=sorted((r for r in rows if r['observer']==obs and r['noise_mode']=='none' and int(r['rank'])==k),key=lambda r:-float(r['amplitude']))
        x=np.array([float(t['amplitude']) for t in r]);y=np.array([float(t['max_principal_angle_deg']) for t in r])
        ax.plot(x,y,marker='o',ls='-' if k==20 else '--',label=f'{obs} | rank {k}')
    ax.set_xscale('log');ax.invert_xaxis();ax.set_xlabel('Perturbation amplitude ε (decreasing to the right)')
    ax.set_ylabel('Largest principal angle vs basis at 1e-5 [degrees]')
    ax.set_title('Q550 v18.27 | Frozen reference vs refitted singular subspaces')
    ax.grid(True,alpha=.2);ax.legend(fontsize=8,ncol=2)
    fig.tight_layout();fig.savefig(R/'original_subspace_stability_v18_27.png',dpi=160);plt.close(fig)
    rows=read('original_singular_spectrum_v18_27.csv')
    fig,ax=plt.subplots(figsize=(9,5))
    for obs in ('material3','raw4','raw6'):
      rr=sorted((r for r in rows if r['observer']==obs and r['noise_mode']=='none'),key=lambda r:-float(r['amplitude']))
      x=[float(r['amplitude']) for r in rr];v=[float(r['s10_over_s11']) for r in rr]
      ax.plot(x,v,'-o',label=obs)
    ax.set_yscale('log');ax.set_xscale('log');ax.invert_xaxis()
    ax.set_xlabel('Perturbation amplitude ε')
    ax.set_ylabel('Singular gap s₁₀ / s₁₁')
    ax.set_title('Q550 v18.27 | How sharply the first ten directions separate')
    ax.grid(True,alpha=.2);ax.legend();fig.tight_layout()
    fig.savefig(R/'original_singular_gap_v18_27.png',dpi=160);plt.close(fig)
    if (R/'original_tight_floor_comparison_v18_27.csv').exists():
      rows=read('original_tight_floor_comparison_v18_27.csv')
      fig,ax=plt.subplots(figsize=(9,5))
      rr=sorted((r for r in rows if r['observer']=='material3' and float(r['anchor'])==4.),key=lambda r:-float(r['amplitude']))
      x=[float(r['amplitude']) for r in rr]
      ax.plot(x,[float(r['tight_even_rms']) for r in rr],'-o',label='Tight normalized even response')
      ax.plot(x,[float(r['even_inter_tier_rms_disagreement']) for r in rr],'-o',label='Reference vs tight disagreement')
      ax.set_xscale('log');ax.set_yscale('log');ax.invert_xaxis()
      ax.set_xlabel('Amplitude ε');ax.set_ylabel('Normalized RMS')
      ax.set_title('Q550 v18.27 | Empirical convergence check (not an error bound)')
      ax.grid(True,alpha=.2);ax.legend();fig.tight_layout()
      fig.savefig(R/'original_tight_floor_v18_27.png',dpi=160);plt.close(fig)
    print('v18.27 PLOTS OK')
if __name__=='__main__':main()
