#!/usr/bin/env python3
"""Plot new ORIGINAL Q550 rows only (no synthetic chart passed off as a result)."""
from pathlib import Path
import csv
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1]
def read(file):
 p=R/'results'/file
 if not p.exists():raise RuntimeError('Missing original v18.44 output '+str(p))
 return list(csv.DictReader(p.open()))

def run():
 h=read('original_v18_44_hessian_dual_contractions.csv')
 j=read('original_v18_44_transported_source_jets.csv')
 if (len(h),len(j))!=(324,36):raise RuntimeError('Incomplete original v18.44 original plot inputs')
 all_ids=list(dict.fromkeys(r['case_id'] for r in j))
 fig,ax=plt.subplots(figsize=(10,5.5))
 for n,case in enumerate(all_ids):
  sub=[r for r in h if r['case_id']==case and r['fraction_of_common_time']=='0.5']
  if not sub:sub=[r for r in h if r['case_id']==case and abs(float(r['fraction_of_common_time'])-.5)<1e-12]
  x=sorted(set(float(r['rhs_normalized_h']) for r in sub))
  y=[float(np.median([float(r['relative_rhs_jac_disagreement']) for r in sub if abs(float(r['rhs_normalized_h'])-v)<1e-12])) for v in x]
  ax.plot(x,y,marker='o',alpha=.75,label=case if n in (0,1,2,3) else None)
 ax.set(yscale='log',xlabel='Normalized mixed-RHS Hessian probe displacement',ylabel='Relative difference from pinned-Jacobian directional estimate',title='Q550 v18.44 — Hessian contraction, two numeric pathways (s=0.5)');ax.grid(alpha=.25)
 ax.legend(fontsize=7);fig.tight_layout();fig.savefig(R/'results/original_v18_44_hessian_probe_comparison.png',dpi=160);plt.close(fig)
 fig,ax=plt.subplots(figsize=(10,5.5))
 idx=np.arange(len(all_ids))
 vals=[np.median([float(r['analytic_vs_finite_chart_disagreement']) for r in j if r['case_id']==case]) for case in all_ids]
 vals2=[np.median([float(r['analytic_jac_vs_rhs_jet_disagreement']) for r in j if r['case_id']==case]) for case in all_ids]
 ax.plot(idx,vals,'o-',label='Exact chart differential vs original finite chart')
 ax.plot(idx,vals2,'s-',label='RHS Hessian vs pinned-Jacobian jet')
 ax.set(yscale='log',xlabel='Original 12 diagnostic families',ylabel='Median relative first-jet disagreement',title='Q550 v18.44 — separation of moving chart and Hessian differentiation')
 ax.grid(alpha=.25);ax.legend();fig.tight_layout();fig.savefig(R/'results/original_v18_44_jet_separation.png',dpi=160);plt.close(fig)
 fig,ax=plt.subplots(figsize=(9.5,5))
 for case in all_ids[:4]:
  sub=sorted((r for r in j if r['case_id']==case),key=lambda r:float(r['fraction_of_common_time']))
  ax.plot([float(r['fraction_of_common_time']) for r in sub],[float(r['source_time_jet_norm']) for r in sub],marker='o',label=case)
 ax.set(xlabel='Fraction of authenticated common support time',ylabel='Norm of source differential in 9D (per normalized time)',title='Q550 v18.44 — source-only jet, separated from reference transport')
 ax.grid(alpha=.25);ax.legend(fontsize=7);fig.tight_layout();fig.savefig(R/'results/original_v18_44_source_transport_terms.png',dpi=160);plt.close(fig)
 print('V18_44_ORIGINAL_PLOTS_3')
if __name__=='__main__':run()
