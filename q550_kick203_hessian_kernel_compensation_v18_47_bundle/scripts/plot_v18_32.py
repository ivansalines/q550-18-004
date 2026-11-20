#!/usr/bin/env python3
"""Plot held-out transfer defects, without calling them a material certificate."""
import sys,json,matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
proof=json.loads((ROOT/'results/original_invariant_manifold_proof_v18_32.json').read_text())
fig,ax=plt.subplots(figsize=(9,5))
for d in proof['finalists']:
    a=d['cross_posture'];xx=[v['xi'] for v in a];yy=[v['relative_residual'] for v in a]
    ax.semilogy(xx,np.maximum(yy,1e-16),marker='o',label=d['id'])
ax.axhline(proof['numerical_acceptance_relative'],ls=':',color='gray',label='Numerical screening threshold')
ax.set(xlabel='Reference xi (held-out transverse states)',ylabel='Relative cofactor residual',
       title='Q550 v18.32 — phase-covariant local hypersurface diagnostic')
ax.grid(alpha=.3);ax.legend(fontsize=7,loc='best');fig.tight_layout()
fig.savefig(ROOT/'results/original_manifold_transfer_v18_32.png',dpi=165)
plt.close(fig)
print('PLOT_V18_32_DONE')
