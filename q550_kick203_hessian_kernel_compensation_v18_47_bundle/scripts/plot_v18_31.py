#!/usr/bin/env python3
"""Diagnostic singular spectra; absence of candidates is an admissible result."""
from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[1]
p=json.loads((root/'results/original_invariant_discovery_proof_v18_31.json').read_text())
fig,ax=plt.subplots(figsize=(8.5,5))
for x in p['discovery']:
    s=np.asarray(x['train_singular_values'],float)
    ax.semilogy(np.arange(1,len(s)+1),s/max(s[0],1e-300),marker='.',label=f"Degree {x['degree']} ({x['feature_count']} features)")
ax.axhline(1e-7,color='gray',ls='--',alpha=.6,label='Strict candidate test scale (reference)')
ax.set_xlabel('Singular-value index');ax.set_ylabel('Singular value / largest singular value')
ax.set_title('Q550 invariant-candidate dictionary — train only')
ax.grid(alpha=.3);ax.legend();fig.tight_layout()
fig.savefig(root/'results/original_invariant_singular_spectrum_v18_31.png',dpi=160);plt.close(fig)
