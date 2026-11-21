"""Reproducible per-trajectory numerical observer alignment and source-active rank charts."""
from pathlib import Path
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1]
with (R/'results/source_active_continuation.csv').open() as f: rows=list(csv.DictReader(f))
for famtype,xi_crest,name in [('phaseblind',3.575,'phaseblind'),('phasecov',4.25,'phasecov')]:
    fig,ax=plt.subplots(figsize=(9,4.8))
    for kind in ('original','autonomous'):
        group=[r for r in rows if famtype in r['family'] and int(r['branch'])==0 and r['kappa']==kind and float(r['cutoff'])==1e-6]
        group.sort(key=lambda v:float(v['xi']))
        ax.plot([float(x['xi']) for x in group],[float(x['active_R']) for x in group],label=kind)
    ax.axvline(xi_crest,ls=':',color='gray',lw=1.2)
    ax.set_yscale('log');ax.set_xlabel('ξ (historical phase section)');ax.set_ylabel('Source-active ratio R')
    ax.set_title(name+' — source-active near-null family (cutoff 1e-6)')
    ax.legend();ax.grid(alpha=.2);fig.tight_layout()
    fig.savefig(R/'results'/f'{name}_near_null_continuation.png',dpi=160)
    plt.close(fig)
