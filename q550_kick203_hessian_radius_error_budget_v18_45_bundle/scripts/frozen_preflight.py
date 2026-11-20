"""Archived 11D Jacobian/source preflight only: no independent ODE evolution."""
from pathlib import Path
import json,sys,numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'reference')]
from checkpoint_io_v18_19 import read
from pair_math import orth_rows,chosen_seed

def basis(d,xi=4.):
    j=int(np.argmin(abs(d['xi_nodes']-xi)))
    if abs(d['xi_nodes'][j]-xi)>1e-10:raise RuntimeError('Missing matching checkpoint')
    xyz=np.column_stack((d['old_source_material_11D'][j,:10],
        d['quadratic_total_material_11D'][j,:10],d['source_integrand_11D'][j,:10]))
    return orth_rows(xyz)

def observers(d):
    return {'material3':basis(d),'raw4':np.eye(10)[:4], 'raw6':np.eye(10)[:6]}

def main():
    data,prov=read();d=data['tight']; j=int(round(4/.0125));A=d['J_full11D'][j,:10,:10]
    obs=observers(d)
    modes=[('material3','jet',False),('material3','jet',True),('material3','state_only',False),
           ('raw4','jet',False),('raw4','state_only',False),('raw6','state_only',False)]
    rows=[]
    for key,kind,second in modes:
        v,diag=chosen_seed(obs[key],A,kind,second)
        rows.append({'observer':key,'kind':kind,'second':second,'seed_10D':v.tolist(),**diag})
        if diag['state_match_defect']>1e-9:raise RuntimeError('Finite state match rejected')
        if kind=='jet' and diag['jet_match_defect']>1e-8:raise RuntimeError('Jet match rejected')
    z={'version':'v18.19','status':'FROZEN_PREFLIGHT_ONLY_NO_NONLINEAR_SERVER_RESULT',
      'xi_match':4.,'source_tolerances_available':list(data),'s0':d['s0'],'slope':d['slope'],
      'original_server_chain':prov,'modes':rows,
      'phase_row_archive_norm':float(np.linalg.norm(d['J_full11D'][j,10,:])),
      'limitations':['Selected projections and jet-matched histories are observer-dependent',
        'Exact reduced matching is constructed; independent nonlinear ODE integration is not run in preflight',
        'No material invariant/energy metric supplied; off-constraint experiment']}
    out=ROOT/'results';out.mkdir(exist_ok=True)
    (out/'frozen_preflight_summary.json').write_text(json.dumps(z,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':z['status'],'modes':len(rows),
        'largest_jet_defect':max(q['jet_match_defect'] for q in rows if q['kind']=='jet'),
        'input_server_sha256':prov['server_v18_18_sha256']},indent=2))
if __name__=='__main__':main()
