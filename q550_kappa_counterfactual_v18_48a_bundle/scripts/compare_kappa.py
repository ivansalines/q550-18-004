#!/usr/bin/env python3
"""Q550 pinned-RHS local kappa counterfactual; no physical identity claims."""
import argparse, hashlib, importlib.util, json, math, sys
from pathlib import Path
import numpy as np

def load_module(path, name):
    spec=importlib.util.spec_from_file_location(name,str(path)); m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m; spec.loader.exec_module(m); return m

def relative(a,b):
    return float(np.linalg.norm(a-b)/max(np.linalg.norm(a),np.linalg.norm(b),1e-30))

def mixed(f,x,u,v,h):
    # central crossed difference, vector-valued; no claim to be exact Hessian
    return (f(x+h*u+h*v)-f(x+h*u-h*v)-f(x-h*u+h*v)+f(x-h*u-h*v))/(4*h*h)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--v47-root',type=Path,required=True,help='Extracted original v18.47 bundle root')
    p.add_argument('--base-root',type=Path,default=Path('/root/elab1/hysteresis'))
    p.add_argument('--output',type=Path,default=Path('results/kappa_comparison.json'))
    p.add_argument('--h',type=float,nargs='+',default=[0.004,0.008,0.012])
    args=p.parse_args()
    root=args.v47_root.resolve();base=args.base_root.resolve()
    if not (root/'scripts/original_material_invariants_v18_31.py').is_file():raise SystemExit('Missing original v18.47 script: '+str(root))
    sys.path[:0]=[str(root/'scripts'),str(root/'reference')]
    from checkpoint_io_v18_31 import read
    from original_material_invariants_v18_31 import select_original_checkpoint
    import q550_state_impulse_response_v13 as original
    checkpoints,*_=read()
    d=select_original_checkpoint(checkpoints)
    hashes={}
    for role,(filename,sha) in original.DEPENDENCIES.items():
        path=base/filename
        if not path.is_file():raise SystemExit(f'Missing original SHA-pinned {role}: {path}')
        actual=hashlib.sha256(path.read_bytes()).hexdigest()
        if actual!=sha:raise SystemExit(f'SHA MISMATCH in {role}: {path}')
        hashes[role]={'path':str(path),'sha256':actual}
    audit=original.load_module(base/original.DEPENDENCIES['audit'][0],'v1848a_audit')
    support=original.load_module(base/original.DEPENDENCIES['support'][0],'v1848a_support')
    params=audit.ModelParameters(gamma_rho=.3)
    slope=float(d['slope']);x=np.asarray(d['source_state_11D'],float)
    if x.shape!=(11,) or not np.all(np.isfinite(x)):raise SystemExit('Malformed original state')
    ref=1/6; geo=35*math.pi*10/(12*550)
    def field(k):return lambda y:slope*np.asarray(support.support_rhs_float(np.asarray(y,dtype=float),kappa0=float(k),params=params,audit=audit),dtype=float)
    fr,fg=field(ref),field(geo)
    # Pairs use common coordinates; the 11th coordinate is the phase, not a material dimension.
    pairs=[(0,0),(0,2),(2,8),(8,8),(0,10)]
    results=[]
    for a,b in pairs:
        u=np.zeros(11);v=np.zeros(11);u[a]=1;v[b]=1
        scans=[]
        for h in args.h:
            ar=mixed(fr,x,u,v,h);ag=mixed(fg,x,u,v,h)
            scans.append(dict(h=h,reference_norm=float(np.linalg.norm(ar)),geometric_norm=float(np.linalg.norm(ag)),relative_change=relative(ar,ag),signed_dot=float(np.dot(ar,ag)),cosine=float(np.dot(ar,ag)/max(np.linalg.norm(ar)*np.linalg.norm(ag),1e-300)),reference_vector=ar.tolist(),geometric_vector=ag.tolist()))
        results.append(dict(pair=[a,b],steps=scans))
    result=dict(version='v18.48a',status='ORIGINAL_Q550_LOCAL_STATE_KAPPA_COUNTERFACTUAL',scope='fixed original source state only; not a restarted trajectory or full v18.47 36-snapshot replay',original_kappa=ref,autonomous_kappa=geo,delta_kappa=geo-ref,relative_kappa_change=(geo-ref)/ref,state_11D=x.tolist(),state_source='original tight checkpoint source_state_11D',original_slope=slope,params={'gamma_rho':0.3},verified_dependencies=hashes,rhs_relative_change=relative(fr(x),fg(x)),rhs_reference=fr(x).tolist(),rhs_geometric=fg(x).tolist(),pairs=results,caveat='Numerical Hessian via central cross difference; coordinate probes are not transported material Hessian directions. No analytic Hessian or structural decomposition is certified.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('status','original_kappa','autonomous_kappa','delta_kappa','relative_kappa_change','rhs_relative_change')},indent=2))
    print('Wrote',args.output)
if __name__=='__main__':main()
