#!/usr/bin/env python3
"""Q550 v18.48: structural RHS/Hessian identity audit. All values are probe diagnostics, not certification."""
import argparse, importlib.util, json, hashlib, sys
from pathlib import Path
import numpy as np

def load(path):
    spec=importlib.util.spec_from_file_location('v18_48_adapter',path)
    m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
    return m

def bilinear(f,x,u,v,h):
    # Centered symmetric mixed difference.
    return (f(x+h*u+h*v)-f(x+h*u-h*v)-f(x-h*u+h*v)+f(x-h*u-h*v))/(4*h*h)

def norm(a):return float(np.linalg.norm(a))
def rel(a,b):return norm(a-b)/max(norm(a),norm(b),1e-14)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--adapter',required=True)
    p.add_argument('--states',required=True,help='NPZ with states: (N,11) and optional directions_u, directions_v: (N,11)')
    p.add_argument('--output',default='outputs/v18_48_report.json')
    p.add_argument('--steps',type=float,nargs='+',default=[1e-3,3e-4,1e-4,3e-5])
    p.add_argument('--tol',type=float,default=1e-5)
    args=p.parse_args()
    a=load(Path(args.adapter).resolve());dat=np.load(args.states)
    xarr=np.asarray(dat['states'],float)
    assert xarr.ndim==2 and xarr.shape[1]==11 and np.all(np.isfinite(xarr))
    comps=a.components
    assert isinstance(comps,dict) and len(comps)>0
    for k,f in comps.items():
        assert callable(f),k
    names=list(comps)
    rng=np.random.default_rng(1848)
    def full(x):return np.asarray(a.rhs(x),float)
    def part(k):return lambda x: np.asarray(comps[k](x),float)
    def summed(x):return sum((part(k)(x) for k in names),np.zeros(11))
    records=[]
    for i,x in enumerate(xarr):
        u=np.asarray(dat['directions_u'][i],float) if 'directions_u' in dat else rng.standard_normal(11)
        v=np.asarray(dat['directions_v'][i],float) if 'directions_v' in dat else rng.standard_normal(11)
        u/=norm(u);v/=norm(v)
        perstep=[]
        for h in args.steps:
            F=full(x);S=summed(x)
            H=bilinear(full,x,u,v,h)
            pieces={k:bilinear(part(k),x,u,v,h) for k in names}
            HS=sum(pieces.values(),np.zeros(11))
            HT=bilinear(full,x,v,u,h)
            piece_norms={k:norm(t) for k,t in pieces.items()}
            amplification=sum(piece_norms.values())/max(norm(HS),1e-14)
            row={'h':h,'rhs_absolute_residual':norm(F-S),'rhs_relative_residual':rel(F,S),
                 'hessian_absolute_residual':norm(H-HS),'hessian_relative_residual':rel(H,HS),
                 'mixed_symmetry_relative_residual':rel(H,HT),'hessian_norm':norm(H),
                 'term_norms':piece_norms,'cancellation_ratio':amplification}
            if hasattr(a,'analytic_bilinear'):
                exact=np.asarray(a.analytic_bilinear(x,u,v),float)
                row['analytic_relative_residual']=rel(H,exact)
            if hasattr(a,'analytic_component_bilinears'):
                an=a.analytic_component_bilinears(x,u,v)
                row['analytic_components_relative_residual']={k:rel(pieces[k],np.asarray(an[k],float)) for k in names}
            perstep.append(row)
        records.append({'sample':i,'steps':perstep})
    last=[q['steps'][-1] for q in records]
    # Numerical second derivative errors need not decrease at smallest h; report entire ladder.
    summary={'version':'v18.48','classification':'NUMERICAL_DIAGNOSTIC_ONLY','adapter':str(Path(args.adapter).resolve()),
             'samples':len(records),'components':names,'step_sizes':args.steps,'requested_tolerance':args.tol,
             'max_rhs_relative_residual':max(r['rhs_relative_residual'] for r in last),
             'max_hessian_relative_residual_at_smallest_step':max(r['hessian_relative_residual'] for r in last),
             'max_mixed_symmetry_relative_residual_at_smallest_step':max(r['mixed_symmetry_relative_residual'] for r in last),
             'largest_cancellation_ratio_at_smallest_step':max(r['cancellation_ratio'] for r in last),
             'records':records,'boundary':'No Q550 claim without an authenticated original-model adapter, parent hash, and server run. Finite differences cannot establish a symbolic identity.'}
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    Path(args.output).write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('records','boundary')},indent=2))
if __name__=='__main__':main()
