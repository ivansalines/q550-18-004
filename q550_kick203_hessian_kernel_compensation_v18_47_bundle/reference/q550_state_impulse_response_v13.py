#!/usr/bin/env python3
"""Q550 v13: original support-state nonlinear paired impulses + tangent audit.

IMPORTANT: This measures the original support-model 11D *state* response, not the
window-covariance perturbation used in v12. Without an explicit invariant
callback the kicks are labeled OFF_CONSTRAINT_DIAGNOSTIC and no physical
constraint-preservation conclusion can be drawn. Never relabel the causal
state-window shape proxy as the v11 dlocal covariance shape.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.util
import json
import os
import platform
from pathlib import Path
import sys
import zipfile

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

VERSION = "2026-10-13-Q550-original-support-state-impulses-v13"
SOURCE_SHA = "09702bc06c5c927ee7656a0fcd510ffddbabdccbe5733ecc1e64f1acd67f8f0d"
DEFAULT_BASE = Path("/root/elab1/hysteresis/q550_full11D_transverse_slow_state_dense_sampling_v1_bundle")
DEFAULT_SOURCE = Path("/root/elab1/hysteresis/q550_x55_multiloop_transport_runner_v1_1_bundle/q550_x55_multiloop_transport_x50000_rebased_checked_v1/q550_x55_multiloop_transport_law_v2_bundle.zip")
DEPENDENCIES = {
 "audit": ("test_support_mobius_monodromy_audit.py", "90792fc9ad144333df521c7c0862193125c4702319fc3aa1a2bcef45491b2864"),
 "support": ("test_support_q550_support_ode_interval_residual_pilot_v1.py", "5b5938e7920b2ecd242fb8f416a18c7c47ae10aa06e027e0727f15e284c98c61"),
 "tube": ("test_support_q550_exact_nonlinear_local_tube_v2_1.py", "a9d1ec342c268da8e721dc641cab8cc4cf39f69d821dfd4191163833ba04e881"),
 "orbit": ("q_550_orbit.npz", "537f6326b841f36f963be3736b1034724041994a07f9ac9e5377de6c99048f34"),
 "events": ("inputs/presence_metrology_events.csv", "bc18df957b9dc311978584132b9cf0690e30a9576a306bda6f189a4ba33df892"),
 "shared": ("support_q550_shared_generator_arc_cocycle_cell49_v1_pilot.json", "be9ad38fc157073efeb95890588ead7ff47e2a8a7e381bba1cdea3d5682567aa"),
}

def hash_file(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()

def load_module(p,name):
    sp=importlib.util.spec_from_file_location(name,str(p))
    if sp is None or sp.loader is None:raise RuntimeError(f"cannot import {p}")
    mod=importlib.util.module_from_spec(sp);sys.modules[name]=mod;sp.loader.exec_module(mod)
    return mod

def clean_unit(v, tol=1e-12):
    v=np.asarray(v,float).reshape(-1);s=float(np.linalg.norm(v))
    if not np.isfinite(s) or s<tol:raise ValueError("degenerate kick vector")
    return v/s

def numeric_constraint_jacobian(fun,y):
    y=np.asarray(y,float); f=np.asarray(fun(y),float).reshape(-1)
    J=np.zeros((len(f),len(y)))
    for j in range(len(y)):
        h=1e-6*max(1.0,abs(y[j]));e=np.zeros_like(y);e[j]=h
        J[:,j]=(np.asarray(fun(y+e),float).reshape(-1)-np.asarray(fun(y-e),float).reshape(-1))/(2*h)
    return J

def invariant_project(v,y,fun):
    if fun is None:return clean_unit(v)
    J=numeric_constraint_jacobian(fun,y)
    if not np.all(np.isfinite(J)):raise RuntimeError("nonfinite invariant derivative")
    U,s,Vh=np.linalg.svd(J,full_matrices=True)
    rank=int(np.sum(s > max(1e-11,float(s[0])*1e-8))) if len(s) else 0
    if rank>=len(y):raise RuntimeError("no invariant-tangent dimensions remain")
    N=Vh[rank:].T;projected=N@(N.T@v)
    if np.linalg.norm(projected)<1e-8*max(1e-12,np.linalg.norm(v)):
        raise RuntimeError("kick direction nearly orthogonal to constraint tangent")
    return clean_unit(projected)

def retract_invariant(y_trial, ref_values, fun, rtol=5e-10, maxit=16):
    if fun is None:return np.asarray(y_trial,float)
    x=np.asarray(y_trial,float).copy();ref=np.asarray(ref_values,float).reshape(-1)
    for it in range(maxit):
        err=np.asarray(fun(x),float).reshape(-1)-ref
        if max(abs(err),default=0.)<=rtol*max(1.,np.max(abs(ref),initial=0.)):return x
        J=numeric_constraint_jacobian(fun,x)
        dx=-np.linalg.pinv(J,rcond=1e-11)@err
        if np.linalg.norm(dx)>1e-2*max(1.,np.linalg.norm(x)):
            raise RuntimeError("constraint retraction too large; invalid model/epsilon")
        x+=dx
    raise RuntimeError("invariant retraction did not converge")

def causal_covariance(states, i, window_pts):
    """Trailing Hann window of *support states*. NOT v11 dlocal covariance."""
    if i+1<window_pts:return np.nan,np.nan,np.nan
    X=states[i+1-window_pts:i+1]
    w=np.hanning(window_pts)
    w=w/w.sum(); mean=np.einsum('n,nd->d',w,X)
    z=X-mean; C=np.einsum('n,ni,nj->ij',w,z,z)
    ev=np.linalg.eigvalsh(C)[::-1]
    gap=float((ev[1]-ev[2])/max(ev[1],1e-50))
    if ev[1]<=1e-24*max(ev[0],1e-300) or gap<1e-4:return np.nan,np.nan,gap
    return 0.5*np.log(float(ev[0]/ev[1])),np.sqrt(float(ev[0]*ev[1])),gap

def classify_gain(gains, signed, linear_error=0.):
    """Descriptive rules only, not a nonlinear stability theorem."""
    a=np.asarray(gains,float);n=len(a)
    if n<9 or not np.all(np.isfinite(a)) or linear_error>0.25:return "inconclusive"
    tail=float(np.median(a[-max(4,n//6):])); peak=float(np.max(a))
    if peak>1.25 and tail<0.75 and np.any(np.sign(signed[1:])!=np.sign(signed[:-1])):
        return "overshoot_and_reabsorption_candidate"
    if tail>1.5 and np.median(a[n//2:])>1.25:return "amplification_candidate"
    if tail<0.5 and np.median(a[n//2:])<0.75:return "decay_candidate"
    return "mixed_or_neutral"

def validate_inputs(base,source):
    files={}
    for role,(name,h) in DEPENDENCIES.items():
        p=base/name
        if not p.is_file():raise FileNotFoundError(f"missing original Q550 {role}: {p}")
        got=hash_file(p)
        if got!=h:raise RuntimeError(f"Q550 {role} SHA mismatch: {got} != {h}")
        files[role]={"path":str(p),"sha256":got}
    if not source.is_file():raise FileNotFoundError(f"missing x500 source {source}")
    got=hash_file(source)
    if got!=SOURCE_SHA:raise RuntimeError(f"Q550 x500 ZIP mismatch: {got} != {SOURCE_SHA}")
    files["x500"]={"path":str(source),"sha256":got}
    return files

def event_section_map(base):
    events=pd.read_csv(base/DEPENDENCIES['events'][0]).set_index('event_n')
    s390=float(events.loc[390,'support_center'])
    s433=float(events.loc[433,'support_center'])
    slope=s433-s390
    if slope<=0:raise RuntimeError("xi->sA must be positively oriented")
    return s390,slope

def driver_samples(source, probes):
    """Stream only five rows, not the 0.5GB original CSV."""
    targets=set(round(float(x),5) for x in probes)
    res={}
    with zipfile.ZipFile(source) as z:
        matches=[n for n in z.namelist() if n.endswith('/inputs/local_y2_driver_hhalf_extended.csv')]
        if len(matches)!=1:raise RuntimeError("x500 bundle missing unique hhalf driver")
        import io
        with z.open(matches[0]) as raw:
            df=pd.read_csv(raw,usecols=['xi','sA','central_v2','central_z'],chunksize=100000)
            for part in df:
                for _,row in part.iterrows():
                    k=round(float(row['xi']),5)
                    if k in targets and k not in res:res[k]={c:float(row[c]) for c in ['xi','sA','central_v2','central_z']}
                if len(res)==len(targets):break
    if len(res)!=len(targets):raise RuntimeError(f"missing original x500 reference xi: {sorted(targets-set(res))}")
    return res

def shape_directions(npz,kick):
    with np.load(npz,allow_pickle=False) as d:
        x=d['hann_matched__hhalf__centers_xi']
        ii=int(np.argmin(abs(x-kick)))
        if abs(x[ii]-kick)>1e-9:raise RuntimeError(f"missing v11 shape covariance at xi {kick}")
        C=d['hann_matched__hhalf__covariance_11d'][ii]
    val,U=np.linalg.eigh(C)
    if val[-2]<=0 or (val[-2]-val[-3])/val[-2]<1e-3:
        raise RuntimeError("Q550 shape basis is not spectrally separated")
    a,b=U[:,-1],U[:,-2]
    return {'mode_plus':clean_unit(a+b),'mode_minus':clean_unit(a-b)}

def integrate(fun,s0,s1,y0,rtol,atol,max_step=1/720):
    z=solve_ivp(fun,(float(s0),float(s1)),np.asarray(y0,float),method='DOP853',
                rtol=rtol,atol=atol,max_step=max_step,dense_output=True)
    if not z.success or z.sol is None:raise RuntimeError(f"ODE solve failed: {z.message}")
    return z

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--base-root',type=Path,default=DEFAULT_BASE)
    ap.add_argument('--source-bundle-zip',type=Path,default=DEFAULT_SOURCE)
    ap.add_argument('--covariances',type=Path,required=True)
    ap.add_argument('--constraints-module',type=Path,default=None,help='Python module exposing invariants(y)->1D array; needs model-specific invariants')
    ap.add_argument('--allow-unconstrained',action='store_true',help='Explicitly authorize off-constraint dynamical diagnostic (NOT physical constrained stability)')
    ap.add_argument('--kicks',nargs='+',type=float,default=[190.,203.,212.75,219.,225.5])
    ap.add_argument('--eps',nargs='+',type=float,default=[1e-7,3e-7,1e-6])
    ap.add_argument('--horizon-xi',type=float,default=24.)
    ap.add_argument('--sample-step-xi',type=float,default=.25)
    ap.add_argument('--window-xi',type=float,default=4.)
    ap.add_argument('--rtol',type=float,default=1e-12)
    ap.add_argument('--atol',type=float,default=1e-14)
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args()
    if not a.constraints_module and not a.allow_unconstrained:
        ap.error('No material/physical constraint has been specified. Supply --constraints-module OR explicitly --allow-unconstrained (diagnostic only).')
    if a.horizon_xi<=0 or a.sample_step_xi<=0 or a.window_xi<=a.sample_step_xi:ap.error('invalid horizon or window')
    if any((x<188 or x>238) for x in a.kicks):ap.error('kicks must lie in archived v11 covariance xi=188..238')
    a.out.mkdir(parents=True,exist_ok=True)
    files=validate_inputs(a.base_root,a.source_bundle_zip)
    q550=load_module(a.base_root/DEPENDENCIES['audit'][0],'q550_v13_audit')
    support=load_module(a.base_root/DEPENDENCIES['support'][0],'q550_v13_support')
    tube=load_module(a.base_root/DEPENDENCIES['tube'][0],'q550_v13_tube')
    params=q550.ModelParameters(gamma_rho=0.3)
    kappa0=(55.0/6.0)*(10.0/550.0)
    s390,slope=event_section_map(a.base_root)
    original=driver_samples(a.source_bundle_zip,a.kicks)
    kick_s={float(k):s390+slope*float(k) for k in a.kicks}
    def rhs(s,y):return np.asarray(support.support_rhs_float(np.asarray(y,float),kappa0=kappa0,params=params,audit=q550),float)
    def jac(s,y):return np.asarray(tube.float_support_jacobian(np.asarray(y,float),kappa0=kappa0,params=params,audit=q550,base=support),float)
    orb=q550.load_orbit(550.0,a.base_root/DEPENDENCIES['orbit'][0])
    tail=q550.collect_uniform_tail(orb,tail_periods=70,steps_per_period=160,samples_per_rotation=720,
                                     initial_x_perturbation=1e-6,params=params,progress_every=1000000)
    arch=q550.build_state_spline(tail)
    seed=max(float(tail.s[0]),float(tail.s[-1])-0.5)
    end=max(kick_s.values())+a.horizon_xi*slope+0.01
    if seed>=min(kick_s.values()):raise RuntimeError('Orbit seed after impulse start; original replay is unusable')
    baseline=integrate(rhs,seed,end,np.asarray(arch(seed),float),a.rtol,a.atol)
    quality=[]
    for x,s in kick_s.items():
        y=np.asarray(baseline.sol(s),float).reshape(11)
        original_row=original[round(x,5)]
        s_err=abs(s-original_row['sA'])
        state_err=abs(y[5]-original_row['central_v2'])/max(1.,abs(original_row['central_v2']))
        quality.append({'kick_xi':x,'sA_error':s_err,'central_v2_relative_error':state_err})
        if s_err>1e-8 or state_err>3e-4:
            raise RuntimeError(f'Unperturbed x500 replay does not match original at xi {x}: {quality[-1]}')
    invariant_fun=None
    if a.constraints_module:
        plugin=load_module(a.constraints_module,'q550_v13_invariant_plugin')
        if not callable(getattr(plugin,'invariants',None)):
            raise RuntimeError('constraints-module must expose callable invariants(y)')
        invariant_fun=plugin.invariants
    status='INVARIANT_RETRACTED_WITH_DECLARED_PLUGIN' if invariant_fun else 'OFF_CONSTRAINT_DIAGNOSTIC'
    rows=[];summary=[];invariant_diag=[]
    for kick in a.kicks:
        s0=kick_s[float(kick)]
        y0=np.asarray(baseline.sol(s0),float).reshape(11)
        seeds=shape_directions(a.covariances,float(kick))
        # Autonomous-flow perturbation is the neutral timing direction control.
        seeds['flow_control']=clean_unit(rhs(s0,y0))
        grid_x=kick+np.arange(round(a.horizon_xi/a.sample_step_xi)+1)*a.sample_step_xi
        grid_s=s390+slope*grid_x
        ybase=np.asarray(baseline.sol(grid_s),float).T
        win_pts=round(a.window_xi/a.sample_step_xi)+1
        if win_pts%2==0:win_pts+=1
        bshape=np.asarray([causal_covariance(ybase,i,win_pts) for i in range(len(grid_s))])
        for direction,raw in seeds.items():
            if invariant_fun is not None:
                v=invariant_project(raw,y0,invariant_fun)
            elif direction=='flow_control':
                v=clean_unit(raw)
            else:
                # Fix flow-phase gauge but DO NOT pretend to preserve invariants.
                f=clean_unit(rhs(s0,y0));v=clean_unit(raw-f*np.dot(f,raw))
            # Verify original tangent Jacobian locally against the nonlinear RHS.
            h=3e-7*max(1.0,np.linalg.norm(y0))
            fd=(rhs(s0,y0+h*v)-rhs(s0,y0-h*v))/(2*h)
            jv=jac(s0,y0)@v
            jac_err=float(np.linalg.norm(fd-jv)/max(1e-10,np.linalg.norm(fd),np.linalg.norm(jv)))
            if jac_err>2e-3:raise RuntimeError(f'original tangent Jacobian check fails at xi={kick} dir={direction}: {jac_err}')
            tangent=integrate(lambda s,z:jac(s,np.asarray(baseline.sol(s),float).reshape(11))@z,
                              s0,grid_s[-1],v,a.rtol,a.atol)
            lin=np.asarray(tangent.sol(grid_s),float).T
            invariant_ref=np.asarray(invariant_fun(y0),float).ravel() if invariant_fun else None
            amp_scale=max(1.,np.linalg.norm(y0))
            for eps in a.eps:
                amp=float(eps)*amp_scale
                pair={}
                for sign in (-1,1):
                    ytrial=y0+sign*amp*v
                    yp0=retract_invariant(ytrial,invariant_ref,invariant_fun)
                    sol=integrate(rhs,s0,grid_s[-1],yp0,a.rtol,a.atol)
                    yp=np.asarray(sol.sol(grid_s),float).T
                    err=yp-ybase
                    pair[sign]=err
                    gain=np.linalg.norm(err,axis=1)/max(np.linalg.norm(yp0-y0),1e-300)
                    signed=err@v/amp
                    sh=np.asarray([causal_covariance(yp,i,win_pts) for i in range(len(grid_s))])
                    if invariant_fun:
                        dev=[float(np.max(np.abs(np.asarray(invariant_fun(yi),float).ravel()-invariant_ref))) for yi in yp]
                        max_inv=max(dev)
                        if max_inv>2e-7*max(1.,np.max(abs(invariant_ref),initial=0.)):
                            raise RuntimeError(f'declared constraint drift exceeds tolerance: {max_inv}')
                    else:max_inv=float('nan')
                    invariant_diag.append({'kick_xi':kick,'direction':direction,'eps_rel':eps,'sign':sign,'max_constraint_drift':max_inv})
                    for i,x in enumerate(grid_x):
                        rows.append({'kick_xi':kick,'direction':direction,'eps_rel':eps,'sign':sign,
                                     'xi':x,'delta_xi':x-kick,'delta_state_norm':float(np.linalg.norm(err[i])),
                                     'full_state_gain':gain[i],'signed_seed_projection':signed[i],
                                     'statewindow_eta':sh[i,0],
                                     'delta_statewindow_eta':sh[i,0]-bshape[i,0],
                                     'statewindow_area_proxy':sh[i,1],
                                     'statewindow_mode_gap':sh[i,2],
                                     'constraint_status':status})
                    pair[('shape',sign)]=sh[:,0]-bshape[:,0]
                    pair[('gain',sign)]=gain
                    pair[('signed',sign)]=signed
                # Symmetric derivative must match the variational equation in linear regime.
                odd=(pair[1]-pair[-1])/(2*amp)
                even=(pair[1]+pair[-1])/(2*amp)
                denom=np.maximum(np.linalg.norm(lin,axis=1),1e-12)
                linear_error=np.linalg.norm(odd-lin,axis=1)/denom
                even_error=np.linalg.norm(even,axis=1)/denom
                gp=.5*(pair[('gain',1)]+pair[('gain',-1)])
                signed=.5*(pair[('signed',1)]-pair[('signed',-1)])
                decision=classify_gain(gp,signed,float(np.median(linear_error[-8:])))
                shape_odd=0.5*(pair[('shape',1)]-pair[('shape',-1)])
                finite=shape_odd[np.isfinite(shape_odd)]
                shape_info={'shape_odd_peak_abs':float(np.max(abs(finite))) if len(finite) else float('nan'),
                            'shape_odd_final':float(finite[-1]) if len(finite) else float('nan'),
                            'shape_odd_valid_points':int(len(finite)),
                            'shape_odd_sign_changes':int(np.sum(np.signbit(finite[1:]) != np.signbit(finite[:-1]))) if len(finite)>1 else 0}
                summary.append({'kick_xi':kick,'direction':direction,'eps_rel':eps,
                                'constraint_status':status,'classification':decision,
                                'peak_gain':float(np.max(gp)),'end_gain':float(gp[-1]),
                                'linear_end_relative_error':float(linear_error[-1]),
                                'linear_tail_median_error':float(np.median(linear_error[-8:])),
                                'nonlinear_even_tail_median':float(np.median(even_error[-8:])),
                                'baseline_mode_gap_min':float(np.nanmin(bshape[:,2])),
                                'jacobian_direction_rel_error':jac_err,**shape_info})
                print(f'xi={kick:g} dir={direction:12s} eps={eps:g} gain_end={gp[-1]:.6g} '
                      f'linear_err={linear_error[-1]:.3g} {decision}',flush=True)
    pd.DataFrame(rows).to_csv(a.out/'impulse_time_series_v13.csv',index=False)
    pd.DataFrame(summary).to_csv(a.out/'impulse_decisions_v13.csv',index=False)
    pd.DataFrame(quality).to_csv(a.out/'x500_replay_regression_v13.csv',index=False)
    pd.DataFrame(invariant_diag).to_csv(a.out/'invariant_checks_v13.csv',index=False)
    meta={'version':VERSION,'status':status,'not_v11_shape':True,
          'scientific_limitations':[
             'Support-state causal-window covariance is NOT dlocal PCA or its v11 eta.',
             'No constraint invariants are known unless supplied as a model-specific plugin.',
             'xi is an affine observation coordinate mapped onto support s, not established physical time.',
             'Amplitude variations and paired linearization must be checked before classifying stability.',
             'No physical material torsion, inertia flip, or Dzhanibekov mechanism is inferred.' ],
          'parameters':vars(a).__str__(),'source_files':files,
          'python':platform.python_version(),'numpy':np.__version__,
          'replay_regression':quality,'event_mapping':{'s390':s390,'sA_per_xi':slope}}
    (a.out/'report_v13.json').write_text(json.dumps(meta,indent=2,default=str))
    print('Complete: '+str(a.out),flush=True)

if __name__=='__main__':main()
