#!/usr/bin/env python3
"""Full 11D original-Q550 source Hessian, second variation, and +/- cross-audit.

Only the data producer imports the original pinned Q550 ODE; no regression or
model-blind estimator is performed here. Uses real original nonlinear RHS,
independent variational ODE and independent paired nonlinear trajectories.
An empirical change-of-stencil indicator is not a rigorous error enclosure.
All state perturbations are OFF_CONSTRAINT_DIAGNOSTIC.
"""
from __future__ import annotations
import argparse,csv,json,sys,zipfile,hashlib
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'reference')]
from checkpoint_io_v18_28 import read as read_parent
from frozen_preflight import observers
from blind_experiment import initial_directions
from blind_math import HISTORY,HORIZONS
from second_variational_math_v18_28 import (directional_second,directional_first,
     second_variational_rhs,paired_even_observation,rms,masked_ratio)
from checkpoint_io_v18_21 import PREFIX as PREFIX20
import q550_state_impulse_response_v13 as original

CASES=(('material3',3.5),('material3',4.5),('raw4',4.),('raw6',4.5))
EPS=(1e-5,1e-6)
STEPS=(0.005,0.02,0.08)
RTOL=3e-13;ATOL=3e-14;MAX_STEP=.0125

def oracle():
    D,provenance,previous_proof,_=read_parent();d=D['tight']
    base=original.DEFAULT_BASE
    for name,(fname,sha) in original.DEPENDENCIES.items():
        p=base/fname
        if not p.exists() or original.hash_file(p)!=sha:
            raise RuntimeError('Pinned Q550 original source SHA failure '+str(p))
    with zipfile.ZipFile(ROOT/'.runtime/v18_20_original_server_sendback.zip') as z:
        seeds=json.loads(z.read(PREFIX20+'results/matched_hidden_seeds.json'))
    directions=initial_directions({r['name']:r['seed'] for r in seeds})
    Hs=observers(d)
    audit=original.load_module(base/original.DEPENDENCIES['audit'][0],'v1828_audit')
    support=original.load_module(base/original.DEPENDENCIES['support'][0],'v1828_support')
    params=audit.ModelParameters(gamma_rho=.3)
    kappa0=(55./6.)*(10./550.);slope=float(d['slope'])
    def f(xi,y):
        return slope*np.asarray(support.support_rhs_float(np.asarray(y,float),
                  kappa0=kappa0,params=params,audit=audit),float)
    return d,provenance,Hs,directions,f

def integrate(fun,a,b,x,times):
    times=np.asarray(times,float)
    if times[0]!=float(a):raise AssertionError('Sample first position not integration start')
    if (b-a>0 and np.any(np.diff(times)<0)) or (b-a<0 and np.any(np.diff(times)>0)):
        raise AssertionError('Nonmonotonic output samples')
    sol=solve_ivp(fun,(float(a),float(b)),np.asarray(x,float),t_eval=times,
       method='DOP853',rtol=RTOL,atol=ATOL,max_step=MAX_STEP)
    if not sol.success:raise RuntimeError('DOP853 integration failed: '+sol.message)
    if not np.all(np.isfinite(sol.y)):raise RuntimeError('Nonfinite trajectory')
    return sol.y.T

def two_sided(fun,a,z):
    back=np.asarray(a+HISTORY[::-1],float)
    forward=np.asarray(a+np.r_[0.,HORIZONS],float)
    left=integrate(fun,a,back[-1],z,back)
    right=integrate(fun,a,forward[-1],z,forward)
    times=np.r_[back[::-1],forward[1:]]
    values=np.vstack((left[::-1],right[1:]))
    if len(times)!=len(HISTORY)+len(HORIZONS):raise AssertionError('Grid count')
    return times,values

def material_seed(d,initial,a):
    if a==4.0:return np.array(initial,float,copy=True)
    P=CubicSpline(np.asarray(d['xi_nodes']),np.asarray(d['P_full11D'][:,:10,:10]),axis=0)
    v=P(a)@np.linalg.solve(P(4.),np.asarray(initial,float))
    return v/np.linalg.norm(v)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--base-root',type=Path,default=original.DEFAULT_BASE)
    args=ap.parse_args();original.DEFAULT_BASE=args.base_root
    d,provenance,Hs,directions,f=oracle()
    results=ROOT/'results';results.mkdir(exist_ok=True)
    centers={};solve_count=0
    for a in sorted({x[1] for x in CASES}):
        center=integrate(f,0.,a,np.asarray(d['source_state_11D'],float),[0.,a])[-1]
        centers[a]=center;solve_count+=1
    rows=[];profile=[];source_rows=[];cases=[]
    for obs,a in CASES:
        center=centers[a]
        H=Hs[obs]
        u0=material_seed(d,directions[obs]['matched_seed'],a)
        if abs(np.linalg.norm(u0)-1)>1e-11:raise RuntimeError('Seed not unit')
        z0=np.r_[center,u0,np.zeros(10)]
        var={}
        for step in STEPS:
            vfun=second_variational_rhs(f,step,first_step=.004,hessian_order=4)
            times,z=two_sided(vfun,a,z0);solve_count+=2
            if np.max(np.abs(z[:,10]-center[10]- (times-a)*f(a,center)[10]))>1e-6:
                # This is diagnostic only, phase derivative could depend on phase
                raise RuntimeError('Inconsistent common phase propagation')
            var[step]=z
            for t,part in zip(times,z):
                y=part[:11];u=np.r_[part[11:21],0.]
                source=directional_second(f,y,u,step)
                source_rows.append({'observer':obs,'anchor':a,'xi':float(t),'step':step,
                   'source_hessian_observed_rms':rms(source[:10]@H.T),
                   'source_hessian_full10_norm':float(np.linalg.norm(source[:10])),
                   'u_full10_norm':float(np.linalg.norm(u[:10])),
                   'w_full10_norm':float(np.linalg.norm(part[21:31]))})
        center_times,baseline=two_sided(f,a,center);solve_count+=2
        if np.max(abs(times-center_times))>1e-12:raise RuntimeError('Variational grid mismatch')
        # Variational center and independently integrated baseline must agree.
        y_consistency=rms(var[STEPS[0]][:,:10]-baseline[:,:10])
        if y_consistency>3e-5:raise RuntimeError('Augmented RHS changes center dynamics')
        var_diff=var[STEPS[0]][:,21:31]-var[STEPS[-1]][:,21:31]
        hessian_w_disagreement=rms(var_diff@H.T)
        source_a=directional_second(f,center,np.r_[u0,0.],STEPS[0])
        source_b=directional_second(f,center,np.r_[u0,0.],STEPS[-1])
        source_gap=rms((source_a-source_b)[:10]@H.T)
        print('SOURCE',obs,'xi',a,'Hessian-gap',f'{source_gap:.3e}',
             'w-gap',f'{hessian_w_disagreement:.3e}',flush=True)
        for eps in EPS:
            initp=center.copy();initm=center.copy()
            initp[:10]+=eps*u0;initm[:10]-=eps*u0
            pt,plus=two_sided(f,a,initp)
            mt,minus=two_sided(f,a,initm)
            solve_count+=4
            if not (np.array_equal(pt,times) and np.array_equal(mt,times)):
                raise RuntimeError('Mismatched physical sampling')
            even,odd=paired_even_observation(baseline,plus,minus,H,eps)
            pred_odd=var[STEPS[1]][:,11:21]@H.T
            pred_even=.5*eps*(var[STEPS[1]][:,21:31]@H.T)
            pred_even_wide=.5*eps*(var[STEPS[-1]][:,21:31]@H.T)
            floor_proxy=.5*eps*hessian_w_disagreement
            # Full-grid and future-only diagnostics: include the actual source
            # numerical discrepancy and the old nonlinear cancellation floor.
            old_even_rms=rms(even);pred_rms=rms(pred_even)
            record={'observer':obs,'anchor':a,'epsilon':eps,
              'paired_even_normalized_rms':old_even_rms,
              'second_variational_even_predicted_rms':pred_rms,
              'paired_odd_normalized_rms':rms(odd),
              'variational_odd_predicted_rms':rms(pred_odd),
              'paired_even_minus_second_variational_rms':rms(even-pred_even),
              'paired_odd_minus_first_variation_rms':rms(odd-pred_odd),
              'second_variation_stencil_difference_rms':floor_proxy,
              'second_variation_wide_step_predicted_rms':rms(pred_even_wide),
              'second_variation_small_step_predicted_rms':rms(.5*eps*(var[STEPS[0]][:,21:31]@H.T)),
              'source_hessian_stencil_difference_at_anchor_rms':source_gap,
              'source_hessian_middle_step_at_anchor_rms':rms(directional_second(
                    f,center,np.r_[u0,0.],STEPS[1])[:10]@H.T),
              'signal_over_hessian_stencil_difference':masked_ratio(pred_rms,floor_proxy),
              'measured_over_predicted_even_rms':masked_ratio(old_even_rms,pred_rms),
              'baseline_variational_state_rms_gap':y_consistency,
              'paired_phase_max_deviation':float(np.max(abs(plus[:,10]-minus[:,10]))),
              'not_a_certified_error_bound':True}
            rows.append(record)
            for i,t in enumerate(times):
                profile.append({'observer':obs,'anchor':a,'epsilon':eps,'xi':float(t),
                    'delta_xi':float(t-a),'paired_even_normalized_norm':float(np.linalg.norm(even[i])),
                    'predicted_even_normalized_norm':float(np.linalg.norm(pred_even[i])),
                    'predicted_even_wide_normalized_norm':float(np.linalg.norm(pred_even_wide[i])),
                    'paired_odd_normalized_norm':float(np.linalg.norm(odd[i])),
                    'predicted_odd_normalized_norm':float(np.linalg.norm(pred_odd[i])),
                    'even_residual_norm':float(np.linalg.norm(even[i]-pred_even[i]))})
            print('PAIR',obs,'xi',a,'eps',eps,'measured-even',f'{old_even_rms:.3e}',
                  'predicted-even',f'{pred_rms:.3e}','stencil-floor',f'{floor_proxy:.3e}',flush=True)
        cases.append({'observer':obs,'anchor':a,'u_norm':float(np.linalg.norm(u0)),
                 'baseline_consistency_rms':y_consistency,'source_gap_rms':source_gap,
                 'max_relative_state_displacement_smallest_step':float(np.max(
                     abs(STEPS[0]*u0)/(abs(center[:10])+1e-12))),
                 'max_relative_state_displacement_largest_step':float(np.max(
                     abs(STEPS[-1]*u0)/(abs(center[:10])+1e-12)))})
    def write_csv(name,data):
        with (results/name).open('w',newline='') as f:
            wr=csv.DictWriter(f,fieldnames=list(data[0]));wr.writeheader();wr.writerows(data)
    write_csv('original_second_variational_pairs_v18_28.csv',rows)
    write_csv('original_second_variational_profiles_v18_28.csv',profile)
    write_csv('original_second_variational_sources_v18_28.csv',source_rows)
    proof={'version':'v18.28','status':'ORIGINAL_Q550_SECOND_VARIATIONAL_SOURCE_AND_NONLINEAR_PAIR_AUDIT_COMPLETED',
       'source_verified_against_pin':True,'parent_verified_sha256':provenance['v18_27_sendback_sha256'],
       'new_original_ode_integrations':solve_count,'number_of_cases':len(cases),
       'epsilon_pairs':len(rows),'hessian_steps':list(STEPS),'variational_prediction_uses_middle_step':True,
       'first_directional_step':.004,'hessian_stencil_order':4,
       'rtol':RTOL,'atol':ATOL,'max_step':MAX_STEP,
       'cases':cases,'sample_count_per_case':len(times),
       'phase_common_within_pair':all(r['paired_phase_max_deviation']<1e-9 for r in rows),
       'signed_quadratic_predicted_by_independently_integrated_variational_equation':True,
       'finite_difference_hessian_is_numeric_not_analytic':True,
       'cross_stencil_differences_are_not_rigorous_error_bounds':True,
       'all_diagnostics_off_constraint':True,
       'no_independent_material_metric_claim':True}
    (results/'original_second_variational_proof_v18_28.json').write_text(json.dumps(proof,indent=2)+'\n')
    print('ORIGINAL_SECOND_VARIATIONAL_V18_28_COMPLETED',solve_count,'ODE integrations,',len(rows),'pair comparisons')

if __name__=='__main__':main()
