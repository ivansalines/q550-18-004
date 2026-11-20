#!/usr/bin/env python3
"""Original pinned Q550 independent RHS-Hessian / constraint-tangent audit.

No material constraints are invented. With --constraints-module, the external,
model-specific invariants are checked for conservation before interpreting
constrained initial second variations. Otherwise only the exact common-phase
fiber is confirmed; all material perturbations remain OFF_CONSTRAINT_DIAGNOSTIC.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,sys
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'reference')]
from checkpoint_io_v18_30 import read
from frozen_preflight import observers
from blind_experiment import initial_directions
from blind_math import HISTORY,HORIZONS
from checkpoint_io_v18_21 import PREFIX as PREFIX20
from hessian_constraint_math_v18_30 import rhs_hessian,jac_hessian,second_variational_rhs,finite_jac_c,finite_c_hessian,constraint_initial_jet,rms
import q550_state_impulse_response_v13 as original
import zipfile

CASES=(('raw6',4.5),('material3',3.5))
STEPS=(.02,.08)
EPS=(1.e-5,1.e-6)
RTOL=3.e-13; ATOL=3.e-14; MAX_STEP=.015

def sample(fun,a,y,back=True):
    grid=np.asarray(a+HISTORY[::-1],float) if back else np.asarray(a+np.r_[0.,HORIZONS],float)
    sol=solve_ivp(fun,(a,float(grid[-1])),np.asarray(y,float),t_eval=grid,method='DOP853',rtol=RTOL,atol=ATOL,max_step=MAX_STEP)
    if not sol.success or not np.all(np.isfinite(sol.y)):raise RuntimeError('Original ODE failed: '+sol.message)
    return grid,sol.y.T

def sided(fun,a,y):
    l,lv=sample(fun,a,y,True);r,rv=sample(fun,a,y,False)
    return np.r_[l[::-1],r[1:]],np.vstack((lv[::-1],rv[1:]))

def original_model(base):
    data,prov,prior,prior_profiles,prior_cases=read()
    d=data['tight']
    for role,(fname,expected) in original.DEPENDENCIES.items():
        p=base/fname
        if not p.is_file() or original.hash_file(p)!=expected:raise RuntimeError('Pinned original Q550 missing or source hash mismatch: '+str(p))
    audit=original.load_module(base/original.DEPENDENCIES['audit'][0],'v1830_audit')
    support=original.load_module(base/original.DEPENDENCIES['support'][0],'v1830_support')
    tube=original.load_module(base/original.DEPENDENCIES['tube'][0],'v1830_tube')
    params=audit.ModelParameters(gamma_rho=.3)
    k0=(55./6.)*(10./550.);slope=float(d['slope'])
    def f(t,y):return slope*np.asarray(support.support_rhs_float(np.asarray(y,float),kappa0=k0,params=params,audit=audit),float)
    def jac(y):return slope*np.asarray(tube.float_support_jacobian(np.asarray(y,float),kappa0=k0,params=params,audit=audit,base=support),float)
    with zipfile.ZipFile(ROOT/'.runtime/v18_20_original_server_sendback.zip') as z:
        seeds=json.loads(z.read(PREFIX20+'results/matched_hidden_seeds.json'))
    direction=initial_directions({r['name']:r['seed'] for r in seeds})
    return d,prov,prior,prior_profiles,observers(d),direction,f,jac

def seed_at(d,v,a):
    v=np.asarray(v,float)
    if abs(a-4)<1e-12:return v/np.linalg.norm(v)
    P=CubicSpline(np.asarray(d['xi_nodes']),np.asarray(d['P_full11D'][:,:10,:10]),axis=0)
    w=P(a)@np.linalg.solve(P(4.),v)
    return w/np.linalg.norm(w)

def write_csv(path,rows):
    if not rows:return
    with path.open('w',newline='') as fp:
        wr=csv.DictWriter(fp,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)

def load_invariants(module_path):
    if module_path is None:return None,None
    if not module_path.is_file():raise FileNotFoundError(module_path)
    mod=original.load_module(module_path,'v1830_user_supplied_invariants')
    callback=getattr(mod,'invariants',None)
    if not callable(callback):raise RuntimeError('Constraint module must supply invariants(y)->1D array')
    return callback,dict(path=str(module_path.resolve()),sha256=original.hash_file(module_path))

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--base-root',type=Path,default=original.DEFAULT_BASE)
    ap.add_argument('--constraints-module',type=Path,default=None,help='Explicit, trusted model-specific Python invariants(y); optional')
    args=ap.parse_args()
    d,prov,parent,prior_profiles,Hs,directions,f,jac=original_model(args.base_root)
    invariants,plugin=load_invariants(args.constraints_module)
    previous=list(csv.DictReader(prior_profiles.splitlines()))
    historical={(r['observer'],round(float(r['anchor']),6),round(float(r['epsilon']),12),r['tier'],round(float(r['xi']),10)):r for r in previous}
    results=ROOT/'results';results.mkdir(exist_ok=True)
    rows=[];src_rows=[];profiles=[];constraints=[]; integrations=0;phase_defect=0.;all_conservation=[]
    baseline_states={}
    for a in sorted({x[1] for x in CASES}):
        sol=solve_ivp(f,(0.,a),np.asarray(d['source_state_11D'],float),t_eval=[a],method='DOP853',rtol=RTOL,atol=ATOL,max_step=MAX_STEP)
        if not sol.success:raise RuntimeError('Original-Q550 anchor baseline failed: '+sol.message)
        baseline_states[a]=sol.y[:,-1];integrations+=1
    for name,a in CASES:
        H=np.asarray(Hs[name],float)
        y0=baseline_states[a]
        v=seed_at(d,directions[name]['matched_seed'],a)
        init_u=np.r_[v,0.];initial_w=np.zeros(11)
        jet_info={'status':'OFF_CONSTRAINT_DIAGNOSTIC_NO_MATERIAL_INVARIANTS'}
        if invariants is not None:
            init_u,initial_w,jet_info=constraint_initial_jet(invariants,y0,init_u)
            jet_info['status']='USER_DECLARED_CONSTRAINT_CANDIDATE'
        z0=np.r_[y0,init_u[:10],initial_w[:10]]
        solutions={}
        for h in STEPS:
            grid,states=sided(second_variational_rhs(f,jac,h),a,z0)
            integrations+=2
            solutions[h]=states
        state=solutions[STEPS[0]]
        mask=(name,round(a,6),round(EPS[0],12),'tight')
        h_by_source=[];cross_source=[];cdef=[]
        for i,(t,z) in enumerate(zip(grid,state)):
            yy=z[:11];uu=np.r_[z[11:21],0.];ww=np.r_[z[21:31],0.]
            phase_defect=max(phase_defect,abs(uu[-1]),abs(ww[-1]))
            fj=f(t,yy)
            jc=jac(yy)
            tangent=(jc@uu)[:10]
            fp=(f(t,yy+3e-4*uu)-f(t,yy-3e-4*uu))/(6e-4)
            if not (np.all(np.isfinite(fj)) and np.all(np.isfinite(jc))):raise RuntimeError('Nonfinite original field/Jacobian')
            # At each time sample, use the same y,u to compare source constructions.
            for h in STEPS:
                ds=rhs_hessian(f,t,yy,uu,h)
                js=jac_hessian(jac,yy,uu,h)
                rel_norm=max(np.linalg.norm(js[:10]),np.linalg.norm(ds[:10]),1e-15)
                src_rows.append(dict(observer=name,anchor=a,xi=float(t),h=h,
                  rhs_source_norm=float(np.linalg.norm(ds[:10])),
                  jacobian_source_norm=float(np.linalg.norm(js[:10])),
                  source_disagreement_norm=float(np.linalg.norm((ds-js)[:10])),
                  source_disagreement_rel=float(np.linalg.norm((ds-js)[:10])/rel_norm),
                  source_observed_gap=float(np.linalg.norm(H@(ds-js)[:10])),
                  max_relative_coordinate_step=float(np.max(np.abs(h*uu[:10]/max(np.linalg.norm(uu[:10]),1e-15))/(np.abs(yy[:10])+1e-12))),
                  phase_source_abs=float(abs(ds[10])),
                  tangent_directional_jac_relative=float(np.linalg.norm(fp[:10]-tangent)/max(np.linalg.norm(tangent),1e-13))))
                h_by_source.append(rms((ds-js)[:10]))
                cross_source.append(float(np.linalg.norm(H@(ds-js)[:10])))
            if invariants is not None:
                G=finite_jac_c(invariants,yy)
                C=np.atleast_1d(invariants(yy)).astype(float)
                C0=np.atleast_1d(invariants(y0)).astype(float)
                C2=finite_c_hessian(invariants,yy,uu)
                drift=float(np.linalg.norm(C-C0))
                f_def=float(np.linalg.norm(G@fj))
                c1=float(np.linalg.norm(G@uu))
                c2=float(np.linalg.norm(G@ww+C2))
                constraints.append(dict(observer=name,anchor=a,xi=float(t),
                  invariant_drift=drift,flow_tangency_defect=f_def,
                  first_constraint_defect=c1,second_constraint_defect=c2))
                cdef.append((drift,f_def,c1,c2))
                all_conservation.append((drift,f_def,c1,c2))
            for h in STEPS:
                z2=solutions[h][i]
                # Second variation may have nonzero curvature correction at anchor.
                even=.5*EPS[0]*(z2[21:31]@H.T)
                old=historical.get((name,round(a,6),round(EPS[0],12),'tight',round(float(t),10)))
                profiles.append(dict(observer=name,anchor=a,xi=float(t),h=h,
                  new_rhs_even_norm=float(np.linalg.norm(even)),
                  archived_jac_second_even_norm=float(old['second_variational_even_norm']) if old and invariants is None else float('nan'),
                  archived_compensated_even_norm=float(old['compensated_even_norm']) if old and invariants is None else float('nan'),
                  new_rhs_observer_norm_vs_archived_jac_abs=float(abs(np.linalg.norm(even)-float(old['second_variational_even_norm']))) if old and invariants is None else float('nan')))
        for eps in EPS:
            wa=solutions[STEPS[0]][:,21:31]@H.T
            wb=solutions[STEPS[1]][:,21:31]@H.T
            prediction=.5*eps*wa
            archive=[p for p in profiles if p['observer']==name and p['h']==STEPS[0]]
            if invariants is None:
                prior_norm=np.array([p['archived_jac_second_even_norm'] for p in archive])
                prev_comp=np.array([p['archived_compensated_even_norm'] for p in archive])
                # For EPS 1e-6, archived prior_norm must be rescaled from eps=1e-5.
                s=eps/EPS[0]
                archived_norm_disagreement=rms(np.linalg.norm(prediction,axis=1)-s*prior_norm)
                archived_comp_disagreement=rms(np.linalg.norm(prediction,axis=1)-s*prev_comp)
            else:archived_norm_disagreement=archived_comp_disagreement=float('nan')
            rows.append(dict(observer=name,anchor=a,epsilon=eps,
                rhs_hessian_even_rms=rms(prediction),
                rhs_stencil_disagreement_rms=.5*eps*rms(wa-wb),
                rhs_vs_archived_jac_norm_profile_rms=archived_norm_disagreement,
                rhs_vs_archived_compensated_norm_profile_rms=archived_comp_disagreement,
                rhs_vs_jac_source_sample_rms=rms(h_by_source),
                rhs_vs_jac_observed_source_sample_rms=rms(cross_source),
                material_invariants_supplied=invariants is not None,
                constraint_status=jet_info['status']))
            print('INDEPENDENT_HESS',name,'anchor',a,'epsilon',eps,'new RMS',f'{rms(prediction):.9e}',
                  'stencil gap',f'{.5*eps*rms(wa-wb):.4e}',
                  'archived gap',f'{archived_norm_disagreement:.4e}',flush=True)
        if invariants is not None:
            constraint_summary=dict(observer=name,anchor=a,initial_jet=jet_info,
               max_invariant_drift=max(q[0] for q in cdef),
               max_flow_tangent=max(q[1] for q in cdef),
               max_first_tangency=max(q[2] for q in cdef),
               max_second_tangency=max(q[3] for q in cdef))
            (results/f'{name}_constraint_diagnostic_v18_30.json').write_text(json.dumps(constraint_summary,indent=2)+'\n')
    write_csv(results/'original_independent_hessian_summary_v18_30.csv',rows)
    write_csv(results/'original_independent_hessian_source_v18_30.csv',src_rows)
    write_csv(results/'original_independent_hessian_profiles_v18_30.csv',profiles)
    if constraints:write_csv(results/'original_declared_constraint_residuals_v18_30.csv',constraints)
    verified=bool(invariants is not None and all(q[0]<1e-6 and q[1]<1e-6 and q[2]<1e-5 and q[3]<1e-4 for q in all_conservation))
    proof=dict(version='v18.30',status='ORIGINAL_Q550_INDEPENDENT_RHS_HESSIAN_DIAGNOSTIC_COMPLETED',
      parent_sha256=prov['v18_29_original_SHA256'],parent_manifest_verified=prov['v18_29_verified_members'],
      original_ODE_input_SHA256_verified=True, original_ode_integrations=integrations,
      case_count=len(CASES),case_summary_rows=len(rows),source_samples=len(src_rows),
      rhs_hessian_stencils=list(STEPS),epsilon_values=list(EPS),
      independent_source='FIVE_POINT_FOURTH_ORDER_DIRECTIONAL_RHS_DIFFERENCE',
      comparison_source='FOUR_POINT_FOURTH_ORDER_DIRECTIONAL_ANALYTIC_JACOBIAN_DIFFERENCE',
      original_parent_second_variation_and_compensated_profiles_reused=True,
      phase_fiber_defect=phase_defect,
      declared_material_invariants=plugin,
      material_constraint_status=('DECLARED_INVARIANTS_NUMERICALLY_COMPATIBLE_LOCAL' if verified else
         'DECLARED_INVARIANTS_NOT_VERIFIED' if plugin else 'UNSPECIFIED_NO_MATERIAL_CURVATURE_CERTIFICATE'),
      constrained_jet_sampled=bool(plugin),
      user_plugin_is_not_pinned_original_model=bool(plugin),
      constraints_not_certified_from_Q550_model=not verified,
      no_new_physical_torsion_or_memory_claim=True,
      comparisons_are_empirical_not_interval_bounds=True)
    (results/'original_independent_hessian_proof_v18_30.json').write_text(json.dumps(proof,indent=2)+'\n')
    print('ORIGINAL_V18_30_COMPLETED',integrations,'original Q550 ODE integrations',len(rows),'summary rows',flush=True)
if __name__=='__main__': main()
