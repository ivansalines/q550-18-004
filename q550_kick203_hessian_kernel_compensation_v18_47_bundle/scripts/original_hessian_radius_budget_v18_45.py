#!/usr/bin/env python3
"""Original Q550 v18.45: 36 matched states, 11 RHS radii, 6 J radii.
All 25 integrations use the original SHA-pinned ODE. No static Q550 claim
is made by frozen/preflight data. The 'best radius' is reference-dependent.
"""
import sys, csv, json, argparse
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
R=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(R/'scripts'),str(R/'reference')]
from checkpoint_io_v18_45 import read_parent_v45
from checkpoint_io_v18_44 import read_parent_v44
from checkpoint_io_v18_43 import read_parent_v43
from checkpoint_io_v18_42 import read_parent_v42
from checkpoint_io_v18_41 import read_parent_v41
from checkpoint_io_v18_40 import read_parent_v40
from checkpoint_io_v18_39 import read_parent_v39
from checkpoint_io_v18_38 import read_parent_v38
from checkpoint_io_v18_37 import read_parent_v37
from checkpoint_io_v18_36 import read_parent_v36
from checkpoint_io_v18_35 import read_parent_v35
from checkpoint_io_v18_34 import read_parent
from original_material_invariants_v18_31 import select_original_checkpoint,model,ANCHORS
from original_invariant_manifold_v18_32 import anchored_flow
from original_cofactor_surface_return_v18_33 import evaluate_samples
from original_flow_normal_v18_36 import NAMES,ANCHOR_INDICES,projected_independent_seed
from seed_genealogy_v18_38 import index_archived_attempts,verify_regenerated_attempts
from near_surface_math_v18_35 import calibration_radius
from residual_gradient_math_v18_37 import source_geometry
from dual_hessian_jet_math_v18_44 import dot_B_transpose_g_from_jacobian,mixed_rhs_hessian_contraction
from hessian_budget_math_v18_45 import RHS_H,JAC_H,relative,jac_reference,neighbor_instability,select_radius,component_budget,vec_cols
import q550_state_impulse_response_v13 as pinned
FRACS=(.125,.5,1.)

def output(name,records):
    if not records:raise RuntimeError('Missing rows '+name)
    dst=R/'results'/name
    with dst.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)

def key(row):return (row['id'],round(float(row['anchor_xi']),8),int(row['branch']))
def sol_record(case_id,kind,sol):
    dt=np.diff(sol.t)
    return dict(case_id=case_id,kind=kind,accepted_steps=len(dt),nfev=sol.nfev,
                min_dt=float(dt.min()),max_dt=float(dt.max()))

def run(args):
    pp,old_h,old_jets,old_family,auth=read_parent_v45()
    p43,*_=read_parent_v44()
    p42,old42,*_=read_parent_v43()
    read_parent_v42();read_parent_v41();read_parent_v40();read_parent_v39();read_parent_v38()
    _,cases36,_,attempts36,_=read_parent_v37()
    read_parent_v36();read_parent_v35()
    checkpoints,_,m33,*_=read_parent()
    original=select_original_checkpoint(checkpoints)
    f,jac=model(args.base_root,original)
    anchors=anchored_flow(f,np.asarray(original['source_state_11D'],float),ANCHORS)
    center=np.asarray(m33['center'],float);scale=np.asarray(m33['scale'],float);theta0=float(m33['theta0'])
    train,*_=evaluate_samples(f,anchors,scale)
    candidates={v['id']:v for v in m33['candidates']}
    case36map={key(r):r for r in cases36};attemptmap=index_archived_attempts(attempts36)
    old42map={key(r):r for r in old42}
    oldmap={ (r['case_id'],round(float(r['fraction_of_common_time']),8),float(r['alpha']),float(r['rhs_normalized_h'])):r for r in old_h}
    hrows=[];jrows=[];comps=[];summ=[];solver=[]
    rng=np.random.default_rng(1836);max_regression=0.;max_support=0.;integrations=1
    for name in NAMES:
        obj=candidates[name];q=np.asarray(obj['q'],float);opts=obj['opts'];mu=float(opts['lam'])
        if mu!=-1 or float(opts['mod'])!=0:raise RuntimeError('Unexpected original cofactor')
        for i in ANCHOR_INDICES:
            anchor=anchors[i];xi=float(ANCHORS[i])
            radius=float(calibration_radius(train[i*56:(i+1)*56],anchor,scale)['radius_max'])
            for branch in (0,1):
                ck=(name,round(xi,8),branch)
                label=f'{name}|xi={xi:g}|branch={branch}'
                old=old42map.get(ck)
                if old is None:raise RuntimeError('Original v18.42 case missing')
                T=float(old['common_within_support_time'])
                y0,attempts=projected_independent_seed(anchor,q,center,scale,theta0,opts,radius,rng,branch)
                verify_regenerated_attempts(ck,attempts,case36map,attemptmap)
                sol=solve_ivp(f,(0,T),y0,method='DOP853',rtol=args.rtol,atol=args.atol,
                              max_step=args.max_step,dense_output=True)
                if not sol.success:raise RuntimeError('Original Q550 center trajectory failure: '+sol.message)
                integrations+=1;solver.append(sol_record(label,'center_seed',sol))
                yref=anchor.copy();yref[10]=y0[10]
                ref=solve_ivp(f,(0,T),yref,method='DOP853',rtol=args.rtol,atol=args.atol,
                              max_step=args.max_step,dense_output=True)
                if not ref.success:raise RuntimeError('Original Q550 phase-matched reference failure')
                integrations+=1;solver.append(sol_record(label,'phase_matched_reference',ref))
                for frac in FRACS:
                    t=float(frac*T);y=np.asarray(sol.sol(t),float);fy=np.asarray(f(0,y),float);J=np.asarray(jac(y),float)
                    if np.linalg.norm(J[10,:10]*scale)>1e-7:raise RuntimeError('Original clock depends materially')
                    support_d=float(np.linalg.norm((y[:10]-ref.sol(t)[:10])/scale))/radius
                    max_support=max(max_support,support_d)
                    if support_d>1+1e-6:raise RuntimeError('Original snapshot exited calibrated support')
                    geometry=source_geometry(y,fy,J,q,center,scale,theta0,opts['degree'],opts['phase'],mu)
                    jvals={a:dot_B_transpose_g_from_jacobian(y,fy,jac,scale,geometry['g'],a) for a in JAC_H}
                    jref=jac_reference(jvals)
                    jref_spread=max(relative(jvals[a],jref) for a in (.0005,.001,.002))
                    rvals={h:mixed_rhs_hessian_contraction(y,fy,f,scale,geometry['g'],h) for h in RHS_H}
                    oldrec=oldmap.get((label,round(float(frac),8),0.0,.006))
                    if oldrec is None:raise RuntimeError('No original v18.44 center regression row')
                    oj=np.array([float(oldrec[f'jac_hessian_{c}']) for c in range(10)])
                    orhs=np.array([float(oldrec[f'rhs_hessian_{c}']) for c in range(10)])
                    errj=relative(jvals[.002],oj);errrhs=relative(rvals[.006],orhs)
                    max_regression=max(max_regression,errj,errrhs)
                    if errj>args.max_parent_regression or errrhs>args.max_parent_regression:
                        raise RuntimeError(f'Original v18.44 Hessian regression failed {label} {frac}: {errj:.2e} / {errrhs:.2e}')
                    rvec=[rvals[h] for h in RHS_H];ibest,errors=select_radius(RHS_H,rvec,jref)
                    best_h=RHS_H[ibest];best=rvec[ibest]
                    for ih,h in enumerate(RHS_H):
                        rvec_h=rvals[h]
                        hrows.append(dict(case_id=label,id=name,anchor_xi=xi,branch=branch,alpha=0,
                             fraction_of_common_time=frac,delta_xi=t,phase_lifted=float(y[10]),
                             phase_advance_lifted=float(y[10]-y0[10]),support_radius_fraction=support_d,
                             rhs_normalized_h=h,relative_to_jac_median=errors[ih],
                             neighbor_relative_change=neighbor_instability(rvec,ih),
                             jac_consensus_spread=jref_spread,original_h_is_parent=(h in (.006,.018,.036)),
                             **vec_cols(rvec_h,'rhs_'),**vec_cols(jref,'jac_reference_')))
                    for h in JAC_H:
                        jrows.append(dict(case_id=label,id=name,anchor_xi=xi,branch=branch,
                            fraction_of_common_time=frac,delta_xi=t,jac_flow_scaled_target=h,
                            relative_to_jac_median=relative(jvals[h],jref),
                            relative_to_parent_jac_0002=relative(jvals[h],oj),
                            **vec_cols(jvals[h],'jac_'),**vec_cols(jref,'jac_reference_')))
                    discrepancy,shares=component_budget(jref,best)
                    for k in range(10):
                        comps.append(dict(case_id=label,id=name,anchor_xi=xi,branch=branch,
                            fraction_of_common_time=frac,component=k,selected_rhs_h=best_h,
                            jac_reference=float(jref[k]),best_rhs=float(best[k]),
                            signed_discrepancy=float(discrepancy[k]),absolute_discrepancy=float(abs(discrepancy[k])),
                            relative_to_reference_component=float(abs(discrepancy[k])/max(abs(jref[k]),1e-12)),
                            squared_norm_error_share=float(shares[k]),
                            parent_rhs_0006=float(orhs[k]),parent_jac_0002=float(oj[k])))
                    summ.append(dict(case_id=label,id=name,anchor_xi=xi,branch=branch,
                        fraction_of_common_time=frac,common_support_time=T,delta_xi=t,
                        phase_lifted=float(y[10]),support_radius_fraction=support_d,
                        selected_rhs_h=best_h,selected_cross_path_error=float(errors[ibest]),
                        baseline_0006_error=float(errors[RHS_H.index(.006)]),
                        selected_neighbor_instability=neighbor_instability(rvec,ibest),
                        jac_internal_reference_spread=jref_spread,
                        regression_relative_jac=errj,regression_relative_rhs=errrhs,
                        dominant_component=int(np.argmax(shares)),dominant_component_error_share=float(max(shares)),
                        all_component_abs_error=float(np.linalg.norm(discrepancy))))
                print('V18_45',label,'grid',len(RHS_H),'x',len(JAC_H),'parent_reg',f'{max_regression:.3e}',flush=True)
    if tuple(map(len,(summ,hrows,jrows,comps,solver)))!=(36,396,216,360,24) or integrations!=25:
        raise RuntimeError('Incomplete original Q550 v18.45; do not package')
    for name,rows in [('original_v18_45_rhs_radius_scan.csv',hrows),
                      ('original_v18_45_jac_target_scan.csv',jrows),
                      ('original_v18_45_component_error_budget.csv',comps),
                      ('original_v18_45_snapshot_summary.csv',summ),
                      ('original_v18_45_solver_steps.csv',solver)]:output(name,rows)
    proof=dict(version='v18.45',status='ORIGINAL_Q550_HESSIAN_RADIUS_ERROR_BUDGET_COMPLETED',
       original_integrations=integrations,original_cases=12,original_snapshots=len(summ),
       rhs_radius_rows=len(hrows),jac_target_rows=len(jrows),component_budget_rows=len(comps),
       solver_rows=len(solver),parent_v18_44_sha256=auth['parent_sha256'],
       parent_v18_44_verified_members=auth['verified_members'],max_v18_44_vector_regression=max_regression,
       max_within_parent_support_radius_fraction=max_support,
       original_ode_sha256_verified=True,source_kappa0_unchanged=True,no_symbolic_hessian_claimed=True,
       no_certified_second_jet=True,no_certified_curvature=True,no_material_metric_certified=True,
       no_physical_constraint_certified=True,
       method=dict(rhs_radii=list(RHS_H),jac_targets=list(JAC_H),
          hessian='original RHS mixed fourth-order derivatives versus numerical directional pinned Jacobian',
          jac_reference='componentwise median of targets 0.0005,0.001,0.002; NOT true Hessian',
          min_discrepancy='descriptive minimum across tested RHS radii relative to the selected Jacobian consensus',
          error_budget='10-component squared Euclidean norm error share and neighbor radius instability',
          ancestor='Original center seed reproduced from pinned v18.36 acceptance ledger; original v18.44 exact-radius regression',
          kappa0='Frozen inherited old source kappa0; geometric proposal NOT introduced',
          caution='No certified precision floor, analytic Hessian, intrinsic torsion or stiffness inferred'),preflight_only=False)
    (R/'results/original_v18_45_proof.json').write_text(json.dumps(proof,indent=2,allow_nan=False)+'\n')
    print('ORIGINAL_V18_45_COMPLETED',integrations,len(hrows),len(jrows),max_regression,flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--base-root',type=Path,default=pinned.DEFAULT_BASE)
    ap.add_argument('--rtol',type=float,default=3e-12)
    ap.add_argument('--atol',type=float,default=3e-13)
    ap.add_argument('--max-step',type=float,default=2.5e-7)
    ap.add_argument('--max-parent-regression',type=float,default=2e-6)
    run(ap.parse_args())
