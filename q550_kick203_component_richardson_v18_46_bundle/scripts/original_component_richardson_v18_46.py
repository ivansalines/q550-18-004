#!/usr/bin/env python3
"""ORIGINAL Q550: v18.46 component-wise Richardson consistency and radius error budget.
The source ODE, initial seeds, phases, kappa0 and first 36 snapshot coordinates
are inherited unmodified and independently reproduced on the Q550 server.
The RHS mixed stencil and pinned Jacobian stencil are both finite differences;
no symbolic Hessian, precision certificate, invariant, or torsion is asserted.
"""
import sys,csv,json,argparse
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
R=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(R/'scripts'),str(R/'reference')]
from checkpoint_io_v18_46 import read_parent_v46
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
from richardson_math_v18_46 import RHS_H,JAC_H,TRIPLES_RHS,TRIPLES_JAC,KEY_COMPONENTS,relative,triple_stats,diagnostic_references,error_shares,vec_fields
import q550_state_impulse_response_v13 as pinned
FRACS=(.125,.5,1.)

def key(r):return (r['id'],round(float(r['anchor_xi']),8),int(r['branch']))
def snapshot_key(r):return (r['case_id'],round(float(r['fraction_of_common_time']),8))
def output(name,rows):
    if not rows:raise RuntimeError('Missing result rows '+name)
    with (R/'results'/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def read_vec(r,pre):return np.array([float(r[f'{pre}{i}']) for i in range(10)])
def finite(x):return float(x) if x is not None else ''

def run(args):
    oldproof,oldrhs,oldjac,_,oldsnap,auth=read_parent_v46()
    read_parent_v45();read_parent_v44()
    _,old42,*_=read_parent_v43()
    read_parent_v42();read_parent_v41();read_parent_v40();read_parent_v39();read_parent_v38()
    _,case36,_,attempt36,_=read_parent_v37()
    read_parent_v36();read_parent_v35()
    checkpoints,_,m33,*_=read_parent()
    original=select_original_checkpoint(checkpoints)
    f,jac=model(args.base_root,original)
    anchors=anchored_flow(f,np.asarray(original['source_state_11D'],float),ANCHORS)
    center=np.asarray(m33['center'],float);scale=np.asarray(m33['scale'],float);theta0=float(m33['theta0'])
    train,*_=evaluate_samples(f,anchors,scale)
    candidates={c['id']:c for c in m33['candidates']}
    old42map={key(r):r for r in old42};case36map={key(r):r for r in case36}
    attempts=index_archived_attempts(attempt36)
    oldr={(snapshot_key(r),round(float(r['rhs_normalized_h']),9)):r for r in oldrhs}
    oldj={(snapshot_key(r),round(float(r['jac_flow_scaled_target']),9)):r for r in oldjac}
    oldsum={snapshot_key(r):r for r in oldsnap}
    snaprows=[];rrows=[];jrows=[];ladderr=[];ladderj=[];comp=[];solver=[]
    rng=np.random.default_rng(1836);max_reg=0.;max_support=0.;count=1
    for name in NAMES:
        obj=candidates[name];q=np.asarray(obj['q'],float);opts=obj['opts'];mu=float(opts['lam'])
        if mu!=-1.0 or float(opts['mod'])!=0.0:raise RuntimeError('Original cofactor changed')
        for i in ANCHOR_INDICES:
            anchor=anchors[i];xi=float(ANCHORS[i])
            radius=float(calibration_radius(train[i*56:(i+1)*56],anchor,scale)['radius_max'])
            for branch in (0,1):
                ck=(name,round(xi,8),branch)
                label=f'{name}|xi={xi:g}|branch={branch}'
                old=old42map.get(ck)
                if old is None:raise RuntimeError('Parent common-time missing')
                T=float(old['common_within_support_time'])
                y0,tries=projected_independent_seed(anchor,q,center,scale,theta0,opts,radius,rng,branch)
                verify_regenerated_attempts(ck,tries,case36map,attempts)
                sol=solve_ivp(f,(0,T),y0,method='DOP853',rtol=args.rtol,atol=args.atol,
                    max_step=args.max_step,dense_output=True)
                if not sol.success:raise RuntimeError('Original Q550 trajectory failed '+sol.message)
                count+=1;solver.append(dict(case_id=label,kind='center_seed',nfev=sol.nfev,
                    accepted_steps=len(sol.t)-1,min_step=float(np.diff(sol.t).min()),max_step=float(np.diff(sol.t).max())))
                yref=anchor.copy();yref[10]=y0[10]
                ref=solve_ivp(f,(0,T),yref,method='DOP853',rtol=args.rtol,atol=args.atol,
                    max_step=args.max_step,dense_output=True)
                if not ref.success:raise RuntimeError('Original Q550 phase matched reference failed')
                count+=1;solver.append(dict(case_id=label,kind='phase_reference',nfev=ref.nfev,
                    accepted_steps=len(ref.t)-1,min_step=float(np.diff(ref.t).min()),max_step=float(np.diff(ref.t).max())))
                for frac in FRACS:
                    t=float(frac*T);y=np.asarray(sol.sol(t),float);fy=np.asarray(f(0,y),float)
                    J=np.asarray(jac(y),float)
                    if np.linalg.norm(J[10,:10]*scale)>1e-7:raise RuntimeError('Clock material coupling differs')
                    sdist=float(np.linalg.norm((y[:10]-ref.sol(t)[:10])/scale))/radius
                    max_support=max(max_support,sdist)
                    if sdist>1+1e-6:raise RuntimeError('Snapshot outside original diagnostic support')
                    geom=source_geometry(y,fy,J,q,center,scale,theta0,opts['degree'],opts['phase'],mu)
                    jr={h:dot_B_transpose_g_from_jacobian(y,fy,jac,scale,geom['g'],h) for h in JAC_H}
                    rh={h:mixed_rhs_hessian_contraction(y,fy,f,scale,geom['g'],h) for h in RHS_H}
                    sk=(label,round(float(frac),8))
                    olds=oldsum.get(sk)
                    if olds is None:raise RuntimeError('Parent snapshot missing')
                    if abs(float(olds['delta_xi'])-t)>2e-12:raise RuntimeError('Original snapshot time changed')
                    if abs(float(olds['support_radius_fraction'])-sdist)>2e-6:raise RuntimeError('Original support regression failed')
                    commonr=set(RHS_H).intersection({.0015,.002,.003,.004,.006,.008,.012,.018,.024,.036,.048})
                    commonj=set(JAC_H).intersection({.00025,.0005,.001,.002,.004,.008})
                    for h in commonr:
                        before=oldr.get((sk,round(h,9)))
                        if before is None:raise RuntimeError('v18.45 RHS regression row missing')
                        err=relative(rh[h],read_vec(before,'rhs_'))
                        max_reg=max(max_reg,err)
                        if err>args.max_parent_regression:raise RuntimeError(f'RHS parent regression {label} {frac} h={h} error={err}')
                    for h in commonj:
                        before=oldj.get((sk,round(h,9)))
                        if before is None:raise RuntimeError('v18.45 Jacobian regression row missing')
                        err=relative(jr[h],read_vec(before,'jac_'))
                        max_reg=max(max_reg,err)
                        if err>args.max_parent_regression:raise RuntimeError(f'Jac parent regression {label} {frac} h={h} error={err}')
                    refs=diagnostic_references(jr);base=refs['parent']
                    base_cols=dict(case_id=label,id=name,anchor_xi=xi,branch=branch,fraction_of_common_time=frac,
                        delta_xi=t,phase_lifted=float(y[10]),phase_advance_lifted=float(y[10]-y0[10]),
                        support_radius_fraction=sdist)
                    for h in RHS_H:
                        rrows.append(dict(**base_cols,rhs_h=h,relative_to_jac_parent=relative(rh[h],base),
                            relative_to_jac_small=relative(rh[h],refs['small']),relative_to_jac_large=relative(rh[h],refs['large']),
                            **vec_fields(rh[h],'rhs_')))
                    for h in JAC_H:
                        jrows.append(dict(**base_cols,jac_target=h,relative_to_jac_parent=relative(jr[h],base),
                            **vec_fields(jr[h],'jac_')))
                    for coarse,med,fine in TRIPLES_RHS:
                        st=triple_stats(rh[coarse],rh[med],rh[fine],floor=args.relative_resolution_floor)
                        common=dict(**base_cols,coarse_h=coarse,medium_h=med,fine_h=fine,
                            observed_order=finite(st['order']),order_resolved=int(st['resolved']),
                            order4_consistent=int(st['order4_consistent']),cos_successive_differences=st['cos_differences'],
                            rich_pair_disagreement=st['richardson_between_pairs'],raw_neighbor_disagreement=st['raw_neighbor_disagreement'],
                            relative_rich_fine_to_jac_parent=relative(st['low'],base),
                            relative_rich_fine_to_jac_small=relative(st['low'],refs['small']),
                            relative_rich_fine_to_jac_large=relative(st['low'],refs['large']),
                            delta_coarse_norm=st['norm_delta_coarse'],delta_fine_norm=st['norm_delta_fine'])
                        ladderr.append(dict(**common,**vec_fields(st['low'],'rich_')))
                        for k in range(10):
                            d01=rh[coarse][k]-rh[med][k];d12=rh[med][k]-rh[fine][k]
                            nscale=max(np.linalg.norm(base),np.linalg.norm(rh[fine]),1e-30)
                            comp_resolved=bool(abs(d01)>args.relative_resolution_floor*nscale and abs(d12)>args.relative_resolution_floor*nscale)
                            cp=float(np.log2(abs(d01/d12))) if comp_resolved else None
                            comp.append(dict(case_id=label,id=name,anchor_xi=xi,branch=branch,
                                fraction_of_common_time=frac,component=k,focus_component=int(k in KEY_COMPONENTS),
                                coarse_h=coarse,medium_h=med,fine_h=fine,
                                observed_component_order=finite(cp),order_resolved=int(comp_resolved),
                                coarse_minus_medium=float(d01),medium_minus_fine=float(d12),
                                component_richardson=float(st['low'][k]),
                                component_jac_parent=float(base[k]),component_jac_small=float(refs['small'][k]),
                                component_jac_large=float(refs['large'][k]),
                                signed_error_parent=float(st['low'][k]-base[k]),
                                signed_error_small=float(st['low'][k]-refs['small'][k]),
                                signed_error_large=float(st['low'][k]-refs['large'][k]),
                                normalized_abs_error_parent=float(abs(st['low'][k]-base[k])/nscale),
                                squared_disagreement_share=float(error_shares(st['low']-base)[k])))
                    for coarse,med,fine in TRIPLES_JAC:
                        st=triple_stats(jr[coarse],jr[med],jr[fine],floor=args.relative_resolution_floor)
                        ladderj.append(dict(**base_cols,coarse_h=coarse,medium_h=med,fine_h=fine,
                             observed_order=finite(st['order']),order_resolved=int(st['resolved']),
                             order4_consistent=int(st['order4_consistent']),
                             cos_successive_differences=st['cos_differences'],
                             rich_pair_disagreement=st['richardson_between_pairs'],
                             relative_rich_fine_to_jac_parent=relative(st['low'],base),
                             **vec_fields(st['low'],'rich_jac_')))
                    # Fixed h=.008 uses NO per-state radius selection. Reported alongside
                    # test-blind h=.004/.002 Richardson against three Jacobian references.
                    focus=triple_stats(rh[.008],rh[.004],rh[.002],floor=args.relative_resolution_floor)
                    rr=focus['low']
                    snaprows.append(dict(**base_cols,fixed_rhs_h=.008,
                        fixed_raw_crosspath=relative(rh[.008],base),
                        fixed_richardson_crosspath=relative(rr,base),
                        jac_small_parent_disagreement=relative(refs['small'],base),
                        jac_large_parent_disagreement=relative(refs['large'],base),
                        focal_richardson_pair_disagreement=focus['richardson_between_pairs'],
                        focal_empirical_order=finite(focus['order']),
                        focal_order4_consistent=int(focus['order4_consistent']),
                        parent_selected_h=float(olds['selected_rhs_h']),
                        parent_selected_disagreement=float(olds['selected_cross_path_error']),
                        parent_radius_not_used_to_select_fixed=int(True),
                        fraction_discrepancy_in_028=float(np.sum(error_shares(rr-base)[list(KEY_COMPONENTS)]))))
                print('V18_46',label,'snapshots',len(FRACS),'parent_reg',f'{max_reg:.2e}',flush=True)
    counts=(len(snaprows),len(rrows),len(jrows),len(ladderr),len(ladderj),len(comp),len(solver))
    expected=(36,432,288,288,216,2880,24)
    if counts!=expected or count!=25:raise RuntimeError(f'Incomplete ORIGINAL v18.46 outputs {counts} != {expected}')
    for name,rows in (
        ('original_v18_46_snapshot_summary.csv',snaprows),
        ('original_v18_46_rhs_radii.csv',rrows),
        ('original_v18_46_jac_radii.csv',jrows),
        ('original_v18_46_rhs_richardson_ladders.csv',ladderr),
        ('original_v18_46_jac_richardson_ladders.csv',ladderj),
        ('original_v18_46_component_ladders.csv',comp),
        ('original_v18_46_solver_steps.csv',solver)):
        output(name,rows)
    proof=dict(version='v18.46',status='ORIGINAL_Q550_COMPONENT_RICHARDSON_CONSISTENCY_COMPLETED',
        original_integrations=count,original_cases=12,original_snapshots=36,rhs_radius_rows=432,
        jac_radius_rows=288,rhs_ladder_rows=288,jac_ladder_rows=216,
        component_ladder_rows=2880,solver_rows=24,
        parent_sha256=auth['parent_sha256'],parent_verified_members=auth['verified_members'],
        max_v18_45_vector_regression=max_reg,max_within_parent_support_fraction=max_support,
        original_ode_sha256_verified=True,source_kappa0_unchanged=True,
        no_symbolic_hessian_claimed=True,no_certified_hessian=True,no_certified_second_jet=True,
        no_certified_curvature=True,no_certified_stiffness=True,no_material_metric_certified=True,
        no_physical_constraint_certified=True,preflight_only=False,
        method=dict(rhs_radii=list(RHS_H),jacobian_targets=list(JAC_H),
            rhs_ladders=[list(v) for v in TRIPLES_RHS],jac_ladders=[list(v) for v in TRIPLES_JAC],
            nominal_order=4,order_identification='observed successive difference norm ratio; not a convergence theorem',
            richardson='two successive order-4 Richardson estimates compared for consistency',
            component_focus=list(KEY_COMPONENTS),component_errors='signed and magnitude errors relative to THREE numerical J consensuses',
            primary_fixed_h=0.008,primary_fixed_ladder=[0.008,0.004,0.002],
            radius_selection='no selecting best radius using held-out data',
            ODE='original SHA-pinned Q550, same kappa0 and seeds',
            caution='Numerical agreement and Richardson consistency are not exact Hessian, stiffness, material metric or curvature certifications'))
    (R/'results/original_v18_46_proof.json').write_text(json.dumps(proof,indent=2,allow_nan=False)+'\n')
    print('ORIGINAL_V18_46_COMPLETED',count,counts,'parent_reg',max_reg,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--base-root',type=Path,default=pinned.DEFAULT_BASE)
    parser.add_argument('--rtol',type=float,default=3e-12)
    parser.add_argument('--atol',type=float,default=3e-13)
    parser.add_argument('--max-step',type=float,default=2.5e-7)
    parser.add_argument('--max-parent-regression',type=float,default=2e-6)
    parser.add_argument('--relative-resolution-floor',type=float,default=5e-12)
    run(parser.parse_args())
