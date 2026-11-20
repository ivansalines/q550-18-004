#!/usr/bin/env python3
"""Original Q550 v18.47: same 36 states, three different numerical Hessian kernels.

Replays the 25 original integrations. No symbolic Hessian, extra precision of
original RHS, intrinsic metric, true curvature or material invariant is claimed.
"""
import sys, csv, json, argparse
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
R=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(R/'scripts'),str(R/'reference')]
from checkpoint_io_v18_47 import read_parent_v47
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
from dual_hessian_jet_math_v18_44 import mixed_rhs_hessian_contraction,dot_B_transpose_g_from_jacobian
from kernel_compensation_math_v18_47 import RHS_RADII,ACCEL_RADII,KEY,mixed_compensated,polarization_rhs,accel_gradient_identity,relative,shares,vector_columns
import q550_state_impulse_response_v13 as pinned
FRACS=(.125,.5,1.)

def key(r):return (r['id'],round(float(r['anchor_xi']),8),int(r['branch']))
def snkey(r):return (r['case_id'],round(float(r['fraction_of_common_time']),8))
def output(name,rows):
    if not rows:raise RuntimeError('No output rows '+name)
    with (R/'results'/name).open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
def oldvec(r,prefix):return np.array([float(r[f'{prefix}{i}']) for i in range(10)])

def run(args):
    pp,oldr,oldj,oldsn,auth=read_parent_v47()
    read_parent_v46();read_parent_v45();read_parent_v44();
    _,old42,*_=read_parent_v43()
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
    cases36map={key(r):r for r in cases36};attemptsmap=index_archived_attempts(attempts36)
    old42map={key(r):r for r in old42}
    rhist={(snkey(r),round(float(r['rhs_h']),9)):r for r in oldr}
    jhist={(snkey(r),round(float(r['jac_target']),9)):r for r in oldj}
    shist={snkey(r):r for r in oldsn}
    rows=[];components=[];snapshots=[];solver=[]
    rng=np.random.default_rng(1836)
    count=1;maxreg=0.;maxsupport=0.
    for name in NAMES:
        obj=candidates[name];q=np.asarray(obj['q'],float);opts=obj['opts'];mu=float(opts['lam'])
        if mu!=-1.0 or float(opts['mod'])!=0:raise RuntimeError('Original candidate drifted')
        for i in ANCHOR_INDICES:
            anchor=anchors[i];xi=float(ANCHORS[i]);radius=float(calibration_radius(train[i*56:(i+1)*56],anchor,scale)['radius_max'])
            for branch in (0,1):
                ck=(name,round(xi,8),branch);label=f'{name}|xi={xi:g}|branch={branch}'
                old=old42map.get(ck)
                if old is None:raise RuntimeError('Old reference time missing')
                T=float(old['common_within_support_time'])
                y0,tries=projected_independent_seed(anchor,q,center,scale,theta0,opts,radius,rng,branch)
                verify_regenerated_attempts(ck,tries,cases36map,attemptsmap)
                sol=solve_ivp(f,(0,T),y0,method='DOP853',rtol=args.rtol,atol=args.atol,max_step=args.max_step,dense_output=True)
                if not sol.success:raise RuntimeError('Original center solver failed '+sol.message)
                count+=1
                solver.append(dict(case_id=label,kind='center_seed',nfev=sol.nfev,accepted_steps=len(sol.t)-1))
                yr=anchor.copy();yr[10]=y0[10]
                ref=solve_ivp(f,(0,T),yr,method='DOP853',rtol=args.rtol,atol=args.atol,max_step=args.max_step,dense_output=True)
                if not ref.success:raise RuntimeError('Original reference solver failed '+ref.message)
                count+=1;solver.append(dict(case_id=label,kind='phase_reference',nfev=ref.nfev,accepted_steps=len(ref.t)-1))
                for frac in FRACS:
                    t=float(frac*T);y=np.asarray(sol.sol(t),float);fy=np.asarray(f(0.,y),float);J=np.asarray(jac(y),float)
                    if np.linalg.norm(J[10,:10]*scale)>1e-7:raise RuntimeError('Clock/material Jacobian relation invalid')
                    sd=float(np.linalg.norm((y[:10]-ref.sol(t)[:10])/scale))/radius;maxsupport=max(maxsupport,sd)
                    if sd>1+1e-6:raise RuntimeError('Snapshot outside original support')
                    geo=source_geometry(y,fy,J,q,center,scale,theta0,opts['degree'],opts['phase'],mu)
                    sk=(label,round(float(frac),8));oldsum=shist.get(sk)
                    if oldsum is None or abs(float(oldsum['delta_xi'])-t)>2e-12 or abs(float(oldsum['support_radius_fraction'])-sd)>2e-6:
                        raise RuntimeError('v18.46 original snapshot replay mismatch')
                    base=dict(case_id=label,id=name,anchor_xi=xi,branch=branch,fraction_of_common_time=frac,
                              delta_xi=t,phase_lifted=float(y[10]),phase_advance_lifted=float(y[10]-y0[10]),support_radius_fraction=sd)
                    refer=oldvec(jhist[(sk,.001)],'jac_')
                    summary=dict(**base)
                    for h in RHS_RADII:
                        hist=rhist.get((sk,round(h,9)))
                        if hist is None:raise RuntimeError('Missing historical RHS vector')
                        raw=mixed_rhs_hessian_contraction(y,fy,f,scale,geo['g'],h)
                        expected=oldvec(hist,'rhs_');reg=relative(raw,expected);maxreg=max(maxreg,reg)
                        if reg>args.max_parent_regression:raise RuntimeError(f'Historic mixed RHS mismatch {label} {h}: {reg}')
                        cmp=mixed_compensated(y,fy,f,scale,geo['g'],h)
                        rev=mixed_compensated(y,fy,f,scale,geo['g'],h,reverse=True)
                        pol=polarization_rhs(y,fy,f,scale,geo['g'],h)
                        diff=cmp-refer
                        rows.append(dict(**base,kernel='mixed_compensated',h=h,relative_to_jac=relative(cmp,refer),
                             relative_to_historic=relative(cmp,raw),relative_to_reverse=relative(cmp,rev),
                             relative_to_polarization=relative(cmp,pol),**vector_columns('kernel_',cmp)))
                        rows.append(dict(**base,kernel='mixed_reversed',h=h,relative_to_jac=relative(rev,refer),
                             relative_to_historic=relative(rev,raw),relative_to_reverse=relative(rev,cmp),
                             relative_to_polarization=relative(rev,pol),**vector_columns('kernel_',rev)))
                        rows.append(dict(**base,kernel='polarized_rhs',h=h,relative_to_jac=relative(pol,refer),
                             relative_to_historic=relative(pol,raw),relative_to_reverse=relative(pol,rev),
                             relative_to_polarization=relative(pol,cmp),**vector_columns('kernel_',pol)))
                        for j in range(10):
                            components.append(dict(**base,kernel='mixed_compensated',h=h,component=j,focus=int(j in KEY),
                                kernel_value=float(cmp[j]),historical_value=float(raw[j]),jac_reference=float(refer[j]),
                                signed_kernel_minus_jac=float(diff[j]),signed_kernel_minus_historic=float(cmp[j]-raw[j]),
                                signed_kernel_minus_polarized=float(cmp[j]-pol[j]),
                                squared_error_share=float(shares(diff)[j])))
                        if h==.008:
                            summary.update(compensated_vs_raw=relative(cmp,raw),mixed_vs_polarization=relative(cmp,pol),
                                mixed_reverse_asymmetry=relative(cmp,rev),compensated_to_jac=relative(cmp,refer),
                                polarized_to_jac=relative(pol,refer),component_0_signed=float(diff[0]),
                                component_2_signed=float(diff[2]),component_8_signed=float(diff[8]),
                                squared_fraction_0_8=float(sum(shares(diff)[[0,8]])))
                    for h in ACCEL_RADII:
                        acc=accel_gradient_identity(y,fy,f,jac,scale,geo['g'],h)
                        rows.append(dict(**base,kernel='acceleration_gradient',h=h,
                            relative_to_jac=relative(acc,refer),relative_to_historic=relative(acc,oldvec(rhist[(sk,.008)],'rhs_')),
                            relative_to_reverse='',relative_to_polarization='',**vector_columns('kernel_',acc)))
                        for j in range(10):
                            components.append(dict(**base,kernel='acceleration_gradient',h=h,component=j,focus=int(j in KEY),
                                kernel_value=float(acc[j]),historical_value='',jac_reference=float(refer[j]),
                                signed_kernel_minus_jac=float(acc[j]-refer[j]),signed_kernel_minus_historic='',
                                signed_kernel_minus_polarized='',
                                squared_error_share=float(shares(acc-refer)[j])))
                        if h==.002:summary['acceleration_to_jac']=relative(acc,refer)
                    snapshots.append(summary)
                print('V18_47',label,'snapshot_count',len(FRACS),'max_reg',f'{maxreg:.2e}',flush=True)
    expected=(36,432,2160,24)
    actual=(len(snapshots),len(rows),len(components),len(solver))
    if actual!=expected or count!=25:raise RuntimeError(f'Incomplete ORIGINAL result {actual} vs {expected}, integration {count}')
    for fn,data in [('original_v18_47_kernel_vectors.csv',rows),('original_v18_47_component_budget.csv',components),
                    ('original_v18_47_snapshot_summary.csv',snapshots),('original_v18_47_solver_steps.csv',solver)]:output(fn,data)
    proof=dict(version='v18.47',status='ORIGINAL_Q550_HESSIAN_KERNEL_COMPENSATION_COMPLETED',
        original_integrations=count,original_cases=12,original_snapshots=36,
        kernel_vector_rows=len(rows),component_budget_rows=len(components),solver_rows=len(solver),
        original_ode_sha256_verified=True,source_kappa0_unchanged=True,
        parent_sha256=auth['parent_sha256'],parent_verified_members=auth['verified_members'],
        max_v18_46_vector_regression=maxreg,max_original_support_fraction=maxsupport,
        no_exact_hessian=True,no_certified_curvature=True,no_certified_stiffness=True,
        no_material_metric_certified=True,no_physical_constraint_certified=True,
        not_full_precision_RHS=True,preflight_only=False,
        method=dict(rhs_h=list(RHS_RADII),acceleration_h=list(ACCEL_RADII),focus_components=list(KEY),
            historic='16-node 4th-order mixed RHS',compensated='same mixed nodes, math.fsum weighted reductions',
            transpose='interchanged floating-point displacement summation order, not independent physics',
            polarization='5-point second directional derivatives of original scalar RHS, bilinear polarization',
            acceleration='differentiate J(y) F(y), subtract J(y)^2 F(y), using pinned original Jacobian and RHS',
            reference='original v18.46 jacobian h=.001 is numerical and NOT ground truth',
            caveat='All methods evaluate binary64 original model; no certified high precision or structural exact Hessian'))
    (R/'results/original_v18_47_proof.json').write_text(json.dumps(proof,indent=2,allow_nan=False)+'\n')
    print('ORIGINAL_V18_47_COMPLETED',count,actual,'parent_reg',maxreg,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base-root',type=Path,default=pinned.DEFAULT_BASE)
    p.add_argument('--rtol',type=float,default=3e-12);p.add_argument('--atol',type=float,default=3e-13)
    p.add_argument('--max-step',type=float,default=2.5e-7);p.add_argument('--max-parent-regression',type=float,default=2e-6)
    run(p.parse_args())
