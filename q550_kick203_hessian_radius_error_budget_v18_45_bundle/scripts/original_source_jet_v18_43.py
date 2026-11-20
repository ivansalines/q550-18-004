#!/usr/bin/env python3
"""v18.43: ORIGINAL Q550 semi-analytic source flow jet and precision audit.

37 independent original integrations: one baseline + 3 material trajectories for
12 families. No solver tolerance change in baseline, no kappa0 change; no physical
constraint/metric certified. Exact polynomial chain rule; directional Jacobian
and geometric chart derivatives numerical and explicitly audited. Retains phase
and material state, never equates a reduced-phase return to full-state return.
"""
from pathlib import Path
import sys,json,csv,argparse
import numpy as np
from scipy.integrate import solve_ivp
R=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(R/'scripts'),str(R/'reference')]
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
from manifold_math_v18_32 import project_to_level,features,features_material_gradient
from near_surface_math_v18_35 import calibration_radius
from residual_gradient_math_v18_37 import source_geometry,unit
from coupled_scaling_math_v18_38 import tangent_to_normal
from seed_genealogy_v18_38 import index_archived_attempts,verify_regenerated_attempts
from tangent_source_math_v18_39 import normal_frame,minimal_rotation
from analytic_source_jet_math_v18_43 import material_source_jet,moving_coords_and_jet,curvature,five_point_d1,five_point_d2,time_step_from_scale
import q550_state_impulse_response_v13 as pinned
H=round(1./3.,10)
FRACS=(0.,.125,.25,.5,1.)
TARGETS=(2e-3,5e-4,1.25e-4)
WINDOWS=(1.,.5,.25,.125)
STEPS=(8,16,32)

def write_csv(fn,rows):
    if not rows:raise RuntimeError('Empty output: '+fn)
    with (R/'results'/fn).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def case_key(row):return (row['id'],round(float(row['anchor_xi']),8),int(row['branch']))

def run(args):
    proof42,old42,profiles42,windows42,stencils42,ladder42,auth43=read_parent_v43()
    read_parent_v42();read_parent_v41();read_parent_v40();read_parent_v39();read_parent_v38()
    _,cases36,_,attempts36,_=read_parent_v37()
    read_parent_v36();read_parent_v35()
    chk,_,model33,_,_,_=read_parent()
    if not {'standard','tight'} <= set(chk):raise RuntimeError('Historical checkpoint tiers absent')
    original=select_original_checkpoint(chk)
    f,jac=model(args.base_root,original)
    anchorstates=anchored_flow(f,np.asarray(original['source_state_11D'],float),ANCHORS)
    cen=np.asarray(model33['center'],float);scale=np.asarray(model33['scale'],float)
    th0=float(model33['theta0'])
    train,*_=evaluate_samples(f,anchorstates,scale)
    candidates={a['id']:a for a in model33['candidates']}
    original_cases={case_key(r):r for r in old42}
    parent_vectors={(r['case_id'],round(float(r['fraction_of_common_time']),10)):np.array([float(r['v'+str(i)]) for i in range(9)]) for r in profiles42}
    cases36map={case_key(r):r for r in cases36}
    attempts36map=index_archived_attempts(attempts36)
    rng=np.random.default_rng(1836)
    rows=[];seconds=[];families=[];scalechecks=[];solver_steps=[]
    original_solves=1; max_replay=0.;max_seed_discrepancy=0.;max_support_ratio=0.
    for name in NAMES:
        candidate=candidates[name];q=np.asarray(candidate['q'],float);opts=candidate['opts'];mu=float(opts['lam'])
        if mu!=-1 or float(opts['mod'])!=0:raise RuntimeError('Unapproved cofactor')
        def geom(y):return source_geometry(y,f(0.,y),jac(y),q,cen,scale,th0,opts['degree'],opts['phase'],mu)
        for i in ANCHOR_INDICES:
            anchor=anchorstates[i];a=float(ANCHORS[i]);radius=float(calibration_radius(train[i*56:(i+1)*56],anchor,scale)['radius_max'])
            for branch in (0,1):
                ref=(name,round(a,8),branch)
                old=original_cases.get(ref)
                if old is None:raise RuntimeError('Original v18.42 case not found')
                t_common=float(old['common_within_support_time'])
                if not (0<t_common<=1e-5):raise RuntimeError('Invalid parent support time')
                y0,attempts=projected_independent_seed(anchor,q,cen,scale,th0,opts,radius,rng,branch)
                verify_regenerated_attempts(ref,attempts,cases36map,attempts36map)
                g0=geom(y0)
                tangent=tangent_to_normal(g0['g'],np.roll(np.arange(1.,11.),branch+2))
                dist0=float(np.linalg.norm((y0[:10]-anchor[:10])/scale))
                margin=.8*radius-dist0
                if margin<=0:raise RuntimeError('Seed starts outside support')
                perturbation=min(.04*radius,.22*margin)
                ys={}
                for alpha in (-H,0.,H):
                    y=y0.copy();y[:10]+=alpha*perturbation*tangent*scale
                    if alpha:
                        y,_,ok=project_to_level(y,q,cen,scale,th0,opts['degree'],opts['phase'],max_scaled_step=.3)
                        if not ok:raise RuntimeError('Projection of perturbation failed')
                    if abs(float(features(y,cen,scale,th0,opts['degree'],opts['phase'])@q))>1e-8:raise RuntimeError('Not on candidate C=0')
                    if np.linalg.norm((y[:10]-anchor[:10])/scale)>.8*radius or abs(y[10]-anchor[10])>.2:
                        raise RuntimeError('Perturbed seed outside authentic support')
                    sol=solve_ivp(f,(0.,t_common),y,method='DOP853',rtol=args.rtol,atol=args.atol,
                        max_step=args.max_step,dense_output=True)
                    if not sol.success:raise RuntimeError('Original Q550 source-jet integration failed: '+sol.message)
                    ys[alpha]=sol;original_solves+=1
                    dt=np.diff(sol.t)
                    solver_steps.append(dict(case_id=f'{name}|xi={a:g}|branch={branch}',kind='perturbed_seed',alpha=alpha,
                        nfev=int(sol.nfev),accepted_steps=len(dt),min_dt=float(np.min(dt)),
                        max_dt=float(np.max(dt)),median_dt=float(np.median(dt)),
                        step_dynamic_range=float(np.max(dt)/max(np.min(dt),1e-300)),
                        rtol=args.rtol,atol=args.atol,max_step=args.max_step))
                case_id=f'{name}|xi={a:g}|branch={branch}'
                n0=unit(g0['g']);frame0=normal_frame(n0)
                # Source functions evaluated at ORIGINAL independently integrated states.
                cache={}
                def response(s,target=5e-4):
                    if not(-1e-12<=s<=1+1e-12):raise RuntimeError('Outside common diagnostic support')
                    s=float(s);t=t_common*s;key=(round(s,14),target)
                    if key in cache:return cache[key]
                    jets={alpha:material_source_jet(np.asarray(ys[alpha].sol(t),float),f,jac,q,cen,scale,th0,
                        opts['degree'],opts['phase'],mu,scaled_target=target) for alpha in (-H,0.,H)}
                    coords={alpha:moving_coords_and_jet(jets[alpha],jets[0.],n0,frame0) for alpha in (-H,H)}
                    v=1.5*(coords[H][0]-coords[-H][0])
                    vd=1.5*(coords[H][1]-coords[-H][1])*t_common  # derivative wrt s=t/t_common
                    gauge=max(coords[H][2],coords[-H][2])
                    if not np.isfinite(v).all() or not np.isfinite(vd).all():raise RuntimeError('Nonfinite response jet')
                    cache[key]=(v,vd,jets,gauge)
                    return cache[key]
                # Recompute the v18.42 source response in its declared tangent gauge.
                for s in FRACS:
                    v,vd,jets,gauge=response(s)
                    prior=parent_vectors.get((case_id,round(s,10)))
                    if prior is None:raise RuntimeError('No original v18.42 profile at required time')
                    gap=float(np.linalg.norm(v-prior))
                    max_replay=max(max_replay,gap)
                    if gap>5e-9+2e-5*np.linalg.norm(prior):raise RuntimeError('Original v18.42 9D source replay failed')
                    # Direct numerical derivative of original dense-response samples: cross-check.
                    s_step=min(1/128.,(1.-s)/2,s/2)
                    if s_step>1e-6:
                        finite=(response(s+s_step)[0]-response(s-s_step)[0])/(2*s_step)
                        first_rel=float(np.linalg.norm(vd-finite)/max(np.linalg.norm(vd),np.linalg.norm(finite),1e-30))
                    else:first_rel=None
                    rows.append(dict(case_id=case_id,id=name,anchor_xi=a,branch=branch,fraction_of_common_time=s,
                        delta_xi=t_common*s,phase_initial_lifted=float(y0[10]),phase_current_lifted=float(ys[0.].sol(t_common*s)[10]),
                        phase_change_lifted=float(ys[0.].sol(t_common*s)[10]-y0[10]),
                        within_parent_support=True,source_norm=float(np.linalg.norm(v)),
                        first_jet_norm_s=float(np.linalg.norm(vd)),
                        first_jet_fd_dense_relative_disagreement=first_rel,
                        gauge_jet_norm_per_original_xi=gauge,original_v18_42_vector_replay_abs=gap,
                        **{'v'+str(j):float(v[j]) for j in range(9)},
                        **{'jet'+str(j):float(vd[j]) for j in range(9)}))
                    for target in TARGETS:
                        _,vj,_,_=response(s,target)
                        if target!=5e-4:
                            scalechecks.append(dict(case_id=case_id,anchor_xi=a,branch=branch,source_fraction=s,
                                scaled_directional_target=target,
                                jet_disagreement_abs=float(np.linalg.norm(vj-vd)),
                                jet_disagreement_rel=float(np.linalg.norm(vj-vd)/max(np.linalg.norm(vd),1e-30))))
                # Directly audit support using the same material reference section.
                refseed=anchor.copy();refseed[10]=y0[10]  # Match material reference CLOCK phase as in v18.38-42.
                rsol=solve_ivp(f,(0.,t_common),refseed,method='DOP853',rtol=args.rtol,atol=args.atol,
                    max_step=args.max_step,dense_output=True)
                if not rsol.success:raise RuntimeError('Original anchor reference propagation failed')
                original_solves+=1
                dtr=np.diff(rsol.t)
                solver_steps.append(dict(case_id=case_id,kind='anchor_reference',alpha='',
                    nfev=int(rsol.nfev),accepted_steps=len(dtr),min_dt=float(np.min(dtr)),
                    max_dt=float(np.max(dtr)),median_dt=float(np.median(dtr)),
                    step_dynamic_range=float(np.max(dtr)/max(np.min(dtr),1e-300)),
                    rtol=args.rtol,atol=args.atol,max_step=args.max_step))
                for alpha in (-H,0.,H):
                    for s in FRACS:
                        d=float(np.linalg.norm((ys[alpha].sol(s*t_common)[:10]-rsol.sol(s*t_common)[:10])/scale))
                        max_support_ratio=max(max_support_ratio,d/radius)
                        if d>radius*(1+1e-6):raise RuntimeError('New original trajectory left authenticated support')
                # Per-case nested first-jet middle points and a cross-check with
                # time-sampled second differences. No exact second analytic claim.
                for w in WINDOWS:
                    mid=w/2
                    for n in STEPS:
                        h=w/n
                        values={k:response(mid+k*h) for k in (-2,-1,0,1,2)}
                        d1_sample=five_point_d1(*(values[k][0] for k in (-2,-1,1,2)),h)
                        d2_sample=five_point_d2(*(values[k][0] for k in (-2,-1,0,1,2)),h)
                        d1_jet=values[0][1]
                        d2_from_jet=five_point_d1(*(values[k][1] for k in (-2,-1,1,2)),h)
                        second_err=float(np.linalg.norm(d2_sample-d2_from_jet)/max(np.linalg.norm(d2_sample),np.linalg.norm(d2_from_jet),1e-30))
                        seconds.append(dict(case_id=case_id,id=name,anchor_xi=a,branch=branch,window_fraction=w,
                            stencil_subdivision=n,step_fraction=h,
                            second_fd_norm=float(np.linalg.norm(d2_sample)),
                            second_from_first_jet_norm=float(np.linalg.norm(d2_from_jet)),
                            second_relative_disagreement=second_err,
                            first_jet_to_first_fd_disagreement=float(np.linalg.norm(d1_jet-d1_sample)/max(np.linalg.norm(d1_jet),np.linalg.norm(d1_sample),1e-30)),
                            curvature_fd_corrected=curvature(d1_sample,d2_sample),
                            curvature_jet_differenced=curvature(d1_jet,d2_from_jet),
                            original_v18_42_wrong_curvature_column_not_used=True))
                famrows=[r for r in rows if r['case_id']==case_id]
                secrows=[r for r in seconds if r['case_id']==case_id]
                families.append(dict(case_id=case_id,id=name,anchor_xi=a,branch=branch,
                    common_within_support_time=t_common,
                    max_original_v18_42_source_replay_abs=max(r['original_v18_42_vector_replay_abs'] for r in famrows),
                    max_first_jet_dense_disagreement=max((r['first_jet_fd_dense_relative_disagreement'] or 0) for r in famrows),
                    median_second_relative_disagreement=float(np.median([r['second_relative_disagreement'] for r in secrows])),
                    max_second_relative_disagreement=max(r['second_relative_disagreement'] for r in secrows),
                    median_solver_stencil_curvature=float(np.median([r['curvature_fd_corrected'] for r in secrows])),
                    max_phase_change_lifted=max(abs(r['phase_change_lifted']) for r in famrows)))
                print('V18_43',case_id,'T',f'{t_common:.3e}',
                      'max replay',f'{max(r["original_v18_42_vector_replay_abs"] for r in famrows):.3e}',
                      'median second-jet mismatch',f'{families[-1]["median_second_relative_disagreement"]:.3g}',flush=True)
    if not(len(families)==12 and len(rows)==60 and len(seconds)==144 and len(scalechecks)==120 and len(solver_steps)==48):
        raise RuntimeError('Original Q550 v18.43 incomplete')
    R.joinpath('results').mkdir(exist_ok=True)
    write_csv('original_v18_43_source_first_jets.csv',rows)
    write_csv('original_v18_43_second_jet_windows.csv',seconds)
    write_csv('original_v18_43_directional_step_stability.csv',scalechecks)
    write_csv('original_v18_43_family_summary.csv',families)
    write_csv('original_v18_43_solver_step_diagnostics.csv',solver_steps)
    result=dict(version='v18.43',status='ORIGINAL_Q550_SOURCE_JET_PRECISION_AUDIT_COMPLETED',
        original_integrations=original_solves,original_cases=len(families),first_jet_rows=len(rows),
        second_window_rows=len(seconds),directional_step_rows=len(scalechecks),solver_step_rows=len(solver_steps),
        parent_v18_42_sha256=auth43['parent_sha256'],parent_v18_42_verified_members=auth43['verified_members'],
        max_parent_vector_replay_abs=max_replay,max_within_support_radius_fraction=max_support_ratio,
        source_kappa0_unchanged=True,original_ode_sha256_verified=True,
        no_certified_analytic_second_jet=True,no_certified_curvature=True,
        no_material_metric_certified=True,no_physical_constraint_certified=True,
        method=dict(first_jet='Exact chain rule for candidate C, material source and projection; five-point directional Jacobian derivative and finite geometric chart derivative',
                    second_jet='Single five-point derivative of semi-analytic first jet, independently compared to two derivatives of original source samples',
                    old_v18_42_bug='Curvature stencil in v18.42 had dd5@dd5 instead of d5@d5. Recomputed independently here; original remains intact',
                    time_parameter='s=delta_xi/T_common',temporal_windows=list(WINDOWS),
                    sampled_amplitudes=[-1/3,0,1/3],directional_scaled_steps=list(TARGETS),
                    phase='Record phase variable and increment, not physical winding or material memory',
                    stiffness='Adaptive DOP853 accepted steps and function evaluations, not a stiffness proof or physical eigenvalue',
                    warning='No Q550 Hessian source code used, no exact second variation claimed; resolved only if stencil and step converge'),
        status_preflight_only=False)
    (R/'results/original_v18_43_proof.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('ORIGINAL_V18_43_COMPLETE',original_solves,len(rows),len(seconds),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--base-root',type=Path,default=pinned.DEFAULT_BASE)
    ap.add_argument('--rtol',type=float,default=3e-12)
    ap.add_argument('--atol',type=float,default=3e-13)
    ap.add_argument('--max-step',type=float,default=2.5e-7)
    run(ap.parse_args())
