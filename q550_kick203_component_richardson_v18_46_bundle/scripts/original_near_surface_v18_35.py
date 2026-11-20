#!/usr/bin/env python3
"""Q550 authentic near-surface micro-time transport validity, v18.35.

Replay the EXACT v18.34 selected projected seeds. Integrate the perturbed
trajectory, the phase-aligned MOVING reference trajectory, and the exact
cofactor residual accumulator together. Compare fixed-anchor distance to
co-moving distance and the support ball inferred from TRAINING-ONLY samples.

Neither a diagnostic training ball nor a well-closed residual identity is a
certified physical invariant manifold or an intrinsic material metric.
"""
import argparse,csv,io,json,sys
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'reference')]
from checkpoint_io_v18_35 import read_parent_v35
from checkpoint_io_v18_34 import read_parent
from original_material_invariants_v18_31 import select_original_checkpoint,model,ANCHORS
from original_invariant_manifold_v18_32 import anchored_flow
from original_cofactor_surface_return_v18_33 import evaluate_samples
from manifold_math_v18_32 import features,features_flow,cofactor,project_to_level
from near_surface_math_v18_35 import microgrid,diagnostic_distance,calibration_radius,refine_first_exit,nearest_training,weighted_exact_budget
import q550_state_impulse_response_v13 as original

NAMES=['phasecov_d2_lam-1.00_mod+0.00','phaseblind_d2_lam-1.00_mod+0.00']
TEND=.0125

def write_csv(p,rows):
    if not rows:raise RuntimeError('Cannot export empty result table')
    with p.open('w',newline='') as fo:
        writer=csv.DictWriter(fo,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def run(args):
    prior,prior_model,archived,auth=read_parent_v35()
    checkpoints,old,older_model,oldreturns,oldoverlap,prov=read_parent()
    d=select_original_checkpoint(checkpoints)
    f,_=model(args.base_root,d)
    bases=anchored_flow(f,np.asarray(d['source_state_11D'],float),ANCHORS)
    center=np.asarray(prior_model['center'],float);scale=np.asarray(prior_model['scale'],float);theta0=float(prior_model['theta0'])
    expect_scale=np.maximum.reduce([.025*np.abs(bases[3,:10]),.1*np.ptp(bases[:,:10],axis=0),np.full(10,1e-6)])
    if not np.allclose(center,bases[3,:10],rtol=5e-9,atol=1e-8) or not np.allclose(expect_scale,scale,rtol=5e-9,atol=1e-8):
        raise RuntimeError('v18.33/v18.34 authentic calibration scales not reproduced')
    if any(np.linalg.norm(np.asarray(x['q']))<1e-15 for x in prior_model['candidates']):raise RuntimeError('Zero candidate')
    train,trainf,test,testf,_,_=evaluate_samples(f,bases,scale)
    if len(train)!=392 or len(test)!=224:raise RuntimeError('Training/holdout split changed')
    candidates={it['id']:it for it in prior_model['candidates']}
    archived_rows={(r['id'],float(r['seed_anchor']),int(r['seed_index'])):r
                   for r in csv.DictReader(io.StringIO(archived)) if abs(float(r['delta_xi'])-TEND)<1e-12}
    if len(archived_rows)!=12:raise RuntimeError('Original v18.34 initial sample count changed')
    times=microgrid(TEND)
    micro=[];events=[];cases=[];integrations=1;max_history_error=0.;max_identity_error=0.;
    tighter_comparisons=[]
    for name in NAMES:
        it=candidates[name];q=np.asarray(it['q'],float);opts=it['opts']
        if abs(float(opts['mod']))>1e-15:raise RuntimeError('v18.35 method currently requires constant cofactor')
        mu=float(opts['lam'])
        def cfun(y):return float(features(y,center,scale,theta0,opts['degree'],opts['phase'])@q)
        def residual(y,dy):return float(features_flow(y,dy,center,scale,theta0,opts['degree'],opts['phase'])@q-mu*cfun(y))
        for k in (1,3,5):
            anchor=float(ANCHORS[k]);local_train=train[k*56:(k+1)*56]
            calib=calibration_radius(local_train,bases[k],scale)
            for idx in (0,1):
                start,projection,ok=project_to_level(test[k*32+idx],q,center,scale,theta0,opts['degree'],opts['phase'])
                if not ok:raise RuntimeError('Original v18.34 projected seed cannot be reproduced')
                phase_aligned=bases[k].copy()
                phase_aligned[10]=start[10]  # Compare material motion at identical forcing-clock phase.
                initial_phase_shift=float(start[10]-bases[k,10])
                x0=np.r_[start,phase_aligned,0.]
                def rhs(t,x):
                    ys=x[:11];yr=x[11:22]
                    fs=f(t,ys);fr=f(t,yr)
                    source=residual(ys,fs)
                    return np.r_[fs,fr,np.exp(-mu*t)*source]
                sol=solve_ivp(rhs,(0.,TEND),x0,method='DOP853',dense_output=True,
                              rtol=args.rtol,atol=args.atol,max_step=args.max_step)
                if not sol.success:raise RuntimeError('Original Q550 micro-time integration failed: '+sol.message)
                integrations+=1
                states=np.array([sol.sol(float(t)) for t in times]);seed_states=states[:,:11];refs=states[:,11:22]
                C=np.array([cfun(y) for y in seed_states])
                F=np.array([f(0.,y) for y in seed_states])
                r=np.array([residual(y,dy) for y,dy in zip(seed_states,F)])
                exact_budget=weighted_exact_budget(C[0],times,mu,states[:,22])
                direct=C-np.exp(mu*times)*C[0]
                closure=direct-exact_budget
                fixed=np.linalg.norm((seed_states[:,:10]-start[:10])/scale,axis=1)
                moving=np.linalg.norm((seed_states[:,:10]-refs[:,:10])/scale,axis=1)
                train_nearest=np.array([nearest_training(y,local_train,scale) for y in seed_states])
                med=float(np.median(np.linalg.norm((local_train[:,:10]-bases[k,:10])/scale,axis=1)))
                stable_start_dist=float(moving[0])
                for t,x,yref,cc,rr,dd,bb,df,dm,dn,cl in zip(times,seed_states,refs,C,r,direct,exact_budget,fixed,moving,train_nearest,closure):
                    micro.append(dict(id=name,anchor_xi=anchor,seed_index=idx,delta_xi=float(t),C=float(cc),source_r=float(rr),
                        observed_transport_defect=float(dd),integrated_source_defect=float(bb),closure_error=float(cl),
                        distance_fixed_anchor=float(df),distance_co_moving=float(dm),distance_nearest_local_training=float(dn),
                        local_training_radius_95=calib['radius_95'],local_training_radius_max=calib['radius_max'],
                        material_phase_offset_from_moving=float(x[10]-yref[10]),
                        moving_reference_phase_advance=float(yref[10]-bases[k,10]),
                        inside_calibrated_moving_ball=bool(dm<=calib['radius_max']),
                        inside_calibration_phase_band=bool(abs(initial_phase_shift)<=.25),
                        within_joint_diagnostic_local_support=bool(dm<=calib['radius_max'] and abs(initial_phase_shift)<=.25)))
                max_identity_error=max(max_identity_error,float(np.max(abs(closure))))
                hist=archived_rows[(name,anchor,idx)]
                history_error=abs(C[-1]-float(hist['C']))
                max_history_error=max(max_history_error,history_error)
                if history_error>5e-6*max(1.,abs(float(hist['C']))):
                    raise RuntimeError('Authentic v18.34 trajectory mismatch '+name+' '+str(anchor)+' '+str(idx)+' '+str(history_error))
                for metric,arr in [('fixed_initial',fixed),('co_moving_baseline',moving)]:
                    for threshold_name,threshold in [('local_train_r95',calib['radius_95']),('local_train_rmax',calib['radius_max']),('unit_1',1.),('unit_3',3.),('unit_10',10.)]:
                        def continuous_distance(t,metric=metric):
                            ss=sol.sol(t);ref=start if metric=='fixed_initial' else ss[11:22]
                            return diagnostic_distance(ss[:11],ref,scale)
                        status,root,left,right=refine_first_exit(times,arr,threshold,continuous_distance)
                        events.append(dict(id=name,anchor_xi=anchor,seed_index=idx,diagnostic_frame=metric,
                            threshold_name=threshold_name,threshold=float(threshold),initial_distance=float(arr[0]),
                            crossing_status=status,first_outward_crossing_xi=root,first_bracket_left=left,first_bracket_right=right,
                            final_distance=float(arr[-1])))
                early_index=int(np.searchsorted(times,.0001,side='right')-1)
                early_defect=float(abs(direct[early_index]))
                cases.append(dict(id=name,anchor_xi=anchor,seed_index=idx,projected_seed_shift_scaled=float(projection),
                    calibration_radius_95=float(calib['radius_95']),calibration_radius_max=float(calib['radius_max']),
                    distance_co_moving_initial=stable_start_dist,
                    initial_phase_shift_from_central=float(initial_phase_shift),
                    inside_training_phase_band=bool(abs(initial_phase_shift)<=.25),
                    inside_joint_calibrated_region_initial=bool(stable_start_dist<=calib['radius_max'] and abs(initial_phase_shift)<=.25),
                    starts_inside_training_ball=bool(stable_start_dist<=calib['radius_max']),
                    distance_fixed_final=float(fixed[-1]),distance_co_moving_final=float(moving[-1]),
                    distance_nearest_train_final=float(train_nearest[-1]),
                    source_r_initial=float(r[0]),source_r_end=float(r[-1]),
                    defect_at_first_0p0001=float(early_defect),
                    defect_end=float(direct[-1]),budget_end=float(exact_budget[-1]),
                    max_integral_identity_error=float(np.max(abs(closure))),
                    historical_endpoint_C_error=float(history_error),
                    material_phase_offset_max=float(np.max(abs(seed_states[:,10]-refs[:,10])))))
                print('MICRO',name,'anchor',anchor,'seed',idx,'initial moving',f'{stable_start_dist:.5g}',
                      'fixed/end',f'{fixed[-1]:.5g}','moving/end',f'{moving[-1]:.5g}',
                      'defect',f'{direct[-1]:.6e}',flush=True)
                if name==NAMES[0] and anchor==3.5 and idx in (0,1):
                    tight=solve_ivp(rhs,(0.,TEND),x0,method='DOP853',dense_output=True,
                               rtol=min(args.rtol,2e-13),atol=min(args.atol,2e-14),max_step=min(args.max_step,2.5e-5))
                    if not tight.success:raise RuntimeError('Tight check failed')
                    integrations+=1
                    tight_final=tight.sol(TEND)
                    tighter_comparisons.append(dict(id=name,anchor_xi=anchor,seed_index=idx,
                        material_state_difference_L2=float(np.linalg.norm(tight_final[:10]-states[-1,:10])),
                        accumulated_source_difference=float(abs(tight_final[22]-states[-1,22])),
                        endpoint_C_difference=float(abs(cfun(tight_final[:11])-C[-1]))))
    out=ROOT/'results';out.mkdir(exist_ok=True)
    write_csv(out/'original_v18_35_microprofiles.csv',micro)
    write_csv(out/'original_v18_35_first_exits.csv',events)
    write_csv(out/'original_v18_35_case_summary.csv',cases)
    report=dict(version='v18.35',status='ORIGINAL_Q550_NEAR_SURFACE_MOVING_FRAME_AND_DEFECT_BUDGET_COMPLETED',
        original_ode_input_sha256_verified=True,parent_v18_34_sha256=auth['parent_sha'],
        parent_v18_34_verified_members=auth['verified_members'],
        original_ode_integrations=integrations,case_count=len(cases),micro_sample_count=len(micro),
        first_exit_diagnostics=len(events),micro_horizon_xi=TEND,
        max_historical_replay_C_absolute_error=max_history_error,
        max_original_integrated_budget_identity_error=max_identity_error,
        tighter_comparison=tighter_comparisons,
        frozen_candidate_surfaces=True,calibration_uses_training_only=True,
        original_v18_34_seed_reproduction=True,
        measured_fixed_anchor_and_comoving_reference=True,
        calibrated_region_is_diagnostic_not_certified_physical=True,
        moving_reference_has_seed_phase_alignment=True,
        joint_training_region_requires_material_ball_and_phase_band=True,
        if_outside_at_start_no_exit_claim=True,
        no_physical_invariant_manifold_certified=True,no_material_metric_certified=True,
        no_material_return_certified=True,
        methods=dict(solver='DOP853',rtol=float(args.rtol),atol=float(args.atol),max_step=float(args.max_step),
            time_grid_count=len(times),cofactor='constant -1',integrated_defect='augmented initial-value quadrature'))
    (out/'original_v18_35_proof.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('ORIGINAL_V18_35_COMPLETED',integrations,'integrations',len(micro),'micro samples',len(events),'crossing diagnostics')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-root',type=Path,default=original.DEFAULT_BASE)
    p.add_argument('--rtol',type=float,default=3e-12)
    p.add_argument('--atol',type=float,default=3e-13)
    p.add_argument('--max-step',type=float,default=1.25e-4)
    run(p.parse_args())
