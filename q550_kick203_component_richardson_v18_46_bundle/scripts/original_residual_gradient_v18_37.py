#!/usr/bin/env python3
"""Original Q550 residual-gradient, normal tilt and moving-level audit.

A transported cotangent is a mathematical reference, NOT a physical constraint.
The direct gradient of the same frozen diagnostic candidate is the test object.
Check both residual-driven level motion and non-homogeneous normal tilt.
"""
from pathlib import Path
import argparse,csv,json,sys
import numpy as np
from scipy.integrate import solve_ivp
R=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(R/'scripts'),str(R/'reference')]
from checkpoint_io_v18_37 import read_parent_v37
from checkpoint_io_v18_36 import read_parent_v36
from checkpoint_io_v18_35 import read_parent_v35
from checkpoint_io_v18_34 import read_parent
from original_material_invariants_v18_31 import select_original_checkpoint,model,ANCHORS
from original_invariant_manifold_v18_32 import anchored_flow
from original_cofactor_surface_return_v18_33 import evaluate_samples
from original_flow_normal_v18_36 import NAMES,ANCHOR_INDICES,projected_independent_seed
from manifold_math_v18_32 import features
from near_surface_math_v18_35 import calibration_radius
from residual_gradient_math_v18_37 import source_geometry,numerical_residual_gradient,projective_angle,continuous_normal_rhs,unit
import q550_state_impulse_response_v13 as original

T=1e-5

def csv_write(path,rows):
    if not rows:raise RuntimeError('Cannot write empty results: '+str(path))
    with path.open('w',newline='') as fh:
        writer=csv.DictWriter(fh,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def key(x):return (x['id'],round(float(x['anchor_xi']),8),int(x['branch']))

def run(args):
    original36, cases36, profiles36, attempts36, provenance=read_parent_v37()
    # Stage/read exact original genealogy; no regenerated or guessed seeds.
    read_parent_v36()
    read_parent_v35()
    checkpoints,proof33,model33,return33,overlap33,history=read_parent()
    d=select_original_checkpoint(checkpoints)
    f,jac=model(args.base_root,d)
    bases=anchored_flow(f,np.asarray(d['source_state_11D'],float),ANCHORS)
    center=np.asarray(model33['center'],float)
    scale=np.asarray(model33['scale'],float)
    theta0=float(model33['theta0'])
    if np.any(scale<=0):raise RuntimeError('Invalid original scaling')
    train,trainf,test,testf,_,_=evaluate_samples(f,bases,scale)
    candidates={c['id']:c for c in model33['candidates']}
    prior={key(c):c for c in cases36};priorattempt={key(c):c for c in attempts36}
    priorsample={key(c):c for c in profiles36 if float(c['delta_xi'])==0.}
    rng=np.random.default_rng(1836)
    tt=np.unique(np.r_[0.,np.geomspace(1e-11,T,64),np.linspace(0.,T,97)])
    gauges={'native':np.ones(10),'diag_stretch':np.geomspace(.7,1.6,10)}
    rows=[]; summaries=[]; finite_difference=[]; integrations=1
    clock_material_max=0.; max_seed_error=0.; max_clock_error=0.
    for name in NAMES:
        item=candidates[name];opts=item['opts'];q=np.asarray(item['q'],float)
        mu=float(opts['lam'])
        if abs(mu+1.)>1e-14 or abs(float(opts['mod']))>1e-14:raise RuntimeError('Unexpected candidate cofactor')
        for i in ANCHOR_INDICES:
            a=float(ANCHORS[i]); anchor=np.asarray(bases[i],float)
            radius=float(calibration_radius(train[i*56:(i+1)*56],anchor,scale)['radius_max'])
            for branch in (0,1):
                refkey=(name,round(a,8),branch)
                if refkey not in prior or refkey not in priorsample:raise RuntimeError('Missing genuine v18.36 seed')
                z,attempts=projected_independent_seed(anchor,q,center,scale,theta0,opts,radius,rng,branch)
                if len(attempts)!=int(prior[refkey]['number_of_seed_attempts']):raise RuntimeError('Seed regeneration attempt count mismatch')
                def geom(y):return source_geometry(y,f(0.,y),jac(y),q,center,scale,theta0,opts['degree'],opts['phase'],mu)
                g0=geom(z); archived=priorsample[refkey]
                seed_error=max(abs(g0['C']-float(archived['candidate_C'])),abs(g0['r']-float(archived['cofactor_source'])))
                max_seed_error=max(max_seed_error,seed_error)
                if seed_error>5e-7:raise RuntimeError('Original v18.36 seed not faithfully reproduced')
                base=anchor.copy();base[10]=z[10]
                # Verify the original Q550 phase has no material Jacobian coupling.
                fullJ=jac(z);clock_material_max=max(clock_material_max,float(np.max(abs(fullJ[10,:10]))))
                if clock_material_max>1e-7:raise RuntimeError('Clock couples materially: residual-gradient identity invalid')
                x0=np.r_[z,base,g0['g'],np.zeros(10),0.]
                def rhs(t,x):
                    y=x[:11];base_=x[11:22];m=x[22:32];e=x[32:42]
                    fy=f(t,y);fref=f(t,base_);J=jac(y)
                    g=source_geometry(y,fy,J,q,center,scale,theta0,opts['degree'],opts['phase'],mu)
                    dm,de=continuous_normal_rhs(g['B'],mu,m,e,g['gr'])
                    return np.r_[fy,fref,dm,de,np.exp(-mu*t)*g['r']]
                sol=solve_ivp(rhs,(0.,T),x0,method='DOP853',rtol=args.rtol,atol=args.atol,max_step=args.max_step,dense_output=True)
                if not sol.success:raise RuntimeError('Original residual-gradient Q550 solve failed: '+sol.message)
                integrations+=1
                res=np.asarray(sol.sol(tt)).T
                diffs=[];angles=[];max_err=0.;max_level_err=0.
                def eval_exact_exit(t):
                    xx=sol.sol(float(t))
                    return float(np.linalg.norm((xx[:10]-xx[11:21])/scale))
                # Precompute first exit: bracket from v18.36, evaluate against new continuous solution.
                from flow_normal_math_v18_36 import first_exit
                distances=np.linalg.norm((res[:,:10]-res[:,11:21])/scale,axis=1)
                status,exit_t=first_exit(tt,distances,radius,eval_exact_exit)
                if status!='CROSSED_AFTER_START' or exit_t is None:raise RuntimeError('Unexpected local-support status: '+status)
                if abs(exit_t-float(prior[refkey]['first_exit_delta_xi']))>2e-8:
                    raise RuntimeError('The original v18.36 support exit was not reproduced')
                fd_times=[0.,min(.5*exit_t,T),min(exit_t,T),T]
                for t in sorted(set(fd_times)):
                    x=np.asarray(sol.sol(t),float);y=x[:11]
                    g=geom(y)
                    h1=numerical_residual_gradient(y,f,q,center,scale,theta0,opts['degree'],opts['phase'],mu,1e-4)
                    h2=numerical_residual_gradient(y,f,q,center,scale,theta0,opts['degree'],opts['phase'],mu,5e-5)
                    ae=float(np.linalg.norm(h1-g['gr']));be=float(np.linalg.norm(h2-g['gr']))
                    norm=max(float(np.linalg.norm(g['gr'])),1e-16)
                    finite_difference.append(dict(id=name,anchor_xi=a,branch=branch,delta_xi=float(t),gradient_norm=float(np.linalg.norm(g['gr'])),
                        rhs_only_step1_relative=ae/norm,rhs_only_step2_relative=be/norm,rhs_only_stencil_relative=float(np.linalg.norm(h1-h2))/norm,
                        jacobian_clock_material_max=float(np.max(abs(jac(y)[10,:10])))))
                for t,xx,d in zip(tt,res,distances):
                    y=xx[:11];m=xx[22:32];e=xx[32:42]
                    g=geom(y)
                    g_hom=np.exp(mu*t)*m
                    g_rebuilt=g_hom+e
                    g_error=float(np.linalg.norm(g['g']-g_rebuilt))
                    max_err=max(max_err,g_error)
                    angle=projective_angle(g_hom,g['g'])
                    angles.append(angle)
                    C_expected=np.exp(mu*t)*(g0['C']+xx[42])
                    level_err=abs(C_expected-g['C']);max_level_err=max(max_level_err,level_err)
                    defect=g['C']-np.exp(mu*t)*g0['C']
                    # Signed offset linearized around candidate normal, not a physical normal distance.
                    level_offset=defect/np.linalg.norm(g['g'])
                    row=dict(id=name,anchor_xi=a,branch=branch,delta_xi=float(t),inside_joint_support=bool(d<=radius),exit_delta_xi=float(exit_t),
                        distance_to_comoving=float(d),training_radius=radius,level_C=g['C'],level_defect=float(defect),source_r=g['r'],
                        moving_level_normal_offset=float(level_offset),residual_gradient_norm=float(np.linalg.norm(g['gr'])),
                        residual_gradient_parallel=g['parallel'],residual_gradient_transverse_norm=float(np.linalg.norm(g['perp'])),
                        residual_gradient_local_tilt_rate_rad_per_xi=g['tilt_rate'],
                        normal_angle_stable_deg=angle,normal_angle_diag_stretch_deg=projective_angle(g_hom*gauges['diag_stretch'],g['g']*gauges['diag_stretch']),
                        compensated_gradient_absolute_error=g_error,gradient_direct_norm=float(np.linalg.norm(g['g'])),
                        transported_homogeneous_norm=float(np.linalg.norm(g_hom)),gradient_forcing_integral_norm=float(np.linalg.norm(e)),
                        level_budget_error=float(level_err))
                    rows.append(row)
                inside=[r for r in rows if key(r)==refkey and r['inside_joint_support']]
                if not inside:raise RuntimeError('No in-domain samples')
                summaries.append(dict(id=name,anchor_xi=a,branch=branch,first_exit_delta_xi=float(exit_t),sample_count=len(tt),inside_count=len(inside),
                    initial_source=float(g0['r']),initial_gradient_parallel=float(g0['parallel']),initial_gradient_perp_norm=float(np.linalg.norm(g0['perp'])),
                    initial_tilt_rate_rad_per_xi=float(g0['tilt_rate']),
                    max_abs_level_defect_in_support=max(abs(x['level_defect']) for x in inside),
                    max_absolute_moving_level_normal_offset_in_support=max(abs(x['moving_level_normal_offset']) for x in inside),
                    max_stable_normal_angle_in_support_deg=max(x['normal_angle_stable_deg'] for x in inside),
                    end_stable_normal_angle_deg=angles[-1],max_vector_gradient_identity_error=max_err,max_level_budget_error=max_level_err,
                    final_level_defect=float(rows[-1]['level_defect']),final_level_offset=float(rows[-1]['moving_level_normal_offset'])))
                print('V18_37',name,'xi',a,'branch',branch,'exit',f'{exit_t:.4g}','tilt_deg',f'{angles[-1]:.5g}',
                      'grad_budget',f'{max_err:.3g}','level_budget',f'{max_level_err:.3g}',flush=True)
    out=R/'results'
    csv_write(out/'original_v18_37_residual_gradient_profiles.csv',rows)
    csv_write(out/'original_v18_37_case_summary.csv',summaries)
    csv_write(out/'original_v18_37_rhs_gradient_crosscheck.csv',finite_difference)
    report=dict(version='v18.37',status='ORIGINAL_Q550_RESIDUAL_GRADIENT_NORMAL_TILT_MOVING_LEVEL_COMPLETED',
                original_ode_sha256_verified=True,parent_sha256=provenance['parent_sha256'],parent_verified_members=provenance['verified_members'],
                original_integrations=integrations,original_cases=len(summaries),time_samples=len(rows),
                max_seed_replay_abs_error=max_seed_error,max_material_phase_jacobian=clock_material_max,
                max_absolute_vector_gradient_budget_error=max(s['max_vector_gradient_identity_error'] for s in summaries),
                max_absolute_scalar_level_budget_error=max(s['max_level_budget_error'] for s in summaries),
                rhs_only_gradient_crosschecks=len(finite_difference),
                all_seeds_jointly_admissible=True,no_physical_material_constraint_certified=True,no_material_metric_certified=True,
                method=dict(t_window=T,rtol=args.rtol,atol=args.atol,max_step=args.max_step,
                    residual_gradient='analytic gradient of Lie_C-mu*C using original model Jacobian, quadratic candidate Hessian and phase derivative',
                    independent_check='five-point scalar residual differentiating original RHS, 2 step sizes',
                    cotangent_source_identity='g(t)=exp(mu*t)*m(t)+e(t); m_dot=-B.T*m; e_dot=(-B.T+mu*I)e+grad(r)',
                    scalar_level_identity='C(t)=exp(mu*t)*(C(0)+int(exp(-mu*s)*r(s)ds))',
                    angular_diagnostic='projective atan2 in explicitly normalized material coordinates'))
    (out/'original_v18_37_proof.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('ORIGINAL_V18_37_COMPLETED',integrations,len(summaries),len(rows),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--base-root',type=Path,default=original.DEFAULT_BASE)
    ap.add_argument('--rtol',type=float,default=3e-12);ap.add_argument('--atol',type=float,default=3e-13)
    ap.add_argument('--max-step',type=float,default=2.5e-7)
    run(ap.parse_args())
