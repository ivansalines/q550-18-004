#!/usr/bin/env python3
"""v18.44 original Q550: dual numerical directional Hessian and chart-transport jet.

Replays SHA-pinned parent v18.43 perturbation seeds and 49 Q550 integrations.
Independent estimates of (L_F B)^T g: directionally differentiate ORIGINAL
pinned analytic Jacobian, and form independent mixed differences of ORIGINAL
pinned RHS F (no analytic ODE Hessian is claimed). A deterministic Fréchet
chart derivative isolates source variation, own-normal mapping, and reference
frame transport. Preserve lifted clock, material support and frozen kappa0.
"""
import sys,csv,json,argparse
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
R=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(R/'scripts'),str(R/'reference')]
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
from manifold_math_v18_32 import project_to_level,features
from near_surface_math_v18_35 import calibration_radius
from residual_gradient_math_v18_37 import source_geometry,unit
from coupled_scaling_math_v18_38 import tangent_to_normal
from seed_genealogy_v18_38 import index_archived_attempts,verify_regenerated_attempts
from tangent_source_math_v18_39 import normal_frame
from analytic_source_jet_math_v18_43 import material_source_jet,moving_coords_and_jet
from dual_hessian_jet_math_v18_44 import (dot_B_transpose_g_from_jacobian,
        mixed_rhs_hessian_contraction,source_jet_from_contraction,exact_chart_jet)
import q550_state_impulse_response_v13 as pinned
H=round(1./3.,10)
FRACS=(.125,.5,1.)
RHS_STEPS=(.006,.018,.036)
JAC_STEPS=(.0005,.002,.008)

def out_csv(name,rows):
    if not rows:raise RuntimeError('No original rows: '+name)
    with (R/'results'/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def case_key(row):return (row['id'],round(float(row['anchor_xi']),8),int(row['branch']))
def relative(a,b):return float(np.linalg.norm(a-b)/max(np.linalg.norm(a),np.linalg.norm(b),1e-30))
def scalarvec(v,p):return {p+str(i):float(v[i]) for i in range(len(v))}

def run(args):
    p43,first43,second43,steps43,fam43,auth44=read_parent_v44()
    p42,old42,profiles42,windows42,stencils42,ladder42,auth43=read_parent_v43()
    read_parent_v42();read_parent_v41();read_parent_v40();read_parent_v39();read_parent_v38()
    _,cases36,_,attempts36,_=read_parent_v37()
    read_parent_v36();read_parent_v35()
    chk,_,model33,_,_,_=read_parent()
    original=select_original_checkpoint(chk)
    f,jac=model(args.base_root,original) # PINNED authentic Q550 RHS and Jacobian
    anchors=anchored_flow(f,np.asarray(original['source_state_11D'],float),ANCHORS)
    center=np.asarray(model33['center'],float);scale=np.asarray(model33['scale'],float)
    theta0=float(model33['theta0']);train,*_=evaluate_samples(f,anchors,scale)
    candidates={a['id']:a for a in model33['candidates']}
    old42map={case_key(a):a for a in old42}
    first43map={(r['case_id'],round(float(r['fraction_of_common_time']),10)):r for r in first43}
    cases36map={case_key(r):r for r in cases36};attempts36map=index_archived_attempts(attempts36)
    rng=np.random.default_rng(1836)
    hrows=[];srows=[];families=[];solvers=[]
    max_parent_vector=0.;max_support_fraction=0.;orig_sols=1
    for name in NAMES:
        candidate=candidates[name];q=np.asarray(candidate['q'],float);opts=candidate['opts'];mu=float(opts['lam'])
        if mu!=-1. or float(opts['mod'])!=0.:raise RuntimeError('Unexpected model cofactor')
        def geom(y):return source_geometry(y,f(0.,y),jac(y),q,center,scale,theta0,opts['degree'],opts['phase'],mu)
        for i in ANCHOR_INDICES:
            anchor=anchors[i];a=float(ANCHORS[i])
            radius=float(calibration_radius(train[i*56:(i+1)*56],anchor,scale)['radius_max'])
            for branch in (0,1):
                key=(name,round(a,8),branch)
                case_id=f'{name}|xi={a:g}|branch={branch}'
                old=old42map.get(key)
                if old is None:raise RuntimeError('Missing v18.42 family')
                T=float(old['common_within_support_time'])
                if not(0<T<=1.e-5):raise RuntimeError('Invalid historical support horizon')
                y0,attempts=projected_independent_seed(anchor,q,center,scale,theta0,opts,radius,rng,branch)
                verify_regenerated_attempts(key,attempts,cases36map,attempts36map)
                g0=geom(y0);tangent=tangent_to_normal(g0['g'],np.roll(np.arange(1.,11.),branch+2))
                margin=.8*radius-float(np.linalg.norm((y0[:10]-anchor[:10])/scale))
                if margin<=0:raise RuntimeError('Uncalibrated seed')
                perturbation=min(.04*radius,.22*margin)
                ys={}
                for alpha in (-H,0.,H):
                    y=y0.copy();y[:10]+=alpha*perturbation*tangent*scale
                    if alpha:
                        y,_,ok=project_to_level(y,q,center,scale,theta0,opts['degree'],opts['phase'],max_scaled_step=.3)
                        if not ok:raise RuntimeError('Projection failed')
                    if abs(float(features(y,center,scale,theta0,opts['degree'],opts['phase'])@q))>1.e-8:
                        raise RuntimeError('Seed does not lie on diagnostic C=0')
                    if np.linalg.norm((y[:10]-anchor[:10])/scale)>.8*radius or abs(y[10]-anchor[10])>.2:
                        raise RuntimeError('Seed outside calibrated support')
                    sol=solve_ivp(f,(0.,T),y,method='DOP853',rtol=args.rtol,atol=args.atol,
                        max_step=args.max_step,dense_output=True)
                    if not sol.success:raise RuntimeError('Original Q550 integration failure '+sol.message)
                    ys[alpha]=sol;orig_sols+=1
                    dt=np.diff(sol.t)
                    solvers.append(dict(case_id=case_id,kind='perturbed_seed',alpha=alpha,accepted_steps=len(dt),
                        nfev=sol.nfev,min_dt=float(dt.min()),max_dt=float(dt.max()),rtol=args.rtol,atol=args.atol,
                        max_step=args.max_step))
                refseed=anchor.copy();refseed[10]=y0[10] # common ORIGINAL clock phase
                refsol=solve_ivp(f,(0.,T),refseed,method='DOP853',rtol=args.rtol,atol=args.atol,
                    max_step=args.max_step,dense_output=True)
                if not refsol.success:raise RuntimeError('Original phase-matched reference failed')
                orig_sols+=1;dt=np.diff(refsol.t)
                solvers.append(dict(case_id=case_id,kind='anchor_reference',alpha='',accepted_steps=len(dt),
                    nfev=refsol.nfev,min_dt=float(dt.min()),max_dt=float(dt.max()),rtol=args.rtol,atol=args.atol,
                    max_step=args.max_step))
                n0=unit(g0['g']);frame0=normal_frame(n0)
                for s in FRACS:
                    t=s*T
                    states={alpha:np.asarray(ys[alpha].sol(t),float) for alpha in (-H,0.,H)}
                    for alpha,y in states.items():
                        d=float(np.linalg.norm((y[:10]-refsol.sol(t)[:10])/scale))
                        max_support_fraction=max(max_support_fraction,d/radius)
                        if d>radius*(1+1e-6):raise RuntimeError('Original state outside authenticated local support')
                    jets={};hess={};oldjets={};jacjets={};discrepancies={}
                    for alpha,y in states.items():
                        fy=np.asarray(f(0.,y),float);J=np.asarray(jac(y),float)
                        if np.linalg.norm(J[10,:10]*scale)>1e-7:raise RuntimeError('Phase velocity depends materially: invalid source identity')
                        geo=source_geometry(y,fy,J,q,center,scale,theta0,opts['degree'],opts['phase'],mu)
                        fixed={}
                        for target in JAC_STEPS:
                            fixed[target]=dot_B_transpose_g_from_jacobian(y,fy,jac,scale,geo['g'],target)
                        js=source_jet_from_contraction(y,fy,J,q,center,scale,theta0,opts['degree'],opts['phase'],mu,fixed[.002])
                        oldjet=material_source_jet(y,f,jac,q,center,scale,theta0,opts['degree'],opts['phase'],mu,scaled_target=.0005)
                        js2=source_jet_from_contraction(y,fy,J,q,center,scale,theta0,opts['degree'],opts['phase'],mu,fixed[.0005])
                        if np.linalg.norm(js2['pdot']-oldjet['pdot'])>1e-7*max(1.,np.linalg.norm(oldjet['pdot'])):
                            raise RuntimeError('New Jacobian-chain implementation differs from original v18.43')
                        rhs={}
                        for h in RHS_STEPS:
                            rhs[h]=mixed_rhs_hessian_contraction(y,fy,f,scale,geo['g'],h)
                            hrows.append(dict(case_id=case_id,id=name,anchor_xi=a,branch=branch,alpha=alpha,
                                fraction_of_common_time=s,delta_xi=t,phase_lifted=float(y[10]),
                                phase_change_lifted=float(y[10]-states[0.][10]),
                                rhs_normalized_h=h,jac_flow_scaled_target=.002,
                                jac_contraction_norm=float(np.linalg.norm(fixed[.002])),
                                rhs_contraction_norm=float(np.linalg.norm(rhs[h])),
                                relative_rhs_jac_disagreement=relative(rhs[h],fixed[.002]),
                                abs_rhs_jac_disagreement=float(np.linalg.norm(rhs[h]-fixed[.002])),
                                jac_step_sensitivity=relative(fixed[.0005],fixed[.008]),
                                **scalarvec(rhs[h],'rhs_hessian_'),**scalarvec(fixed[.002],'jac_hessian_')))
                        jsrhs=source_jet_from_contraction(y,fy,J,q,center,scale,theta0,opts['degree'],opts['phase'],mu,rhs[.018])
                        jacjets[alpha]=js;hess[alpha]=jsrhs;oldjets[alpha]=oldjet
                        discrepancies[alpha]=relative(js['pdot'],jsrhs['pdot'])
                    # Coordinate transport uses analytic Fréchet derivative; compare
                    # original finite gauge rule on precisely the same source jets.
                    chartjac={alpha:exact_chart_jet(jacjets[alpha],jacjets[0.],n0,frame0) for alpha in (-H,H)}
                    chartrhs={alpha:exact_chart_jet(hess[alpha],hess[0.],n0,frame0) for alpha in (-H,H)}
                    finitechart={alpha:moving_coords_and_jet(oldjets[alpha],oldjets[0.],n0,frame0) for alpha in (-H,H)}
                    v=1.5*(chartjac[H]['v']-chartjac[-H]['v'])
                    vd=1.5*T*(chartjac[H]['jet']-chartjac[-H]['jet'])
                    v_rhs=1.5*(chartrhs[H]['v']-chartrhs[-H]['v'])
                    vd_rhs=1.5*T*(chartrhs[H]['jet']-chartrhs[-H]['jet'])
                    vd_old=1.5*T*(finitechart[H][1]-finitechart[-H][1])
                    archived=first43map.get((case_id,round(s,10)))
                    if archived is None:raise RuntimeError('Missing authentic v18.43 source jet row')
                    pv=np.array([float(archived['v'+str(k)]) for k in range(9)])
                    pjet=np.array([float(archived['jet'+str(k)]) for k in range(9)])
                    replay=float(np.linalg.norm(v-pv));max_parent_vector=max(max_parent_vector,replay)
                    if replay>5e-9+2e-5*np.linalg.norm(pv):raise RuntimeError('Archived original Q550 source vector not reproduced')
                    gauge_ref=1.5*T*(chartjac[H]['frame_transport_differential']-chartjac[-H]['frame_transport_differential'])
                    gauge_own=1.5*T*(chartjac[H]['own_alignment_differential']-chartjac[-H]['own_alignment_differential'])
                    source_part=1.5*T*(chartjac[H]['source_differential']-chartjac[-H]['source_differential'])
                    srows.append(dict(case_id=case_id,id=name,anchor_xi=a,branch=branch,
                        fraction_of_common_time=s,delta_xi=t,common_support_time=T,
                        initial_phase_lifted=float(y0[10]),phase_lifted=float(states[0.][10]),
                        phase_change_lifted=float(states[0.][10]-y0[10]),
                        jac_chart_jet_norm=float(np.linalg.norm(vd)),rhs_chart_jet_norm=float(np.linalg.norm(vd_rhs)),
                        old_v18_43_chart_jet_norm=float(np.linalg.norm(pjet)),
                        analytic_vs_finite_chart_disagreement=relative(vd,vd_old),
                        original_v18_43_jet_disagreement=relative(vd,pjet),
                        analytic_jac_vs_rhs_jet_disagreement=relative(vd,vd_rhs),
                        source_norm=float(np.linalg.norm(v)),
                        source_time_jet_norm=float(np.linalg.norm(source_part)),
                        reference_frame_transport_norm=float(np.linalg.norm(gauge_ref)),
                        own_normal_alignment_norm=float(np.linalg.norm(gauge_own)),
                        chart_total_vs_components=float(np.linalg.norm(vd-(source_part+gauge_ref+gauge_own))),
                        mean_material_jet_disagreement=float(np.mean(list(discrepancies.values()))),
                        original_v18_43_source_replay_abs=replay,
                        **scalarvec(v,'v'),**scalarvec(vd,'analytic_jet_'),
                        **scalarvec(vd_rhs,'rhs_jet_'),**scalarvec(pjet,'prior_jet_')))
                this=[r for r in srows if r['case_id']==case_id]
                hthis=[r for r in hrows if r['case_id']==case_id]
                families.append(dict(case_id=case_id,id=name,anchor_xi=a,branch=branch,
                    common_support_time=T,
                    median_hessian_disagreement=float(np.median([r['relative_rhs_jac_disagreement'] for r in hthis])),
                    max_hessian_disagreement=max(r['relative_rhs_jac_disagreement'] for r in hthis),
                    median_analytic_vs_finite_chart_disagreement=float(np.median([r['analytic_vs_finite_chart_disagreement'] for r in this])),
                    median_jet_disagreement_from_v18_43=float(np.median([r['original_v18_43_jet_disagreement'] for r in this])),
                    median_independent_rhs_jet_disagreement=float(np.median([r['analytic_jac_vs_rhs_jet_disagreement'] for r in this])),
                    max_parent_source_replay_abs=max(r['original_v18_43_source_replay_abs'] for r in this)))
                print('V18_44',case_id,'hessian median',f'{families[-1]["median_hessian_disagreement"]:.3e}',
                      'jet',f'{families[-1]["median_independent_rhs_jet_disagreement"]:.3e}',flush=True)
    if tuple(map(len,(families,srows,hrows,solvers)))!=(12,36,324,48) or orig_sols!=49:
        raise RuntimeError('Incomplete original Q550 v18.44, no sendback permitted')
    (R/'results').mkdir(exist_ok=True)
    out_csv('original_v18_44_hessian_dual_contractions.csv',hrows)
    out_csv('original_v18_44_transported_source_jets.csv',srows)
    out_csv('original_v18_44_family_summary.csv',families)
    out_csv('original_v18_44_solver_diagnostics.csv',solvers)
    proof=dict(version='v18.44',status='ORIGINAL_Q550_DIRECTIONAL_HESSIAN_TRANSPORTED_SOURCE_JET_COMPLETED',
        original_integrations=orig_sols,original_cases=len(families),hessian_comparison_rows=len(hrows),
        transported_jet_rows=len(srows),solver_rows=len(solvers),
        parent_v18_43_sha256=auth44['parent_sha256'],parent_v18_43_verified_members=auth44['verified_members'],
        original_ode_sha256_verified=True,source_kappa0_unchanged=True,
        max_parent_source_replay_abs=max_parent_vector,max_within_parent_support_radius_fraction=max_support_fraction,
        no_symbolic_hessian_claimed=True,no_certified_second_jet=True,no_certified_curvature=True,
        no_material_metric_certified=True,no_physical_constraint_certified=True,
        method=dict(hessian='Fourth-order independent mixed ORIGINAL RHS field differences vs directional pinned analytic Jacobian derivative',
            hessian_steps=list(RHS_STEPS),jacobian_direction_steps=list(JAC_STEPS),
            source_jet='Exact polynomial/phase chain rule with substituted numerical directional Hessian contraction',
            moving_chart='Analytic Frechet differentiation of minimal rotations and tangent projector; original v18.43 finite chart jet cross-checked',
            kappa0='Frozen inherited 55/6*(10/550); alternative geometric hypothesis NOT introduced',
            material_metric='Declared normalized Euclidean diagnostic convention only',
            phase='Unwrapped original clock phase recorded; phase return is not full material return',
            warning='Agreement of two numerical Hessian routes is not a symbolic or physical Hessian certificate'),
        preflight_only=False)
    (R/'results/original_v18_44_proof.json').write_text(json.dumps(proof,indent=2,allow_nan=False)+'\n')
    print('ORIGINAL_V18_44_COMPLETED',orig_sols,len(hrows),len(srows),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--base-root',type=Path,default=pinned.DEFAULT_BASE)
    ap.add_argument('--rtol',type=float,default=3e-12)
    ap.add_argument('--atol',type=float,default=3e-13)
    ap.add_argument('--max-step',type=float,default=2.5e-7)
    run(ap.parse_args())
