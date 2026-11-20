#!/usr/bin/env python3
"""Original Q550: symmetric tangent perturbations on frozen diagnostic surfaces.

No change to kappa0 is made. Every seed starts within the declared joint
material/phase support. Frozen surfaces are tested, NOT certified as physical
invariant manifolds. Fits use inner seeds, outer seeds are held out.
"""
from pathlib import Path
import argparse,csv,json,sys
import numpy as np
from scipy.integrate import solve_ivp
R=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(R/'scripts'),str(R/'reference')]
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
from coupled_scaling_math_v18_38 import tangent_to_normal,signed_initial_sources,signed_tilt,response_fit,normalized_pairs
from seed_genealogy_v18_38 import index_archived_attempts,verify_regenerated_attempts
from flow_normal_math_v18_36 import first_exit
from support_censor_v18_38 import checked_support_horizon, common_inside_time, summarize_exits
import q550_state_impulse_response_v13 as pinned
T=1.e-5
ALPHAS=(-1.,-1./3.,0.,1./3.,1.)

def write_csv(p,rows):
    if not rows:raise RuntimeError('Cannot write empty CSV')
    with p.open('w',newline='') as h:
        w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def key(row):return (row['id'],round(float(row['anchor_xi']),8),int(row['branch']))

def run(args):
    p37,cases37,profiles37,auth=read_parent_v38()
    _,cases36,_,attempts36,auth36=read_parent_v37()
    read_parent_v36();read_parent_v35()
    checkpoints,proof33,model33,returns,overlap,historical=read_parent()
    original=select_original_checkpoint(checkpoints)
    f,jac=model(args.base_root,original)
    bases=anchored_flow(f,np.asarray(original['source_state_11D'],float),ANCHORS)
    center=np.asarray(model33['center'],float)
    scale=np.asarray(model33['scale'],float)
    theta0=float(model33['theta0'])
    train,trainf,test,testf,_,_=evaluate_samples(f,bases,scale)
    candidates={x['id']:x for x in model33['candidates']}
    cases37map={key(x):x for x in cases37}
    cases36map={key(x):x for x in cases36}
    attempts36map=index_archived_attempts(attempts36)
    samples37map={key(x):x for x in profiles37 if float(x['delta_xi'])==0.}
    rng=np.random.default_rng(1836)
    rows=[];summaries=[];fits=[];num_solves=1;max_seed_error=0.;max_phase_perturb=0.
    total_observed_exits=0;total_right_censored=0;families_all_censored=0
    # Advance the exact deterministic 1836 seed stream at all 3 anchors.
    for name in NAMES:
        item=candidates[name];q=np.asarray(item['q'],float);opts=item['opts'];mu=float(opts['lam'])
        if mu!=-1. or float(opts['mod'])!=0.:raise RuntimeError('Cofactor changed')
        def geom(y):return source_geometry(y,f(0.,y),jac(y),q,center,scale,theta0,opts['degree'],opts['phase'],mu)
        for i in ANCHOR_INDICES:
            anchor=bases[i];a=float(ANCHORS[i]);radius=float(calibration_radius(train[i*56:(i+1)*56],anchor,scale)['radius_max'])
            for branch in (0,1):
                ref=(name,round(a,8),branch)
                z0,attempts=projected_independent_seed(anchor,q,center,scale,theta0,opts,radius,rng,branch)
                if ref not in cases37map:raise RuntimeError('Authentic v18.37 case missing')
                verify_regenerated_attempts(ref,attempts,cases36map,attempts36map)
                g0=geom(z0);old=samples37map[ref]
                mismatch=max(abs(g0['C']-float(old['level_C'])),abs(g0['r']-float(old['source_r'])))
                max_seed_error=max(max_seed_error,mismatch)
                if mismatch>5e-7:raise RuntimeError('Authentic v18.37 seed replay failed')
                # Tangent is fixed BEFORE any future computation.
                probe=np.roll(np.arange(1.,11.),branch+2)
                tangent=tangent_to_normal(g0['g'],probe)
                dist0=float(np.linalg.norm((z0[:10]-anchor[:10])/scale))
                margin=.8*radius-dist0
                if margin<=0:raise RuntimeError('Baseline is not in inner support')
                perturbation=min(.04*radius,.22*margin)
                seedlist=[]
                for alpha in ALPHAS:
                    y=z0.copy();y[:10]+=alpha*perturbation*tangent*scale
                    if alpha:
                        y,_,ok=project_to_level(y,q,center,scale,theta0,opts['degree'],opts['phase'],max_scaled_step=.3)
                        if not ok:raise RuntimeError('On-surface projection failed')
                    d=float(np.linalg.norm((y[:10]-anchor[:10])/scale))
                    if d>.8*radius or abs(y[10]-anchor[10])>.2:raise RuntimeError('Perturbed seed outside local support')
                    C=float(features(y,center,scale,theta0,opts['degree'],opts['phase'])@q)
                    if abs(C)>1e-8:raise RuntimeError('Perturbed seed not on candidate surface')
                    max_phase_perturb=max(max_phase_perturb,abs(y[10]-z0[10]))
                    seedlist.append((alpha,y,d,C))
                # Every case uses a common within-support evaluation time, determined
                # only by the initial seed and the pre-declared support thresholds.
                trialsol=[];horizons=[];exit_records=[]
                for alpha,y,d,C in seedlist:
                    gg=geom(y)
                    m0=gg['g'];refstate=anchor.copy();refstate[10]=y[10]
                    x0=np.r_[y,refstate,m0,np.zeros(10),0.]
                    def rhs(t,x):
                        ys=x[:11];yr=x[11:22];m=x[22:32];e=x[32:42]
                        fy=f(t,ys);fr=f(t,yr)
                        G=source_geometry(ys,fy,jac(ys),q,center,scale,theta0,opts['degree'],opts['phase'],mu)
                        return np.r_[fy,fr,-G['B'].T@m,(-G['B'].T+mu*np.eye(10))@e+G['gr'],np.exp(-mu*t)*G['r']]
                    sol=solve_ivp(rhs,(0.,T),x0,method='DOP853',rtol=args.rtol,atol=args.atol,max_step=args.max_step,dense_output=True)
                    if not sol.success:raise RuntimeError('Original v18.38 Q550 integration failed: '+sol.message)
                    num_solves+=1
                    tt=np.unique(np.r_[0.,np.geomspace(1.e-11,T,48),np.linspace(0.,T,64)])
                    xx=np.asarray(sol.sol(tt)).T
                    dd=np.linalg.norm((xx[:,:10]-xx[:,11:21])/scale,axis=1)
                    def evald(t):
                        x=sol.sol(float(t));return float(np.linalg.norm((x[:10]-x[11:21])/scale))
                    status,exitt=first_exit(tt,dd,radius,evald)
                    horizon=checked_support_horizon(status,exitt,T)
                    # A right-censored observation is not an invented exit at T.
                    horizons.append(horizon);exit_records.append((status,exitt))
                    trialsol.append((alpha,y,C,sol,status,exitt,horizon))
                t_common=common_inside_time(horizons,T)
                exit_stats=summarize_exits(exit_records,horizons)
                total_observed_exits+=exit_stats['observed_exit_count']
                total_right_censored+=exit_stats['right_censored_count']
                families_all_censored+=int(exit_stats['observed_exit_count']==0)
                case_id=f'{name}|xi={a:g}|branch={branch}'
                # Verify the shared observation interval at a finer grid as an
                # additional numerical guard (not an interval-arithmetic proof).
                check_grid=np.linspace(0.,t_common,81)
                for alpha,y,C,sol,status,exitt,horizon in trialsol:
                    xt=np.asarray(sol.sol(check_grid)).T
                    dv=np.linalg.norm((xt[:,:10]-xt[:,11:21])/scale,axis=1)
                    if not np.all(np.isfinite(dv)) or np.max(dv)>radius+1.e-10:
                        raise RuntimeError('Shared measurement interval left diagnostic support')
                for alpha,y,C,sol,status,exitt,horizon in trialsol:
                    initial_geom=geom(y)
                    initial_r,initial_signed=signed_initial_sources(initial_geom,tangent)
                    for moment,t in [('initial',0.),('common_inside',t_common)]:
                        x=np.asarray(sol.sol(t),float)
                        g=geom(x[:11]);m=x[22:32];e=x[32:42];g_hom=np.exp(mu*t)*m
                        level_rate,signed_rate=signed_initial_sources(g,tangent)
                        defect=float(g['C']-np.exp(mu*t)*C)
                        reconstructed=float(np.exp(mu*t)*(C+x[42]))
                        rows.append(dict(case_id=case_id,id=name,anchor_xi=a,branch=branch,alpha=alpha,scaled_tangent_offset=alpha*perturbation,
                            moment=moment,delta_xi=t,exit_delta_xi=exitt if exitt is not None else '',
                            support_exit_status=status,support_observed_horizon=horizon,
                            inside_support=t<horizon,
                            initial_distance_to_comoving=float(np.linalg.norm((y[:10]-anchor[:10])/scale)),training_radius=radius,
                            candidate_C=float(g['C']),level_defect=defect,level_rate=level_rate,
                            signed_tilt_rate=signed_rate,tilt_magnitude_rate=float(g['tilt_rate']),
                            signed_normal_tilt_rad=signed_tilt(g_hom,g['g'],tangent),
                            level_normal_offset=defect/np.linalg.norm(g['g']),
                            vector_gradient_reconstruction_error=float(np.linalg.norm(g['g']-g_hom-e)),
                            level_balance_error=abs(reconstructed-g['C'])))
                for moment in ('initial','common_inside'):
                    block=[r for r in rows if r['case_id']==case_id and r['moment']==moment]
                    for field in ('level_rate','signed_tilt_rate','level_normal_offset','signed_normal_tilt_rad'):
                        fit=response_fit(block,field)
                        fits.append(dict(case_id=case_id,id=name,anchor_xi=a,branch=branch,moment=moment,observable=field,
                            **fit))
                    coupled=normalized_pairs(block)
                    summaries.append(dict(case_id=case_id,id=name,anchor_xi=a,branch=branch,moment=moment,
                        common_within_support_time=t_common,
                        min_exit_delta_xi=exit_stats['min_observed_exit_delta_xi'],
                        max_exit_delta_xi=exit_stats['max_observed_exit_delta_xi'],
                        observed_exit_count=exit_stats['observed_exit_count'],
                        right_censored_count=exit_stats['right_censored_count'],
                        min_support_horizon_delta_xi=min(horizons),
                        perturbation_scale=perturbation,initial_base_distance=dist0,initial_margin_to_inner_support=margin,
                        **coupled))
                print('V18_38',name,a,branch,
                    'observed_exits',exit_stats['observed_exit_count'],
                    'right_censored',exit_stats['right_censored_count'],
                    'horizon_min',f'{min(horizons):.4g}',
                    'common_t',f'{t_common:.4g}',flush=True)
    out=R/'results';out.mkdir(exist_ok=True)
    write_csv(out/'original_v18_38_coupled_profiles.csv',rows)
    write_csv(out/'original_v18_38_coupled_fits.csv',fits)
    write_csv(out/'original_v18_38_coupled_summary.csv',summaries)
    proof=dict(version='v18.38',status='ORIGINAL_Q550_COUPLED_LEVEL_NORMAL_SCALING_COMPLETED',
        source_kappa0_unchanged=True,original_ode_sha256_verified=True,
        parent_sha256=auth['parent_sha256'],parent_verified_members=auth['verified_members'],
        original_integrations=num_solves,original_cases=len(summaries)//2,
        samples=len(rows),fit_rows=len(fits),max_original_seed_replay_error=max_seed_error,max_seed_phase_change=max_phase_perturb,
        observed_exit_count=total_observed_exits,right_censored_count=total_right_censored,
        families_all_right_censored=families_all_censored,
        max_vector_budget_error=max(r['vector_gradient_reconstruction_error'] for r in rows),
        max_level_budget_error=max(r['level_balance_error'] for r in rows),
        no_material_metric_certified=True,no_physical_constraint_certified=True,
        method=dict(amplitudes=list(ALPHAS),rtol=args.rtol,atol=args.atol,max_step=args.max_step,time_window=T,
            symmetric_perturbations='tangent to frozen candidate level, then projected back to C=0',
            fit='inner +/-1/3 seeds, external +/-1 are independently held out',
            initial_comparison='signed co-moving normalized level and signed projected tilt source',
            common_time='half of earliest support horizon: actual first exit when observed, otherwise right-censoring at end of measurement window',
            exit_censoring='NO_EXIT_OBSERVED is a right-censored observation; never report window end as an exit',
            caution='Only within-case, normalized-coordinate diagnostic comparisons. kappa0 unchanged.'))
    (out/'original_v18_38_proof.json').write_text(json.dumps(proof,indent=2,allow_nan=False)+'\n')
    print('ORIGINAL_V18_38_COMPLETED',num_solves,len(rows),len(fits),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--base-root',type=Path,default=pinned.DEFAULT_BASE)
    ap.add_argument('--rtol',type=float,default=3e-12);ap.add_argument('--atol',type=float,default=3e-13)
    ap.add_argument('--max-step',type=float,default=2.5e-7)
    run(ap.parse_args())
