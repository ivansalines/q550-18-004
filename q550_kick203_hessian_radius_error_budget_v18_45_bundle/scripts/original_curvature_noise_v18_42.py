#!/usr/bin/env python3
"""Original Q550 v18.42: SECOND-DERIVATIVE noise and window-scaling audit.

No kappa0 change. 60 genuinely new original Q550 integrations with authentic
v18.38 seeds and frozen candidate surfaces. Full material 10D transverse
source, common moving 9D tangent frame, held-out vector fits. The material
Euclidean normalization is a declared gauge, NOT an invariant metric.
"""
from pathlib import Path
import argparse,csv,json,sys
import numpy as np
from scipy.integrate import solve_ivp
R=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(R/'scripts'),str(R/'reference')]
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
from coupled_scaling_math_v18_38 import tangent_to_normal,signed_initial_sources,signed_tilt,response_fit,normalized_pairs
from seed_genealogy_v18_38 import index_archived_attempts,verify_regenerated_attempts
from flow_normal_math_v18_36 import first_exit
from tangent_source_math_v18_39 import normal_frame,minimal_rotation,common_vector,tilt_vector,vector_holdout,source_level_coherence,rotation_angle
from gauge_temporal_math_v18_40 import stable_turn,direct_frame,path_frames,bridge_frame,tangent_twist,align_coordinate,source_in_frame,odd_response,angular_travel
from support_censor_v18_38 import checked_support_horizon, common_inside_time, summarize_exits
from original_noise_window_v18_42_math import case_audit, make_response
from second_derivative_noise_math_v18_42 import window_diagnostics, WINDOWS
import q550_state_impulse_response_v13 as pinned
T=1.e-5
ALPHAS=(-1.,-1./3.,0.,1./3.,1.)

def write_csv(p,rows):
    if not rows:raise RuntimeError('Cannot write empty CSV')
    with p.open('w',newline='') as h:
        w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def key(row):return (row['id'],round(float(row['anchor_xi']),8),int(row['branch']))

def run(args):
    # Authenticate v18.41 FIRST; its nested v18.40 ZIP is staged in .runtime.
    proof41,original41_families,original41_profiles,original41_resolution,auth42=read_parent_v42()
    original41_family={r['case_id']:r for r in original41_families}
    proof40,temporal40,families40,resolution40,auth41=read_parent_v41()
    original40_families={x['case_id']:x for x in families40}
    original40_resolution={(x['case_id'],int(x['uniform_time_subintervals'])):x for x in resolution40}
    proof39,frames39,vectors39,auth40=read_parent_v40()
    proof38,summaries38,profiles38,fits38,auth39=read_parent_v39()
    frames39map={x['case_id']:x for x in frames39}
    vectors39map={(x['case_id'],round(float(x['alpha']),10),x['moment']):x for x in vectors39}
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
    rows=[];summaries=[];fits=[];vectors=[];vectorfits=[];rotation=[];new_raw=[]
    temporal=[];gauges=[];family_temporal=[];resolution=[]
    plane_profiles=[];plane_families=[];plane_resolutions=[]
    noise_profiles=[];noise_windows=[];noise_derivatives=[];noise_families=[];noise_solver=[];extra_solves=0
    max_parent_vector_replay=0.;max_chart_alignment=0.;max_twist_alignment=0.;max_gauge_norm=0.;max_old_rotation_replay=0.
    old_profiles_map={(r['case_id'],round(float(r['alpha']),10),r['moment']):r for r in profiles38}
    old_summary_map={(r['case_id'],r['moment']):r for r in summaries38}
    num_solves=1;max_seed_error=0.;max_phase_perturb=0.;max_parent_discrepancy=0.;max_source_reduction_error=0.
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
                case_raw=[]
                for alpha,y,C,sol,status,exitt,horizon in trialsol:
                    initial_geom=geom(y)
                    initial_r,initial_signed=signed_initial_sources(initial_geom,tangent)
                    for moment,t in [('initial',0.),('common_inside',t_common)]:
                        x=np.asarray(sol.sol(t),float)
                        g=geom(x[:11]);m=x[22:32];e=x[32:42];g_hom=np.exp(mu*t)*m
                        level_rate,signed_rate=signed_initial_sources(g,tangent)
                        defect=float(g['C']-np.exp(mu*t)*C)
                        reconstructed=float(np.exp(mu*t)*(C+x[42]))
                        raw_source=g['perp']/np.linalg.norm(g['g'])
                        raw_tilt=tilt_vector(g_hom,g['g'])
                        case_raw.append(dict(alpha=alpha,moment=moment,normal=g['g'],hom=g_hom,
                            source=raw_source,tilt=raw_tilt,geom=g,observation=t,scalar_rate=signed_rate))
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
                # Replay authentic v18.38 scalar measurements and the support
                # event classifications before interpreting the new 9D vectors.
                for rr in rows[-10:]:
                    old=old_profiles_map.get((rr['case_id'],round(float(rr['alpha']),10),rr['moment']))
                    if old is None:raise RuntimeError('Missing original v18.38 trajectory record')
                    if rr['support_exit_status']!=old['support_exit_status']:
                        raise RuntimeError('Original support censoring classification changed')
                    for fld in ('delta_xi','level_rate','signed_tilt_rate','tilt_magnitude_rate',
                                'signed_normal_tilt_rad','level_normal_offset','vector_gradient_reconstruction_error'):
                        diff=abs(float(rr[fld])-float(old[fld]));ref=abs(float(old[fld]))
                        max_parent_discrepancy=max(max_parent_discrepancy,diff)
                        if diff>3.e-9+1.e-5*ref:raise RuntimeError('Original v18.38 scalar trajectory replay failed: '+fld)
                # Establish a single common 9D diagnostic frame at the alpha=0
                # normal. Transport that frame minimally to the later base normal.
                raw_base0=next(z for z in case_raw if z['alpha']==0. and z['moment']=='initial')
                n0=unit(raw_base0['normal']);frame0=normal_frame(n0)
                raw_base_t=next(z for z in case_raw if z['alpha']==0. and z['moment']=='common_inside')
                nt=unit(raw_base_t['normal'])
                for moment in ('initial','common_inside'):
                    nbase=n0 if moment=='initial' else nt
                    frame=frame0 if moment=='initial' else minimal_rotation(n0,nt)@frame0
                    block=[z for z in case_raw if z['moment']==moment]
                    for z in block:
                        sn=unit(z['normal']);hn=unit(z['hom']);g=z['geom']
                        sv=common_vector(z['source'],sn,nbase,frame)
                        tv=common_vector(z['tilt'],hn,nbase,frame)
                        observer=tangent_to_normal(z['normal'],tangent)
                        sourced_scalar=float(z['source']@observer)
                        reduction_error=abs(sourced_scalar-z['scalar_rate'])
                        max_source_reduction_error=max(max_source_reduction_error,reduction_error)
                        if reduction_error>2.e-11:raise RuntimeError('9D source disagrees with v18.38 observer projection')
                        normal_component=float(np.linalg.norm(sv));orig_component=float(np.linalg.norm(z['source']))
                        if abs(normal_component-orig_component)>2.e-11:raise RuntimeError('Full transverse source is not preserved by common frame')
                        if abs(np.linalg.norm(tv)-np.linalg.norm(z['tilt']))>2.e-11:raise RuntimeError('Angular tilt vector lost energy on frame transfer')
                        vectors.append(dict(case_id=case_id,id=name,anchor_xi=a,branch=branch,alpha=z['alpha'],moment=moment,
                            delta_xi=z['observation'],within_support=True,
                            source_components=[float(v) for v in sv],tilt_components=[float(v) for v in tv],
                            source_norm=orig_component,source_v38_projection=sourced_scalar,
                            observer_fraction=abs(sourced_scalar)/max(orig_component,1e-30),
                            normal_tilt_vector_norm=float(np.linalg.norm(tv)),
                            center_normal_gauge= [float(v) for v in nbase]))
                for moment in ('initial','common_inside'):
                    block=[r for r in rows if r['case_id']==case_id and r['moment']==moment]
                    for field in ('level_rate','signed_tilt_rate','level_normal_offset','signed_normal_tilt_rad'):
                        fit=response_fit(block,field)
                        fits.append(dict(case_id=case_id,id=name,anchor_xi=a,branch=branch,moment=moment,observable=field,
                            **fit))
                    coupled=normalized_pairs(block)
                    vblock=[z for z in vectors if z['case_id']==case_id and z['moment']==moment]
                    vsrc=vector_holdout(vblock,'source_components')
                    vtilt=vector_holdout(vblock,'tilt_components')
                    for observable,fit in [('full9d_source_rate',vsrc),('full9d_integrated_tilt',vtilt)]:
                        vectorfits.append(dict(case_id=case_id,id=name,anchor_xi=a,branch=branch,moment=moment,
                            observable=observable,**fit))
                    coh=source_level_coherence([u['level_rate'] for u in sorted(block,key=lambda u:u['alpha'])],
                        [u['source_components'] for u in sorted(vblock,key=lambda u:u['alpha'])])
                    summaries.append(dict(case_id=case_id,id=name,anchor_xi=a,branch=branch,moment=moment,
                        common_within_support_time=t_common,
                        min_exit_delta_xi=exit_stats['min_observed_exit_delta_xi'],
                        max_exit_delta_xi=exit_stats['max_observed_exit_delta_xi'],
                        observed_exit_count=exit_stats['observed_exit_count'],
                        right_censored_count=exit_stats['right_censored_count'],
                        min_support_horizon_delta_xi=min(horizons),
                        perturbation_scale=perturbation,initial_base_distance=dist0,initial_margin_to_inner_support=margin,
                        **coupled))
                    summaries[-1].update(full_source_level_coherence=coh,full_source_holdout_error=vsrc['holdout_relative_error'],
                        full_source_odd_turn_deg=vsrc['inner_outer_odd_turn_deg'],
                        full_source_integrated_tilt_holdout_error=vtilt['holdout_relative_error'],
                        full_source_median_observer_fraction=float(np.median([z['observer_fraction'] for z in vblock])))
                src0={float(z['alpha']):np.asarray(z['source_components'],float) for z in vectors if z['case_id']==case_id and z['moment']=='initial'}
                srct={float(z['alpha']):np.asarray(z['source_components'],float) for z in vectors if z['case_id']==case_id and z['moment']=='common_inside'}
                odd0=(src0[1./3.]-src0[-1./3.])*1.5
                oddt=(srct[1./3.]-srct[-1./3.])*1.5
                old0={float(z['alpha']):float(z['signed_tilt_rate']) for z in rows if z['case_id']==case_id and z['moment']=='initial'}
                oldt={float(z['alpha']):float(z['signed_tilt_rate']) for z in rows if z['case_id']==case_id and z['moment']=='common_inside'}
                s0=(old0[1./3.]-old0[-1./3.])*1.5
                st=(oldt[1./3.]-oldt[-1./3.])*1.5
                rotation.append(dict(case_id=case_id,id=name,anchor_xi=a,branch=branch,
                    common_within_support_time=t_common,
                    base_normal_turn_deg=rotation_angle(n0,nt),
                    full_odd_source_transport_turn_deg=rotation_angle(odd0,oddt),
                    full_odd_source_cosine=float((odd0@oddt)/max(np.linalg.norm(odd0)*np.linalg.norm(oddt),1.e-30)),
                    old_projected_odd_slope_initial=s0,old_projected_odd_slope_common=st,
                    old_projected_sign_reversal=bool(s0*st<0),
                    old_projected_sign_reversal_with_full_odd_cos_positive=bool(s0*st<0 and odd0@oddt>0),
                    max_tilt_vector_norm_common=max(float(z['normal_tilt_vector_norm']) for z in vectors if z['case_id']==case_id and z['moment']=='common_inside')))
                # Authentic v18.39 parent regression, before new temporal observations.
                prev=frames39map.get(case_id)
                if prev is None:raise RuntimeError('Original v18.39 frame case missing')
                rot39=rotation[-1]
                for fld in ('common_within_support_time','full_odd_source_transport_turn_deg','old_projected_odd_slope_initial','old_projected_odd_slope_common'):
                    delta=abs(float(rot39[fld])-float(prev[fld]))
                    max_old_rotation_replay=max(max_old_rotation_replay,delta)
                    if delta>1.e-7+2.e-5*abs(float(prev[fld])):raise RuntimeError('v18.39 frame replay failed: '+fld)
                for vr in [v for v in vectors if v['case_id']==case_id]:
                    oldv=vectors39map.get((case_id,round(float(vr['alpha']),10),vr['moment']))
                    if oldv is None:raise RuntimeError('v18.39 vector row missing')
                    discrepancy=float(np.linalg.norm(np.asarray(vr['source_components'],float)-np.asarray(json.loads(oldv['source_components'].replace("'",'"')),float)))
                    max_parent_vector_replay=max(max_parent_vector_replay,discrepancy)
                    if discrepancy>2.e-9:raise RuntimeError('Original v18.39 9D vector replay failed')
                # The new v18.40 time sweep integrates nothing new beyond these
                # original 61 deterministic 11D integrations. Every diagnostic
                # time is within the COMMON support interval, including its end.
                from gauge_temporal_math_v18_40 import temporal_case_audit
                data=temporal_case_audit(case_id,name,a,branch,trialsol,geom,scale,q,center,theta0,opts,mu,
                    n0,frame0,t_common,tangent,s0,st,rot39,prev)
                temporal.extend(data['temporal']);gauges.extend(data['gauges']);resolution.extend(data['resolution'])
                family_temporal.append(data['family'])
                max_chart_alignment=max(max_chart_alignment,data['max_chart_alignment'])
                max_twist_alignment=max(max_twist_alignment,data['max_twist_alignment'])
                max_gauge_norm=max(max_gauge_norm,data['max_gauge_norm'])
                from original_response_plane_v18_41_math import response_plane_case_audit
                old40=original40_families.get(case_id)
                oldres40=original40_resolution.get((case_id,32))
                if old40 is None or oldres40 is None:raise RuntimeError('Authentic v18.40 family/32-grid missing')
                # Verify complete parent endpoint and nested angular-grid figures.
                for fld,observed in (
                    ('endpoint_turn_direct_deg',data['family']['endpoint_turn_direct_deg']),
                    ('angular_travel_32_intervals_deg',data['family']['angular_travel_32_intervals_deg']),
                    ('common_within_support_time',t_common),
                ):
                    oldnum=float(old40[fld]);diff=abs(oldnum-observed)
                    if diff>2.e-7+1.e-5*abs(oldnum):raise RuntimeError('v18.40 replay failed: '+fld)
                plane=response_plane_case_audit(case_id,name,a,branch,trialsol,geom,scale,q,center,theta0,opts,
                    n0,frame0,t_common,float(old40['endpoint_turn_direct_deg']),
                    float(oldres40['total_spherical_angular_travel_deg']))
                plane_profiles.extend(plane['profile'])
                plane_resolutions.extend(plane['resolutions'])
                plane_families.append(plane['family'])
                # Verify v18.41 ORIGINAL parent BEFORE interpreting new v18.42 data.
                old41=original41_family.get(case_id)
                if old41 is None:raise RuntimeError('Authenticated original v18.41 family missing')
                noise=case_audit(case_id,name,a,branch,trialsol,geom,scale,q,center,theta0,opts,
                    n0,frame0,t_common,f,old41)
                noise_profiles.extend(noise['profile']);noise_windows.extend(noise['windows'])
                noise_derivatives.extend(noise['derivatives']);noise_families.append(noise['family'])
                # Independent empirical tolerance ladder, strictly two predeclared
                # high-angle original Q550 cases; no new seeds, no new kappa0.
                focus=('phaseblind' in name and a in (3.5,4.0) and branch==0)
                if focus:
                    baseline_response=noise['response']
                    baseline_vectors=np.asarray([baseline_response(i/128) for i in range(129)])
                    baseline_windows=window_diagnostics(baseline_vectors)
                    focus_set={-round(1./3.,10),0.,round(1./3.,10)}
                    for mode,rtolf,atolf,stepf in [('coarse',10.,10.,2.),('tight',.1,.1,.5)]:
                        auxiliary=[]
                        for alpha,y,C,oldsol,status,exitt,horizon in trialsol:
                            if round(float(alpha),10) not in focus_set:continue
                            newsol=solve_ivp(rhs,(0.,T),np.asarray(oldsol.y[:,0],float),
                                method='DOP853',rtol=args.rtol*rtolf,atol=args.atol*atolf,
                                max_step=args.max_step*stepf,dense_output=True)
                            if not newsol.success:raise RuntimeError('Original tolerance ladder failed: '+newsol.message)
                            extra_solves+=1
                            # Confirm that rerun remains inside the comparison
                            # horizon; reference exit classification is unchanged.
                            zz=np.asarray(newsol.sol(np.linspace(0,t_common,65))).T
                            d_alt=np.linalg.norm((zz[:,:10]-zz[:,11:21])/scale,axis=1)
                            if np.max(d_alt)>radius+1e-10:raise RuntimeError('Tolerance rerun left local diagnostic support')
                            auxiliary.append((alpha,y,C,newsol,status,exitt,horizon))
                        alternative,_=make_response(auxiliary,geom,scale,q,center,theta0,opts,n0,frame0,t_common)
                        alt_vectors=np.asarray([alternative(i/128) for i in range(129)])
                        alt_windows=window_diagnostics(alt_vectors)
                        for base_w,alt_w in zip(baseline_windows,alt_windows):
                            w=base_w['window_fraction'];end=round(128*w)
                            rawdiff=np.linalg.norm(baseline_vectors[:end+1]-alt_vectors[:end+1],axis=1)
                            noise_solver.append(dict(case_id=case_id,id=name,anchor_xi=a,branch=branch,
                                solver_tier=mode,solver_rtol=args.rtol*rtolf,
                                solver_atol=args.atol*atolf,solver_max_step=args.max_step*stepf,
                                window_fraction=w,max_source_vector_disagreement=float(np.max(rawdiff)),
                                baseline_raw_chord_abs=base_w['chord_max_abs'],
                                variant_raw_chord_abs=alt_w['chord_max_abs'],
                                abs_chord_response_disagreement=abs(base_w['chord_max_abs']-alt_w['chord_max_abs']),
                                source_disagreement_to_chord_ratio=float(np.max(rawdiff)/max(base_w['chord_max_abs'],1e-30)),
                                baseline_endpoint_turn_deg=base_w['end_angle_deg'],
                                variant_endpoint_turn_deg=alt_w['end_angle_deg'],
                                end_angle_disagreement_deg=abs(base_w['end_angle_deg']-alt_w['end_angle_deg'])))
                print('V18_42',name,a,branch,'chord_slope',f"{noise['family']['chord_slope']:.4g}",
                      'focus_ladder',focus,flush=True)
                # Compare the shared observation horizon and exit counts to v18.38.
                for rr in summaries[-2:]:
                    oldsum=old_summary_map.get((case_id,rr['moment']))
                    if oldsum is None:raise RuntimeError('v18.38 reference summary missing')
                    if abs(float(rr['common_within_support_time'])-float(oldsum['common_within_support_time']))>1.e-10:
                        raise RuntimeError('v18.38 common diagnostic time changed')
                    if int(rr['observed_exit_count'])!=int(oldsum['observed_exit_count']) or int(rr['right_censored_count'])!=int(oldsum['right_censored_count']):
                        raise RuntimeError('v18.38 exit/censoring counts changed')
                print('V18_39',name,a,branch,
                    'observed_exits',exit_stats['observed_exit_count'],
                    'right_censored',exit_stats['right_censored_count'],
                    'horizon_min',f'{min(horizons):.4g}',
                    'common_t',f'{t_common:.4g}',flush=True)
    out=R/'results';out.mkdir(exist_ok=True)
    write_csv(out/'original_v18_40_temporal_sources.csv',temporal)
    write_csv(out/'original_v18_40_gauge_comparison.csv',gauges)
    write_csv(out/'original_v18_40_family_summary.csv',family_temporal)
    write_csv(out/'original_v18_40_resolution.csv',resolution)
    write_csv(out/'original_v18_41_plane_profiles.csv',plane_profiles)
    write_csv(out/'original_v18_41_curvature_resolution.csv',plane_resolutions)
    write_csv(out/'original_v18_41_family_summary.csv',plane_families)
    write_csv(out/'original_v18_42_odd_vector_profiles.csv',noise_profiles)
    write_csv(out/'original_v18_42_window_scaling.csv',noise_windows)
    write_csv(out/'original_v18_42_second_derivative_stencils.csv',noise_derivatives)
    write_csv(out/'original_v18_42_family_summary.csv',noise_families)
    write_csv(out/'original_v18_42_tolerance_noise.csv',noise_solver)
    proof=dict(version='v18.40',status='ORIGINAL_Q550_GAUGE_ROBUST_SOURCE_TEMPORAL_COMPLETED',
        source_kappa0_unchanged=True,original_ode_sha256_verified=True,
        parent_sha256=auth40['parent_sha256'],parent_verified_members=auth40['verified_members'],
        original_integrations=num_solves,original_cases=len(family_temporal),
        temporal_samples=len(temporal),gauge_samples=len(gauges),resolution_rows=len(resolution),
        max_v18_38_scalar_replay_discrepancy=max_parent_discrepancy,
        max_v18_39_vector_replay_discrepancy=max_parent_vector_replay,
        max_v18_39_rotation_replay_discrepancy=max_old_rotation_replay,
        max_chart_coordinate_alignment=max_chart_alignment,
        max_twisted_coordinate_alignment=max_twist_alignment,
        max_path_norm_error=max_gauge_norm,
        max_vector_budget_error=max(r['vector_gradient_reconstruction_error'] for r in rows),
        max_level_budget_error=max(r['level_balance_error'] for r in rows),
        observed_exit_count=total_observed_exits,right_censored_count=total_right_censored,
        no_material_metric_certified=True,no_physical_constraint_certified=True,
        method=dict(amplitudes=list(ALPHAS),rtol=args.rtol,atol=args.atol,max_step=args.max_step,time_window=T,
            temporal_fractions='0,1/128,1/64,1/32,1/16,1/8,3/16,1/4,3/8,1/2,5/8,3/4,7/8,1',
            path_meshes='8 vs 64 normal increments; reference based on alpha=0 base normal',
            angular_resolution='odd source 33 samples; total spherical angular travel on 4,8,16,32 intervals',
            coordinate_gauges='direct moving frame; instantaneous Householder; artificial SO(9) chart; exact frame alignment',
            transfer_connections='direct minimal plane reference vs stepwise incremental minimal rotation; not gauge equivalent a priori',
            caution='Angles of transverse diagnostic residual response, not intrinsic material torsion. Metric is conventional; no invariant constraint. kappa0 unchanged.'))
    (out/'original_v18_40_proof.json').write_text(json.dumps(proof,indent=2,allow_nan=False)+'\n')
    if num_solves!=61 or len(plane_profiles)!=12*129 or len(plane_resolutions)!=12*4 or len(plane_families)!=12:
        raise RuntimeError('Incomplete original v18.41 plane audit')
    result=dict(version='v18.41',status='ORIGINAL_Q550_ODD_SOURCE_RESPONSE_PLANE_CURVATURE_COMPLETED',
        original_integrations=num_solves,original_cases=len(plane_families),
        original_samples=len(plane_profiles),resolution_rows=len(plane_resolutions),
        source_kappa0_unchanged=True,original_ode_sha256_verified=True,
        parent_original_v18_40_sha256=auth41['parent_sha256'],
        parent_original_v18_40_verified_members=auth41['verified_members'],
        v18_39_parent_sha256=auth40['parent_sha256'],
        max_parent_endpoint_angle_replay_deg=max(x['endpoint_angle_replay_error_deg'] for x in plane_families),
        max_parent_angular_travel32_replay_deg=max(x['angular_path32_replay_error_deg'] for x in plane_families),
        max_time_varying_chart_closure_error=max(x['max_time_varying_chart_closure_error'] for x in plane_families),
        max_v18_38_scalar_replay_discrepancy=max_parent_discrepancy,
        max_v18_39_vector_replay_discrepancy=max_parent_vector_replay,
        max_v18_39_rotation_replay_discrepancy=max_old_rotation_replay,
        no_material_metric_certified=True,no_physical_constraint_certified=True,
        method=dict(declared_metric='normalized Euclidean material10',
            time_fractions='129 uniform including endpoints, within common diagnostic support',
            endpoint_plane='span of normalized odd-source initial and final 9D vectors ONLY',
            plane_deviation='orthogonal 9D component relative to endpoint plane',
            affine_raw_chord='linear interpolation of raw unnormalized odd-source 9D endpoint vectors',
            spherical_geodesic_curvature='central finite difference, only interior nodes, normalized direction on S8',
            raw_euclidean_curvature='central finite difference; scaled time s=t/t_common',
            nested_grids='16,32,64,128 intervals; report numerical convergence, not interval bounds',
            coordinate_change='SO9 time-dependent gauge undone before plane metrics',
            caution='Angular geodesicity and plane membership do not imply intrinsic material torsion or surface invariance'))
    (out/'original_v18_41_proof.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    if not (num_solves==61 and extra_solves==12 and len(noise_profiles)==1548
            and len(noise_windows)==48 and len(noise_derivatives)==144
            and len(noise_families)==12 and len(noise_solver)==16):
        raise RuntimeError('Incomplete original Q550 v18.42 integration or results')
    result42=dict(version='v18.42',
        status='ORIGINAL_Q550_SECOND_DERIVATIVE_NOISE_WINDOW_SCALING_COMPLETED',
        original_integrations=num_solves+extra_solves,baseline_original_integrations=num_solves,
        tolerance_original_integrations=extra_solves,original_cases=len(noise_families),
        vector_temporal_samples=len(noise_profiles),window_rows=len(noise_windows),
        second_derivative_rows=len(noise_derivatives),tolerance_comparison_rows=len(noise_solver),
        original_ode_sha256_verified=True,source_kappa0_unchanged=True,
        parent_original_v18_41_sha256=auth42['parent_sha256'],
        parent_original_v18_41_verified_members=auth42['verified_members'],
        max_parent_endpoint_angle_replay_deg=max(x['endpoint_angle_replay_error_deg'] for x in noise_families),
        max_parent_chord_replay=max(x['v18_41_chord_replay_error'] for x in noise_families),
        max_parent_plane_replay=max(x['v18_41_plane_replay_error'] for x in noise_families),
        no_analytic_second_variation_claim=True,no_certified_curvature=True,
        no_material_metric_certified=True,no_physical_constraint_certified=True,
        method=dict(time_window_fractions=list(WINDOWS),window_nested_uniform_sampling='129 samples, 4 nested windows, all inside common support',
            derivative_stencils='central three/five and degree-4 seven-point least squares, scale s=t/t_common',
            first_derivative_cross_check='Euler directional ORIGINAL RHS first derivative vs dense-solution first derivative (NOT analytic second derivative)',
            solver_ladder='baseline 3e-12/3e-13 vs x10 coarser and x10 tighter, two predeclared phaseblind branch-0 families, 3 amplitudes each',
            noise_floor='empirical solver differences and stencil disagreement; NOT rigorous bounds',
            caution='Curvature of the odd residual-gradient diagnostic in declared normalized Euclidean gauge, no intrinsic torsion or material constraint'))
    (out/'original_v18_42_proof.json').write_text(json.dumps(result42,indent=2,allow_nan=False)+'\n')
    print('ORIGINAL_V18_42_COMPLETED',num_solves+extra_solves,len(noise_profiles),len(noise_windows),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--base-root',type=Path,default=pinned.DEFAULT_BASE)
    ap.add_argument('--rtol',type=float,default=3e-12);ap.add_argument('--atol',type=float,default=3e-13)
    ap.add_argument('--max-step',type=float,default=2.5e-7)
    run(ap.parse_args())
