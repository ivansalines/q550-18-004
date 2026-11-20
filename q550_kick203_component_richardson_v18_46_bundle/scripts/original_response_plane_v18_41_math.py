"""Inside-support odd-source temporal plane audit for *original* Q550 trajectories.

No original integrations are performed by this helper; only the parent driver
may supply accepted original Q550 scipy dense solutions. Synthetic unit tests
operate on separate test data and never create an ORIGINAL proof.
"""
import numpy as np
from manifold_math_v18_32 import features_material_gradient
from tangent_source_math_v18_39 import minimal_rotation,common_vector
from residual_gradient_math_v18_37 import unit
from response_plane_math_v18_41 import plane_fit,curve_curvature,summarize_response
from gauge_temporal_math_v18_40 import stable_turn,tangent_twist


def response_plane_case_audit(case_id,name,a,branch,trialsol,geom,scale,q,center,theta0,opts,
                              n0,frame0,t_common,reference_turn_deg,reference_angular_path32):
    by={round(float(alpha),10):sol for alpha,seed,C,sol,status,exitt,horizon in trialsol}
    h=round(1./3.,10)
    if set(by)!={-1.,-h,0.,h,1.}:raise RuntimeError('Original 5 amplitudes missing')
    N=128
    tt=np.linspace(0.,1.,N+1)
    vectors=[];check_max=0.
    for s in tt:
        t=float(s*t_common)
        y0=np.asarray(by[0.].sol(t)[:11],float)
        gradient=features_material_gradient(y0,center,scale,theta0,opts['degree'],opts['phase']).T@q*scale
        base_n=unit(gradient)
        F=minimal_rotation(n0,base_n)@frame0
        source={}
        for alpha in (-h,h):
            y=np.asarray(by[alpha].sol(t)[:11],float)
            G=geom(y); own_n=unit(G['g'])
            v=G['perp']/np.linalg.norm(G['g'])
            source[alpha]=common_vector(v,own_n,base_n,F)
        response=1.5*(source[h]-source[-h])
        # A deliberately time-varying chart MUST be undone.  This detects a
        # coordinate-induced apparent turn before interpreting geometry.
        Q=tangent_twist(float(s))
        transformed=Q.T@response
        restored=Q@transformed
        check_max=max(check_max,float(np.linalg.norm(restored-response)))
        vectors.append(response)
    vectors=np.asarray(vectors,float)
    if check_max>2.e-11:raise RuntimeError('Time-dependent 9D chart closure failed')
    fit=plane_fit(vectors)
    curv=curve_curvature(vectors,N)
    resolutions=[]
    for n in (16,32,64,128):
        sub=vectors[::N//n]
        info=summarize_response(sub,n)
        resolutions.append(dict(case_id=case_id,id=name,anchor_xi=a,branch=branch,**info))
    if abs(fit['endpoint_turn_deg']-reference_turn_deg)>3.e-5:
        raise RuntimeError('v18.40 original endpoint angle not reproduced')
    if abs(resolutions[1]['total_spherical_path_deg']-reference_angular_path32)>3.e-5:
        raise RuntimeError('v18.40 original 32-interval angular path not reproduced')
    phase=fit['phase_rad'];chord=(1-tt[:,None])*vectors[0]+tt[:,None]*vectors[-1]
    profile=[]
    for i,s in enumerate(tt):
        profile.append(dict(case_id=case_id,id=name,anchor_xi=a,branch=branch,
            fraction_of_common_time=float(s),delta_xi=float(s*t_common),inside_common_support=True,
            odd_source_norm=float(fit['norm'][i]),odd_source_norm_gain=float(fit['norm'][i]/fit['norm'][0]),
            endpoint_plane_phase_deg=float(np.degrees(phase[i])),
            departure_from_endpoint_plane_fraction=float(fit['plane_deviation'][i]),
            departure_from_endpoint_plane_deg=float(fit['plane_deviation_angle_deg'][i]),
            raw_chord_error_rel=float(np.linalg.norm(vectors[i]-chord[i])/fit['raw_chord_scale']),
            initial_direction_turn_deg=stable_turn(vectors[0],vectors[i]),
            directional_speed_s=float(curv['directional_speed'][i]),
            geodesic_curvature_of_direction=float(curv['geodesic'][i]) if np.isfinite(curv['geodesic'][i]) else '',
            euclidean_curvature_of_raw_response=float(curv['raw'][i]) if np.isfinite(curv['raw'][i]) else '',
            curvature_numerically_resolved=bool(curv['safe'][i])))
    p128=resolutions[-1]
    fam=dict(case_id=case_id,id=name,anchor_xi=a,branch=branch,
        common_within_support_time=t_common,parent_v18_40_endpoint_turn_deg=reference_turn_deg,
        parent_v18_40_angular_travel32_deg=reference_angular_path32,
        endpoint_angle_replay_error_deg=abs(fit['endpoint_turn_deg']-reference_turn_deg),
        angular_path32_replay_error_deg=abs(resolutions[1]['total_spherical_path_deg']-reference_angular_path32),
        max_time_varying_chart_closure_error=check_max,
        **{k:v for k,v in p128.items() if k not in ('case_id','id','anchor_xi','branch')},
        plane_residual_max_64_to_128_change=abs(resolutions[-1]['max_plane_deviation_fraction']-resolutions[-2]['max_plane_deviation_fraction']),
        path_excess_64_to_128_change_deg=abs(resolutions[-1]['path_excess_deg']-resolutions[-2]['path_excess_deg']))
    return dict(profile=profile,resolutions=resolutions,family=fam)
