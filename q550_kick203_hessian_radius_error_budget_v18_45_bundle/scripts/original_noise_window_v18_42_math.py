"""Inside-support ORIGINAL Q550 response diagnostics from authenticated dense solves.

This helper does not run the original ODE or claim an analytical second
variational derivative. Replays EXACT geometry and frame convention of v18.41.
"""
import numpy as np
from manifold_math_v18_32 import features_material_gradient
from tangent_source_math_v18_39 import minimal_rotation,common_vector
from residual_gradient_math_v18_37 import unit
from second_derivative_noise_math_v18_42 import WINDOWS,STEPS,window_diagnostics,centered_accel,loglog_slope,convergence_ratio
from response_plane_math_v18_41 import plane_fit

H=round(1./3.,10)

def make_response(trialsol,geom,scale,q,center,theta0,opts,n0,frame0,t_common):
    by={round(float(alpha),10):sol for alpha,seed,C,sol,status,exitt,horizon in trialsol}
    if not {-H,0.,H} <= set(by):raise RuntimeError('Missing original symmetric response seeds')
    cache={}
    def response(s):
        s=float(s)
        if not (-1e-14<=s<=1+1e-14):raise RuntimeError('Source evaluated outside common support horizon')
        key=round(s,14)
        if key in cache:return cache[key]
        t=s*t_common
        y0=np.asarray(by[0.].sol(t)[:11],float)
        gg=features_material_gradient(y0,center,scale,theta0,opts['degree'],opts['phase']).T@q*scale
        bn=unit(gg);frame=minimal_rotation(n0,bn)@frame0
        vv={}
        for alpha in (-H,H):
            y=np.asarray(by[alpha].sol(t)[:11],float)
            G=geom(y)
            vec=G['perp']/np.linalg.norm(G['g'])
            vv[alpha]=common_vector(vec,unit(G['g']),bn,frame)
        cache[key]=1.5*(vv[H]-vv[-H])
        return cache[key]
    def directional_first_check(s,h,f):
        # Independent FIRST-derivative consistency check via local Euler jets
        # of ORIGINAL ODE, not a second-derivative analytic certificate.
        t=s*t_common;dt=h*t_common
        yy={alpha:np.asarray(by[alpha].sol(t)[:11],float) for alpha in (-H,0.,H)}
        ff={alpha:np.asarray(f(t,y),float) for alpha,y in yy.items()}
        def shifted(sign):
            ybase=yy[0.]+sign*dt*ff[0.]
            grad=features_material_gradient(ybase,center,scale,theta0,opts['degree'],opts['phase']).T@q*scale
            bn=unit(grad);frame=minimal_rotation(n0,bn)@frame0
            src={}
            for alpha in (-H,H):
                y=yy[alpha]+sign*dt*ff[alpha]
                G=geom(y)
                src[alpha]=common_vector(G['perp']/np.linalg.norm(G['g']),unit(G['g']),bn,frame)
            return 1.5*(src[H]-src[-H])
        euler_d=(shifted(+1)-shifted(-1))/(2*h)
        dense_d=(response(s+h)-response(s-h))/(2*h)
        return float(np.linalg.norm(euler_d-dense_d)/max(np.linalg.norm(dense_d),1e-30))
    return response,directional_first_check

def case_audit(case_id,name,a,branch,trialsol,geom,scale,q,center,theta0,opts,n0,frame0,t_common,f,old_family):
    response,first=make_response(trialsol,geom,scale,q,center,theta0,opts,n0,frame0,t_common)
    v=np.asarray([response(i/128) for i in range(129)])
    fit=plane_fit(v)
    parent_ang=float(old_family['endpoint_turn_deg'])
    parent_plane=float(old_family['max_plane_deviation_fraction'])
    parent_chord=float(old_family['max_raw_chord_error_rel'])
    if abs(fit['endpoint_turn_deg']-parent_ang)>3e-5:raise RuntimeError('v18.41 endpoint turn replay failed')
    global_scale=max(float(np.linalg.norm(v[-1]-v[0])),float(np.linalg.norm(v[0])),1e-30)
    chord=np.asarray([np.linalg.norm(vec-((1-s)*v[0]+s*v[-1]))/global_scale for s,vec in zip(np.linspace(0,1,129),v)])
    maxerr=max(float(np.max(chord)),float(np.max(fit['plane_deviation'])))
    if abs(float(np.max(chord))-parent_chord)>2e-8 or abs(float(np.max(fit['plane_deviation']))-parent_plane)>2e-8:
        raise RuntimeError('v18.41 plane/chord replay failed')
    windows=window_diagnostics(v)
    derivs=[]
    for row in windows:
        w=row['window_fraction'];center_s=w/2
        echeck=first(center_s,w/64,f)
        row.update(case_id=case_id,id=name,anchor_xi=a,branch=branch,
                   common_within_support_time=t_common,
                   first_derivative_euler_proxy_relative_error=echeck)
        for subdiv in STEPS:
            d=centered_accel(response,center_s,w/subdiv)
            derivs.append(dict(case_id=case_id,id=name,anchor_xi=a,branch=branch,
                window_fraction=w,stencil_subdivision=subdiv,**d))
    slopes=dict(chord_slope=loglog_slope([z['window_fraction'] for z in windows],[z['chord_max_abs'] for z in windows]),
        plane_slope=loglog_slope([z['window_fraction'] for z in windows],[z['plane_max_frac'] for z in windows]),
        angle_slope=loglog_slope([z['window_fraction'] for z in windows],[z['end_angle_deg'] for z in windows]))
    family=dict(case_id=case_id,id=name,anchor_xi=a,branch=branch,
        common_within_support_time=t_common,original_v18_41_endpoint_angle_deg=parent_ang,
        endpoint_angle_replay_error_deg=abs(fit['endpoint_turn_deg']-parent_ang),
        v18_41_chord_replay_error=abs(float(np.max(chord))-parent_chord),
        v18_41_plane_replay_error=abs(float(np.max(fit['plane_deviation']))-parent_plane),
        max_euler_jet_first_derivative_disagreement=max(x['first_derivative_euler_proxy_relative_error'] for x in windows),
        **slopes)
    return dict(profile=[dict(case_id=case_id,id=name,anchor_xi=a,branch=branch,
        fraction_of_common_time=i/128,delta_xi=i/128*t_common,
        odd_source_norm=float(np.linalg.norm(v[i])),
        parent_endpoint_plane_departure_frac=float(fit['plane_deviation'][i]),
        parent_full_chord_error_rel=float(chord[i]),
        **{'v'+str(k):float(v[i,k]) for k in range(9)}) for i in range(129)],
        windows=windows,derivatives=derivs,family=family,response=response)
