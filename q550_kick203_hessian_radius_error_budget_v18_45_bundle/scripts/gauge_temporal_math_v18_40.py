"""Declared metric and gauge diagnostics; not an intrinsic material connection.

A coordinate-gauge change is undone by exact 9D orthogonal alignment.
A different temporal transport connection is NOT a mere coordinate gauge change.
"""
import json
import numpy as np
from tangent_source_math_v18_39 import minimal_rotation,normal_frame,common_vector
from residual_gradient_math_v18_37 import unit

def stable_turn(u,v,floor=1.e-14):
    u=np.asarray(u,float);v=np.asarray(v,float)
    a=float(np.linalg.norm(u));b=float(np.linalg.norm(v))
    if min(a,b)<floor:return None
    dot=float(u@v/(a*b));dot=float(np.clip(dot,-1.,1.))
    perp=float(np.linalg.norm(v-v@u/(a*a)*u))/b
    return float(np.degrees(np.arctan2(perp,dot)))

def direct_frame(n0,n,F0):return minimal_rotation(n0,n)@F0

def path_frames(normals,F0):
    """Discrete incremental minimal rotations; normals ordered in time."""
    F=np.asarray(F0,float).copy();out=[F.copy()]
    for old,new in zip(normals[:-1],normals[1:]):
        F=minimal_rotation(old,new)@F
        out.append(F.copy())
    return out

def bridge_frame(normals,frames,t_fraction,base_normal):
    """Carry last accepted grid frame to actual intermediate normal."""
    i=min(int(np.floor(t_fraction*(len(normals)-1)+1e-12)),len(normals)-1)
    return minimal_rotation(normals[i],base_normal)@frames[i]

def tangent_twist(fraction,dim=9):
    """Arbitrary deterministic pure change of 9D coordinates; NOT transport."""
    a=.43*np.sin(2*np.pi*fraction)+.19*fraction
    c=np.cos(a);s=np.sin(a);Q=np.eye(dim)
    Q[0,0]=c;Q[1,1]=c;Q[0,1]=-s;Q[1,0]=s
    b=.21*np.sin(3*np.pi*fraction)
    c=np.cos(b);s=np.sin(b)
    Q[2,2]=c;Q[3,3]=c;Q[2,3]=-s;Q[3,2]=s
    return Q

def align_coordinate(v,from_frame,to_frame):
    return np.asarray(to_frame,float).T@np.asarray(from_frame,float)@np.asarray(v,float)

def source_in_frame(source,own_normal,base_normal,frame):
    return common_vector(source,own_normal,base_normal,frame)

def odd_response(by):
    h=round(1./3.,10)
    return (np.asarray(by[h])-np.asarray(by[-h]))*1.5

def angular_travel(vectors,stride):
    seq=[vectors[i] for i in range(0,len(vectors),stride)]
    if (len(vectors)-1)%stride:seq.append(vectors[-1])
    angles=[stable_turn(a,b) for a,b in zip(seq[:-1],seq[1:])]
    return None if any(x is None for x in angles) else float(sum(angles))

def temporal_case_audit(case_id,name,a,branch,trialsol,geom,scale,q,center,theta0,opts,mu,
                        n0,frame0,t_common,tangent,s0,st,rot39,prev):
    """Measure source rotation at many times within the v18.38 common support.

    Replays the EXACT original v18.38/39 seeds and solutions provided by the
    enclosing original driver. Uses the baseline alpha=0 normal, with two
    incremental transports; coordinate charts are always explicitly aligned.
    """
    from manifold_math_v18_32 import features_material_gradient
    from coupled_scaling_math_v18_38 import signed_initial_sources
    from tangent_source_math_v18_39 import source_level_coherence
    yy={round(float(alpha),10):sol for alpha,seed,C,sol,status,exitt,horizon in trialsol}
    h=round(1./3.,10)
    if set(yy)!={-1.,-h,0.,h,1.}:raise RuntimeError('Five v18.39 replicas missing')
    def state(alpha,frac):return np.asarray(yy[round(float(alpha),10)].sol(float(frac)*t_common)[:11],float)
    def nbase(frac):
        z=state(0.,frac)
        g=features_material_gradient(z,center,scale,theta0,opts['degree'],opts['phase']).T@q*scale
        return unit(g)
    mesh8=[nbase(i/8) for i in range(9)]
    mesh64=[nbase(i/64) for i in range(65)]
    frames8=path_frames(mesh8,frame0);frames64=path_frames(mesh64,frame0)
    fractions=sorted(set([0.,1/128,1/64,1/32,1/16,1/8,3/16,1/4,3/8,1/2,5/8,3/4,7/8,1.]))
    alpha_all=(-1.,-h,0.,h,1.)
    temporal=[];gauges=[];cache={};max_chart=0.;max_twist=0.;max_path_norm=0.
    def read(alpha,frac):
        key=(alpha,frac)
        if key not in cache:
            z=state(alpha,frac);g=geom(z);sn=unit(g['g']);v=g['perp']/np.linalg.norm(g['g'])
            level_rate,signed_proj=signed_initial_sources(g,tangent)
            cache[key]=dict(normal=sn,vector=v,source=g['r'],level_rate=level_rate,
                            level_C=g['C'],old_scalar=signed_proj)
        return cache[key]
    def one(alpha,frac,base_n,F):
        obs=read(alpha,frac)
        return source_in_frame(obs['vector'],obs['normal'],base_n,F)
    def direct_odd(frac):
        bn=nbase(frac);F=direct_frame(n0,bn,frame0)
        return (one(h,frac,bn,F)-one(-h,frac,bn,F))*1.5
    odd_origin=direct_odd(0.)
    if np.linalg.norm(odd_origin)<1.e-13:raise RuntimeError('Original initial source response direction unresolved')
    fine_vectors=[direct_odd(i/32) for i in range(33)]
    path_metrics={}
    for frac in fractions:
        bn=nbase(frac);Fd=direct_frame(n0,bn,frame0)
        F8=bridge_frame(mesh8,frames8,frac,bn)
        F64=bridge_frame(mesh64,frames64,frac,bn)
        Fc=normal_frame(bn)
        Q=tangent_twist(frac);Ftw=Fd@Q
        # All observations MUST lie within the common accepted support window.
        raw={alpha:read(alpha,frac) for alpha in alpha_all}
        vectors={alpha:one(alpha,frac,bn,Fd) for alpha in alpha_all}
        v8={alpha:one(alpha,frac,bn,F8) for alpha in (-h,h)}
        v64={alpha:one(alpha,frac,bn,F64) for alpha in (-h,h)}
        odd=(vectors[h]-vectors[-h])*1.5
        odd8=(v8[h]-v8[-h])*1.5
        odd64=(v64[h]-v64[-h])*1.5
        oddscalar=(raw[h]['old_scalar']-raw[-h]['old_scalar'])*1.5
        oddlevel=(raw[h]['level_rate']-raw[-h]['level_rate'])*1.5
        oldangled=stable_turn(odd_origin,odd)
        path8angle=stable_turn(odd_origin,odd8)
        path64angle=stable_turn(odd_origin,odd64)
        if any(v is None for v in (oldangled,path8angle,path64angle)):
            raise RuntimeError('Degenerate temporal response: rotation is unresolved')
        max_norm=0.;max_chart_err=0.;max_twist_err=0.
        for alpha in alpha_all:
            vd=vectors[alpha]
            vc=one(alpha,frac,bn,Fc)
            vtw=one(alpha,frac,bn,Ftw)
            rec=align_coordinate(vc,Fc,Fd)
            rec_tw=align_coordinate(vtw,Ftw,Fd)
            max_chart_err=max(max_chart_err,float(np.linalg.norm(rec-vd)))
            max_twist_err=max(max_twist_err,float(np.linalg.norm(rec_tw-vd)))
            max_norm=max(max_norm,abs(float(np.linalg.norm(vd)-np.linalg.norm(raw[alpha]['vector']))))
        max_chart=max(max_chart,max_chart_err);max_twist=max(max_twist,max_twist_err)
        max_path_norm=max(max_path_norm,max_norm)
        if max_chart_err>2.e-10 or max_twist_err>2.e-10 or max_norm>2.e-10:
            raise RuntimeError('Coordinate-gauge and norm closure failed')
        direct_path_frame_deg=stable_turn(Fd[:,0],F64[:,0])
        # This compares frame axes, not Q550 source directions.
        path_metrics[frac]=(oldangled,path8angle,path64angle)
        temporal.append(dict(case_id=case_id,id=name,anchor_xi=a,branch=branch,
            fraction_of_common_time=frac,delta_xi=frac*t_common,within_common_support=True,
            base_normal_turn_deg=stable_turn(n0,bn),
            odd_source_turn_direct_deg=oldangled,odd_source_turn_path8_deg=path8angle,
            odd_source_turn_path64_deg=path64angle,
            odd_source_norm=float(np.linalg.norm(odd)),odd_source_gain=float(np.linalg.norm(odd)/np.linalg.norm(odd_origin)),
            odd_scalar_projection=oddscalar,odd_scalar_sign_relative_initial=float(np.sign(oddscalar*s0)),
            odd_level_rate=oddlevel,center_level_rate=raw[0.]['level_rate'],
            center_residual_r=raw[0.]['source'],
            centered_level_vector_coherence=source_level_coherence(
                [raw[x]['level_rate'] for x in alpha_all],[vectors[x] for x in alpha_all]),
            chart_closure_error=max_chart_err,twisted_chart_closure_error=max_twist_err,
            norm_preservation_error=max_norm))
        gauges.append(dict(case_id=case_id,id=name,anchor_xi=a,branch=branch,
            fraction_of_common_time=frac,delta_xi=frac*t_common,
            odd_turn_direct_deg=oldangled,odd_turn_incremental8_deg=path8angle,
            odd_turn_incremental64_deg=path64angle,
            path8_minus_direct_deg=path8angle-oldangled,
            path64_minus_direct_deg=path64angle-oldangled,
            incremental64_minus_incremental8_deg=path64angle-path8angle,
            frame_axis_direct_vs_path64_deg=direct_path_frame_deg,
            chart_coordinate_closure_error=max_chart_err,
            artificial_SO9_coordinate_closure_error=max_twist_err))
    # Time-grid convergence: total turning (not endpoint angle) of odd-response
    # direction. 4/8/16/32 uniform subintervals are nested and reproducible.
    resolution=[]
    for steps in (4,8,16,32):
        stride=32//steps
        travel=angular_travel(fine_vectors,stride)
        if travel is None:raise RuntimeError('Angular travel unresolved at temporal mesh')
        resolution.append(dict(case_id=case_id,id=name,anchor_xi=a,branch=branch,
            uniform_time_subintervals=steps,within_support_time=t_common,
            total_spherical_angular_travel_deg=travel,
            fixed_endpoint_turn_deg=stable_turn(fine_vectors[0],fine_vectors[-1]),
            endpoint_odd_norm=float(np.linalg.norm(fine_vectors[-1]))))
    last=temporal[-1]
    parentangle=float(prev['full_odd_source_transport_turn_deg'])
    if abs(last['odd_source_turn_direct_deg']-parentangle)>3.e-5:
        raise RuntimeError('v18.39 endpoint source rotation not reproduced')
    if abs(temporal[0]['odd_scalar_projection']-s0)>2.e-9 or abs(last['odd_scalar_projection']-st)>2.e-9:
        raise RuntimeError('v18.39 signed projected response not reproduced')
    flips=[]
    for v,w in zip(temporal[:-1],temporal[1:]):
        if v['odd_scalar_projection']*w['odd_scalar_projection']<0:
            flips.append([v['fraction_of_common_time'],w['fraction_of_common_time']])
    family=dict(case_id=case_id,id=name,anchor_xi=a,branch=branch,
        common_within_support_time=t_common,original_v18_39_angle_deg=parentangle,
        endpoint_turn_direct_deg=last['odd_source_turn_direct_deg'],
        endpoint_turn_path8_deg=last['odd_source_turn_path8_deg'],
        endpoint_turn_path64_deg=last['odd_source_turn_path64_deg'],
        endpoint_path8_vs_path64_difference_deg=abs(last['odd_source_turn_path8_deg']-last['odd_source_turn_path64_deg']),
        endpoint_direct_vs_path64_difference_deg=abs(last['odd_source_turn_direct_deg']-last['odd_source_turn_path64_deg']),
        max_chart_coordinate_closure_error=max_chart,
        max_twisted_SO9_coordinate_closure_error=max_twist,
        signed_projection_reversed_across_horizon=bool(s0*st<0),
        signed_projection_observed_crossing_brackets=json.dumps(flips),
        signed_projection_first_crossing_bracket=json.dumps(flips[0]) if flips else '',
        initial_signed_projection=s0,endpoint_signed_projection=st,
        first_sample_turn_deg=temporal[1]['odd_source_turn_direct_deg'],
        angular_travel_4_intervals_deg=resolution[0]['total_spherical_angular_travel_deg'],
        angular_travel_8_intervals_deg=resolution[1]['total_spherical_angular_travel_deg'],
        angular_travel_16_intervals_deg=resolution[2]['total_spherical_angular_travel_deg'],
        angular_travel_32_intervals_deg=resolution[3]['total_spherical_angular_travel_deg'],
        min_odd_source_norm=min(float(x['odd_source_norm']) for x in temporal))
    return dict(temporal=temporal,gauges=gauges,resolution=resolution,family=family,
                max_chart_alignment=max_chart,max_twist_alignment=max_twist,max_gauge_norm=max_path_norm)
