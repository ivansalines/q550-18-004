"""Diagnostic 9D odd-response plane/curvature mathematics.

The material metric and temporal connection are DECLARED conventions.
No quantity in this module certifies a physical material torsion or invariant.
"""
import numpy as np
from gauge_temporal_math_v18_40 import stable_turn


def plane_fit(vectors, floor=1e-14):
    """Fit the *endpoint-defined* two-plane, independently of intermediate samples.

    Report both the absolute normalized departure from the plane and the
    (unoriented) angular deviation.  No intermediate vector enters the fit.
    """
    v=np.asarray(vectors,float)
    if v.ndim!=2 or v.shape[1]<3 or v.shape[0]<3 or not np.isfinite(v).all():
        raise ValueError('Expected finite N x D vectors, N>=3, D>=3')
    norms=np.linalg.norm(v,axis=1)
    if np.min(norms)<floor:raise ValueError('Odd-source orientation unresolved')
    u=v/norms[:,None];e0=u[0]
    direction=u[-1]-(e0@u[-1])*e0
    se=float(np.linalg.norm(direction))
    if se<1e-8:raise ValueError('Endpoint-defined response plane is ill-conditioned')
    e1=direction/se
    coeff=np.c_[u@e0,u@e1]
    projection=coeff[:,0,None]*e0 + coeff[:,1,None]*e1
    residual=u-projection
    departure=np.linalg.norm(residual,axis=1)
    # Numerically stable off-plane angle, rather than arccos near unity.
    off_angle=np.degrees(np.arctan2(departure,np.linalg.norm(projection,axis=1)))
    phase=np.unwrap(np.arctan2(coeff[:,1],coeff[:,0]))
    endpoint=float(stable_turn(v[0],v[-1]))
    # Endpoint-interpolated RAW response vector: not a spherical slerp fit.
    tt=np.linspace(0.,1.,len(v))
    chord=(1-tt[:,None])*v[0]+tt[:,None]*v[-1]
    chord_error=np.linalg.norm(v-chord,axis=1)
    norm_scale=max(float(np.linalg.norm(v[-1]-v[0])),float(np.linalg.norm(v[0])),floor)
    return dict(norm=norms,directions=u,e0=e0,e1=e1,
                plane_components=coeff,plane_deviation=departure,
                plane_deviation_angle_deg=off_angle,phase_rad=phase,
                endpoint_turn_deg=endpoint,raw_chord_deviation=chord_error,
                raw_chord_scale=norm_scale)


def angular_path(u, stride=1):
    x=np.asarray(u,float)[::stride]
    if (len(u)-1)%stride:x=np.r_[x,np.asarray(u,float)[-1,None]]
    return float(sum(stable_turn(x[i],x[i+1]) for i in range(len(x)-1)))


def curve_curvature(vectors, samples_per_horizon):
    """Central finite-difference Euclidean curvature and geodesic curvature.

    The speed parameter is s=t/T in [0,1].  Exclude first and last TWO
    nodes to avoid boundary finite-difference differentiation artifacts.
    Low-speed nodes are marked NaN instead of fabricating curvature.
    Curvature magnitudes depend on the declared Euclidean normalization.
    """
    a=np.asarray(vectors,float)
    if a.shape[0]!=samples_per_horizon+1:raise ValueError('Uniform temporal grid required')
    ds=1./samples_per_horizon
    u=a/np.linalg.norm(a,axis=1)[:,None]
    d1=np.gradient(u,ds,axis=0,edge_order=2)
    d2=np.gradient(d1,ds,axis=0,edge_order=2)
    raw1=np.gradient(a,ds,axis=0,edge_order=2)
    raw2=np.gradient(raw1,ds,axis=0,edge_order=2)
    speed=np.linalg.norm(d1,axis=1)
    speed0=np.linalg.norm(raw1,axis=1)
    t=d1/np.maximum(speed[:,None],1e-300)
    traw=raw1/np.maximum(speed0[:,None],1e-300)
    # Intrinsic sphere geodesic curvature: exclude radial accel AND tangential accel.
    source=d2-(np.sum(d2*u,axis=1)[:,None])*u
    source=source-(np.sum(source*t,axis=1)[:,None])*t
    kc=np.linalg.norm(source,axis=1)/np.maximum(speed**2,1e-300)
    kr=np.linalg.norm(raw2-(np.sum(raw2*traw,axis=1)[:,None])*traw)/np.maximum(speed0**2,1e-300)
    safe=(speed>max(1e-10,float(np.max(speed))*1e-6))&(speed0>max(1e-14,float(np.max(speed0))*1e-6))
    safe[:2]=False;safe[-2:]=False
    kc[~safe]=np.nan;kr[~safe]=np.nan
    return dict(geodesic=kc,raw=kr,directional_speed=speed,raw_speed=speed0,safe=safe)


def summarize_response(vectors, n):
    """Independent endpoint plane and angular/curvature analysis at grid n."""
    v=np.asarray(vectors,float)
    fit=plane_fit(v)
    curv=curve_curvature(v,n)
    if int(np.sum(curv['safe']))<3:raise ValueError('Too few numerically supported curvature nodes')
    mid=np.arange(1,len(v)-1)
    turns=np.diff(fit['phase_rad'])
    neg=turns[turns<0]
    path=angular_path(fit['directions'])
    return dict(uniform_intervals=n,endpoint_turn_deg=fit['endpoint_turn_deg'],
        total_spherical_path_deg=path,path_excess_deg=path-fit['endpoint_turn_deg'],
        max_plane_deviation_fraction=float(np.max(fit['plane_deviation'])),
        median_plane_deviation_fraction=float(np.median(fit['plane_deviation'][mid])),
        max_off_plane_angle_deg=float(np.max(fit['plane_deviation_angle_deg'])),
        max_raw_chord_error_rel=float(np.max(fit['raw_chord_deviation'])/fit['raw_chord_scale']),
        median_raw_chord_error_rel=float(np.median(fit['raw_chord_deviation'])/fit['raw_chord_scale']),
        signed_plane_phase_advance_deg=float(np.degrees(fit['phase_rad'][-1]-fit['phase_rad'][0])),
        count_reverse_plane_phase_steps=len(neg),
        reverse_plane_phase_total_deg=float(np.degrees(np.sum(-neg))) if len(neg) else 0.,
        max_geodesic_curvature=float(np.nanmax(curv['geodesic'])),
        median_geodesic_curvature=float(np.nanmedian(curv['geodesic'])),
        max_raw_euclidean_curvature=float(np.nanmax(curv['raw'])),
        median_raw_euclidean_curvature=float(np.nanmedian(curv['raw'])),
        curvature_resolved_nodes=int(np.sum(curv['safe'])),
        start_norm=float(fit['norm'][0]),end_norm=float(fit['norm'][-1]),
        norm_ratio=float(fit['norm'][-1]/fit['norm'][0]))
