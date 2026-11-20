"""Pure finite-resolution response-curvature diagnostics. No intrinsic metric.

All time derivatives below are with respect to s=t/t_common (not physical
material time), and therefore depend explicitly on the declared window scale.
Finite-difference instability is never promoted to material curvature.
"""
import numpy as np
from response_plane_math_v18_41 import plane_fit
from gauge_temporal_math_v18_40 import stable_turn

WINDOWS=(1.,.5,.25,.125)
STEPS=(8,16,32)  # offset in subdivisions of each WINDOW for center stencils

def chord_metrics(v,w,global_scale=None):
    """v contains uniform samples on [0,w], with its OWN s grid."""
    v=np.asarray(v,float)
    if v.ndim!=2 or v.shape[0]<9 or not np.isfinite(v).all():raise ValueError('Invalid finite time vectors')
    if w<=0:raise ValueError('Window length must be positive')
    d=v[-1]-v[0]
    s=np.linspace(0,1,len(v))
    line=v[0][None,:]+s[:,None]*d[None,:]
    err=np.linalg.norm(v-line,axis=1)
    denom=max(float(np.linalg.norm(v[0])),float(np.linalg.norm(d)),1e-30) if global_scale is None else global_scale
    fit=plane_fit(v)
    return dict(window_fraction=float(w),end_angle_deg=fit['endpoint_turn_deg'],
        chord_max_abs=float(np.max(err)),chord_max_rel=float(np.max(err)/denom),
        plane_max_frac=float(np.max(fit['plane_deviation'])),
        endpoint_change_norm=float(np.linalg.norm(d)),
        local_start_norm=float(np.linalg.norm(v[0])),
        chord_peak_fraction=float(s[int(np.argmax(err))]))

def centered_accel(read,time,step):
    """Second time derivative from two 3/5-point stencils and quartic LS.

    Returned acceleration is wrt s = t/t_common. No analytic ODE Hessian
    is represented by the finite differences below.
    """
    if step<=0:raise ValueError('Positive derivative step required')
    v={i:np.asarray(read(float(time+i*step)),float) for i in range(-3,4)}
    if not all(np.isfinite(x).all() for x in v.values()):raise ValueError('Invalid source response')
    dd3=(v[1]-2*v[0]+v[-1])/step**2
    dd5=(-v[2]+16*v[1]-30*v[0]+16*v[-1]-v[-2])/(12*step**2)
    X=np.vander(np.arange(-3,4,dtype=float),N=5,increasing=True)
    coef=np.linalg.lstsq(X,np.stack([v[i] for i in range(-3,4)]),rcond=None)[0]
    dd_ls=2*coef[2]/step**2
    # First derivatives, needed for geometric normal acceleration, in same units.
    d5=(v[-2]-8*v[-1]+8*v[1]-v[2])/(12*step)
    # Curvature is computed using transverse acceleration only; scalar value
    # can be unstable when acceleration is beneath roundoff.
    ss=float(dd5@dd5)
    def normal_part(dd):return dd-(float(dd@dd5)/ss)*dd5 if ss>1e-50 else np.zeros_like(dd)
    a3=normal_part(dd3);a5=normal_part(dd5);als=normal_part(dd_ls)
    denom=max(np.linalg.norm(v[0]),1e-30)
    return dict(step=float(step),center_fraction=float(time),
        accel3_norm=float(np.linalg.norm(dd3)),accel5_norm=float(np.linalg.norm(dd5)),
        accel_poly4_norm=float(np.linalg.norm(dd_ls)),
        accel3_5_disagreement_rel=float(np.linalg.norm(dd3-dd5)/max(np.linalg.norm(dd5),denom*1e-12)),
        accel5_poly4_disagreement_rel=float(np.linalg.norm(dd5-dd_ls)/max(np.linalg.norm(dd5),denom*1e-12)),
        accel3_transverse_norm=float(np.linalg.norm(a3)),
        accel5_transverse_norm=float(np.linalg.norm(a5)),
        accel_poly4_transverse_norm=float(np.linalg.norm(als)),
        curvature3=float(np.linalg.norm(a3)/ss) if ss>1e-50 else float('nan'),
        curvature5=float(np.linalg.norm(a5)/ss) if ss>1e-50 else float('nan'),
        curvature_poly4=float(np.linalg.norm(als)/ss) if ss>1e-50 else float('nan'),
        first_derivative_norm=float(np.sqrt(ss)))

def window_diagnostics(full):
    """Nested windows from a single uniformly spaced [0,1] input grid of 129 points."""
    v=np.asarray(full,float)
    if v.shape[0]!=129:raise ValueError('Must have 129 original dense-source samples')
    global_scale=max(np.linalg.norm(v[0]),np.linalg.norm(v[-1]-v[0]),1e-30)
    out=[]
    for w in WINDOWS:
        j=round(w*128)
        subset=v[:j+1]
        m=chord_metrics(subset,w,global_scale)
        m['uniform_subintervals']=int(j)
        out.append(m)
    return out

def loglog_slope(w,y):
    w=np.asarray(w,float);y=np.asarray(y,float)
    ok=np.isfinite(y)&(y>0)&(w>0)
    if sum(ok)<3:return float('nan')
    return float(np.polyfit(np.log(w[ok]),np.log(y[ok]),1)[0])

def convergence_ratio(a,b):
    return float(np.linalg.norm(np.asarray(a)-np.asarray(b))/max(np.linalg.norm(np.asarray(a)),1e-30))
