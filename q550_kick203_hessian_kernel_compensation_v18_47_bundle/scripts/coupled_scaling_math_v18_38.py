"""Diagnostic on-surface displacement and signed coupled level/normal response.

All norms and orientations depend on the declared normalized coordinates.
"""
import numpy as np
from residual_gradient_math_v18_37 import unit

def tangent_to_normal(g,probe):
    n=unit(g);v=np.asarray(probe,float);v=v-n*float(n@v)
    return unit(v)

def signed_initial_sources(geom,direction):
    """Oriented signed level-rate and normal-tilt-rate (not material invariants)."""
    nu=unit(geom['g']);u=tangent_to_normal(nu,direction)
    r=float(geom['r']/np.linalg.norm(geom['g']))
    omega=float(geom['perp']@u/np.linalg.norm(geom['g']))
    return r,omega

def signed_tilt(g_hom,g_direct,direction):
    h=unit(g_hom);g=unit(g_direct)
    if h@g<0:g=-g
    u=tangent_to_normal(h,direction)
    sinpart=float(g@u);cosval=float(g@h)
    return float(np.arctan2(sinpart,cosval))

def response_fit(samples,field,train_scale=1./3.):
    """Fit a linear odd response on symmetric inner amplitudes; test outer amplitudes.

    This is a held-out linear-response check, NOT evidence of constitutive memory.
    """
    by={round(float(row['alpha']),10):float(row[field]) for row in samples}
    for a in (0.,-train_scale,train_scale,-1.,1.):
        if round(a,10) not in by:raise ValueError('Missing symmetric amplitude in response scan')
    inner=(by[round(train_scale,10)]-by[round(-train_scale,10)])/(2*train_scale)
    center=by[0.];delta=np.array([by[1.]-center,by[-1.]-center])
    expected=np.array([inner,-inner])
    scale=max(float(np.linalg.norm(delta)),float(np.linalg.norm(expected)),1e-16)
    return dict(intercept=center,odd_slope=inner,
        holdout_relative_error=float(np.linalg.norm(delta-expected)/scale),
        outer_odd_slope=float((by[1.]-by[-1.])/2.),
        outer_even_curvature=float((by[1.]+by[-1.]-2*center)/2.),
        inner_even_curvature=float((by[round(train_scale,10)]+by[round(-train_scale,10)]-2*center)/(2*train_scale*train_scale)))

def normalized_pairs(rows):
    """Level/tilt must be tested separately; their ratio is singular at a level zero."""
    x=np.array([float(r['level_rate']) for r in rows]);y=np.array([float(r['signed_tilt_rate']) for r in rows])
    if len(rows)<3:raise ValueError('Insufficient cases')
    xc=x-x.mean();yc=y-y.mean();den=float(np.linalg.norm(xc)*np.linalg.norm(yc))
    return dict(correlation=float(xc@yc/den) if den>1e-20 else None,level_span=float(np.ptp(x)),tilt_span=float(np.ptp(y)))
