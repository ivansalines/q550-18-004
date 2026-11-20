"""Local normalized-coordinate cotangent transport tests (not a physical metric)."""
import numpy as np
from scipy.optimize import brentq

def scaled_jacobian(A,scale):
    scale=np.asarray(scale,float);A=np.asarray(A,float)
    return A[:10,:10]*scale[None,:]/scale[:,None]

def unit(x):
    x=np.asarray(x,float);n=np.linalg.norm(x)
    if not np.isfinite(n) or n<1e-14:raise ValueError('Degenerate vector')
    return x/n

def tangent_to(n,probe):
    n=unit(n);v=np.asarray(probe,float)-n*np.dot(n,probe)
    return unit(v)

def normal_angle(a,b,weights=None):
    a=np.asarray(a,float);b=np.asarray(b,float)
    if weights is not None:
        w=np.asarray(weights,float)
        a=a*w;b=b*w
    a=unit(a);b=unit(b)
    return float(np.degrees(np.arccos(np.clip(abs(np.dot(a,b)),0.,1.))))

def first_exit(times,values,threshold,exact):
    if values[0]>threshold+1e-12:return 'OUTSIDE_AT_START',0.
    inds=np.flatnonzero(np.asarray(values)>threshold)
    if not len(inds):return 'NO_EXIT_OBSERVED',None
    j=int(inds[0]);
    if j==0:return 'OUTSIDE_AT_START',0.
    a,b=float(times[j-1]),float(times[j])
    return 'CROSSED_AFTER_START',float(brentq(lambda t:exact(t)-threshold,a,b,xtol=1e-15,rtol=1e-12))

def transport_rhs(B,m,u):
    return -B.T@m,B@u

def source_budget(c0,mu,t,z):
    return np.exp(mu*np.asarray(t))*np.asarray(z)
