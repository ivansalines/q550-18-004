"""Micro-time event bracketing and geometrical support diagnostics.

All coordinate scales and radii are declared diagnostics, not intrinsic
material distances or certified invariant neighborhoods.
"""
import numpy as np
from scipy.optimize import brentq

def microgrid(T=.0125):
    a=np.geomspace(1e-10,T,71)
    b=np.linspace(0.,T,101)
    return np.unique(np.r_[0.,a,b,T])

def diagnostic_distance(x,ref,scale):
    x=np.asarray(x,float);ref=np.asarray(ref,float);scale=np.asarray(scale,float)
    return float(np.linalg.norm((x[:10]-ref[:10])/scale))

def nearest_training(x,train,scale):
    return float(np.linalg.norm((train[:,:10]-np.asarray(x)[:10])/scale,axis=1).min())

def classify_start_and_exit(times,distances,threshold):
    """Observed bracket, distinguish seeds outside from actual outward exits."""
    t=np.asarray(times,float);d=np.asarray(distances,float)
    if d[0]>threshold:return 'OUTSIDE_AT_START',0.,0.
    idx=np.flatnonzero(d>threshold)
    if len(idx)==0:return 'NO_EXIT_OBSERVED',None,None
    j=int(idx[0]);return 'CROSSED_AFTER_START',float(t[j-1]),float(t[j])

def refine_first_exit(times,distances,threshold,continuous_distance):
    status,left,right=classify_start_and_exit(times,distances,threshold)
    if status!='CROSSED_AFTER_START':return status,None,left,right
    if left==right:return status,float(left),left,right
    root=brentq(lambda t:continuous_distance(t)-threshold,left,right,xtol=1e-13,rtol=1e-12)
    return status,float(root),left,right

def weighted_exact_budget(c0,t,mu,int_source):
    """For constant mu, integrate Z'=exp(-mu t) r and compute e^(mu t)Z."""
    return np.exp(mu*t)*np.asarray(int_source,float)

def calibration_radius(train,anchor,scale):
    ds=np.linalg.norm((np.asarray(train)[:,:10]-np.asarray(anchor)[:10])/scale,axis=1)
    return dict(radius_max=float(ds.max()),radius_95=float(np.quantile(ds,.95)),
                radius_median=float(np.median(ds)),sample_count=int(len(ds)))
