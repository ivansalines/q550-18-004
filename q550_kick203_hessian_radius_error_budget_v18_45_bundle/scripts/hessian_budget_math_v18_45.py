"""Hessian radius diagnostic: *reference-dependent* numerical error budgets.
JAC and RHS are distinct finite-difference pathways; neither is exact.
Never interpret a smallest cross-path difference as a certified optimal step.
"""
import numpy as np
RHS_H=(.0015,.002,.003,.004,.006,.008,.012,.018,.024,.036,.048)
JAC_H=(.00025,.0005,.001,.002,.004,.008)
ANCHOR_J=(.0005,.001,.002)

def relative(x,y):
    x=np.asarray(x,float);y=np.asarray(y,float)
    return float(np.linalg.norm(x-y)/max(np.linalg.norm(x),np.linalg.norm(y),1e-30))

def jac_reference(items):
    return np.median(np.stack([items[h] for h in ANCHOR_J]),axis=0)

def neighbor_instability(vs,i):
    """Local adjacent-radius change; boundary uses nearest available neighbor."""
    if len(vs)<2:raise ValueError('Need multiple radii')
    v=np.asarray(vs[i]);near=[vs[j] for j in (i-1,i+1) if 0<=j<len(vs)]
    return max(relative(v,a) for a in near)

def select_radius(rad,vals,reference):
    """Descriptive minimum *cross-path* residual, not an optimal certified radius."""
    errors=np.array([relative(v,reference) for v in vals])
    best=int(np.argmin(errors))
    return best,errors

def squared_error_shares(discrepancy):
    x=np.asarray(discrepancy,float);sq=x*x
    return sq/max(float(sq.sum()),1e-60)

def component_budget(ref,test):
    ref=np.asarray(ref,float);test=np.asarray(test,float)
    delta=test-ref;shares=squared_error_shares(delta)
    return delta,shares

def observed_power(a,b,ratio):
    if a<=0 or b<=0 or ratio<=1:raise ValueError('Unfit power samples')
    return float(np.log(b/a)/np.log(ratio))

def forecast_richardson(xh,xh2,order=4):
    """Only a diagnostic Richardson extrapolation. Never an exact solution."""
    if order<=0:raise ValueError('Invalid order')
    return np.asarray(xh2)+(np.asarray(xh2)-np.asarray(xh))/(2.**order-1.)

def vec_cols(v,prefix):return {f'{prefix}{i}':float(x) for i,x in enumerate(v)}
