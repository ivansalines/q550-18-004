"""Exact cofactor-defect transport identity and resolved quadrature diagnostics.

 C'(t)=mu(t)*C(t)+r(t); r=C'-mu*C.
 C(T)-exp(M(T))*C(0)=int_0^T exp(M(T)-M(s))*r(s) ds.

This identity is algebraic; numerical closure validates quadrature and book-
keeping, NOT material invariance or the correctness of the candidate surface.
"""
import numpy as np
from scipy.integrate import cumulative_trapezoid,cumulative_simpson

def integrate_budget(times,values,derivatives,mus):
    t=np.asarray(times,float);C=np.asarray(values,float);D=np.asarray(derivatives,float);mu=np.asarray(mus,float)
    if len(t)<5 or any(len(a)!=len(t) for a in (C,D,mu)) or not np.all(np.diff(t)>0):raise ValueError('Invalid time grid')
    if not all(np.all(np.isfinite(a)) for a in (t,C,D,mu)):raise ValueError('Nonfinite input')
    M=np.r_[0.,cumulative_trapezoid(mu,t)]
    r=D-mu*C
    source=r*np.exp(-M)
    Jtrap=np.r_[0.,cumulative_trapezoid(source,t)]
    Jsimp=np.r_[0.,cumulative_simpson(source,x=t,initial=None)]
    pred_trap=np.exp(M)*(C[0]+Jtrap)
    pred_simp=np.exp(M)*(C[0]+Jsimp)
    baseline=C[0]*np.exp(M)
    signed_direct=C-baseline
    cum_plus=np.r_[0.,cumulative_trapezoid(np.maximum(source,0.),t)]
    cum_minus=np.r_[0.,cumulative_trapezoid(np.minimum(source,0.),t)]
    return dict(t=t,C=C,D=D,mu=mu,M=M,r=r,source=source,
                Jtrap=Jtrap,Jsimp=Jsimp,pred_trap=pred_trap,pred_simp=pred_simp,
                baseline=baseline,signed_direct=signed_direct,
                signed_simpson=np.exp(M)*Jsimp,
                abs_contribution=np.exp(M)*(cum_plus-cum_minus),
                accumulated_pos=np.exp(M)*cum_plus,accumulated_neg=np.exp(M)*cum_minus)

def segment_budget(budget,a,b,T):
    t=budget['t'];idxa=int(np.argmin(abs(t-a)));idxb=int(np.argmin(abs(t-b)));idxT=int(np.argmin(abs(t-T)))
    if max(abs(t[idxa]-a),abs(t[idxb]-b),abs(t[idxT]-T))>1e-10:raise ValueError('Segment boundary must be sampled exactly')
    J=budget['Jsimp']; M=budget['M']; end_weight=float(np.exp(M[idxT]))
    signed=end_weight*(J[idxb]-J[idxa])
    return dict(begin=float(a),end=float(b),signed_to_T=float(signed),
                magnitude_to_T=float(abs(signed)),
                relative_to_total=float(signed/max(abs(budget['signed_direct'][idxT]),1e-20)))

def first_crossing(t,dist,threshold):
    mask=np.flatnonzero(np.asarray(dist)>=threshold)
    return float(t[mask[0]]) if len(mask) else None

def unoriented_angle(g,h,metric=None):
    g=np.asarray(g,float);h=np.asarray(h,float)
    if metric is not None:
        m=np.asarray(metric,float)
        if m.shape!=g.shape or np.any(m<=0):raise ValueError('Diagonal diagnostic scale must be positive')
        g=g*m;h=h*m
    ng=np.linalg.norm(g);nh=np.linalg.norm(h)
    if min(ng,nh)<1e-14:return None
    return float(np.degrees(np.arccos(np.clip(abs(np.dot(g,h))/(ng*nh),0,1))))
