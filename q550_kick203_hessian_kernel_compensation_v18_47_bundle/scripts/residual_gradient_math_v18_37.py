"""Exact polynomial/phase residual gradient in normalized material coordinates.

These are diagnostic coordinates and no material metric is certified.
"""
import numpy as np
from manifold_math_v18_32 import features,features_flow,features_material_gradient,PAIRS


def unit(v):
    v=np.asarray(v,float);n=float(np.linalg.norm(v))
    if not np.isfinite(n) or n<1e-25:raise ValueError('Degenerate normal')
    return v/n


def projective_angle(a,b):
    """Stable atan2 principal unoriented angle in degrees, avoiding arccos cancellation."""
    a=unit(a); b=unit(b); dp=float(a@b)
    perp=b-dp*a
    return float(np.degrees(np.arctan2(np.linalg.norm(perp),abs(dp))))


def source_geometry(y,fy,J,q,center,scale,theta0,degree,phase,mu):
    """Return C,r, grad_z C, grad_z r, and its parallel/perpendicular components.

    Assumes theta_dot independent of all material coordinates, and mu(theta) only.
    Checked separately on the original Q550 Jacobian before any interpretation.
    """
    y=np.asarray(y,float); fy=np.asarray(fy,float); J=np.asarray(J,float)
    q=np.asarray(q,float);scale=np.asarray(scale,float)
    zz=(y[:10]-center)/scale
    C=float(features(y,center,scale,theta0,degree,phase)@q)
    lie=float(features_flow(y,fy,center,scale,theta0,degree,phase)@q)
    r=lie-mu*C
    grad=features_material_gradient(y,center,scale,theta0,degree,phase).T@q*scale
    H=np.zeros((10,10))
    if degree==2:
        for k,(i,j) in enumerate(PAIRS):
            a=q[10+k];H[i,j]+=a;H[j,i]+=a
    gtheta=np.zeros(10)
    if phase:
        idx=10+(55 if degree==2 else 0)
        a=y[10]-theta0
        gtheta=q[idx:idx+10]*np.cos(a)-q[idx+10:idx+20]*np.sin(a)
    v=fy[:10]/scale
    B=J[:10,:10]*scale[None,:]/scale[:,None]
    gr=H@v+B.T@grad+fy[10]*gtheta-mu*grad
    nu=unit(grad)
    parallel=float(gr@nu)
    perp=gr-parallel*nu
    return dict(C=C,r=r,g=grad,gr=gr,parallel=parallel,perp=perp,tilt_rate=float(np.linalg.norm(perp)/np.linalg.norm(grad)),B=B)


def numerical_residual_gradient(y,f,q,center,scale,theta0,degree,phase,mu,step):
    """Independent five-point differentiation of scalar r evaluated using original F."""
    def resid(y_):
        fy=f(0.,y_)
        c=features(y_,center,scale,theta0,degree,phase)@q
        d=features_flow(y_,fy,center,scale,theta0,degree,phase)@q
        return float(d-mu*c)
    vals=[]
    for k in range(10):
        samples=[]
        for m in (-2.,-1.,1.,2.):
            yy=np.asarray(y,float).copy()
            yy[k]+=m*step*scale[k]
            samples.append(resid(yy))
        v1,v2,v3,v4=samples
        vals.append((v1-8*v2+8*v3-v4)/(12*step))
    return np.asarray(vals)


def continuous_normal_rhs(B,mu,m,e,gr):
    return -B.T@m,(-B.T+mu*np.eye(10))@e+gr


def align_coefficient(a,b):
    """Least-squares amplitude of candidate b along reference a (not a metric invariant)."""
    a=np.asarray(a,float);b=np.asarray(b,float)
    return float((a@b)/(a@a))
