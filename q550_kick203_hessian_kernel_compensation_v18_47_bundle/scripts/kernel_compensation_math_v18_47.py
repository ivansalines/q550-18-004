"""Q550 v18.47 alternative numerical kernels for H_F[u,e]^T g.

All kernels operate in the declared normalized (z,theta) chart.  They share
binary64 ORIGINAL Q550 calls.  Compensated summation cannot repair rounding
inside the original RHS/Jacobian; agreement alone is not a Hessian certificate.
"""
import math
import numpy as np
from dual_hessian_jet_math_v18_44 import normalized_full_speed

OFF=np.array([-2.,-1.,1.,2.]);WT=np.array([1.,-8.,8.,-1.])
KEY=(0,2,8)
RHS_RADII=(.004,.008,.012)
ACCEL_RADII=(.001,.002,.004)

def scalar_force(f,y,zoffset,weights,g):
    p=np.asarray(y,float)+weights*np.asarray(zoffset,float)
    raw=np.asarray(f(0.,p),float)[:10]/weights[:10]
    if raw.shape!=(10,) or not np.isfinite(raw).all():raise RuntimeError('Nonfinite Q550 field')
    return math.fsum(float(g[k])*float(raw[k]) for k in range(10))

def mixed_compensated(y,fy,f,scale,g,h,reverse=False):
    """Same 4x4 mixed field nodes as historic kernel; fsum dot and outer sum.

    reverse=True swaps the displacement construction order, checking a
    numerical Hessian interchange identity without assuming exact symmetry.
    """
    y=np.asarray(y,float);scale=np.asarray(scale,float);g=np.asarray(g,float)
    w=np.r_[scale,1.];v=normalized_full_speed(fy,scale);speed=float(np.linalg.norm(v))
    if speed<=0:raise ValueError('Degenerate speed')
    u=v/speed;out=[]
    for j in range(10):
        e=np.zeros(11);e[j]=1.
        terms=[]
        for a,wa in zip(OFF,WT):
            for b,wb in zip(OFF,WT):
                disp=(b*h*e+a*h*u) if reverse else (a*h*u+b*h*e)
                terms.append(float(wa*wb)*scalar_force(f,y,disp,w,g))
        out.append(speed*math.fsum(terms)/(144*h*h))
    return np.asarray(out)

def polarization_rhs(y,fy,f,scale,g,h):
    """Polarization H[u,e]=(H[u+e,u+e]-H[u-e,u-e])/4.

    Each directional Hessian uses a five-point SECOND derivative of scalar
    g.F_z, rather than the historic mixed 4x4 FIRST-derivative stencil.
    The direction is NOT re-normalized: this is an algebraic bilinear probe.
    """
    y=np.asarray(y,float);g=np.asarray(g,float);w=np.r_[scale,1.]
    v=normalized_full_speed(fy,scale);speed=np.linalg.norm(v)
    if speed<=0:raise ValueError('Degenerate speed')
    u=v/speed;center=scalar_force(f,y,np.zeros(11),w,g)
    def d2(vec):
        # fsum prevents the summation order from dominating the cancellation.
        samples=[-scalar_force(f,y,2*h*vec,w,g),16*scalar_force(f,y,h*vec,w,g),
                 -30*center,16*scalar_force(f,y,-h*vec,w,g),-scalar_force(f,y,-2*h*vec,w,g)]
        return math.fsum(samples)/(12*h*h)
    out=[]
    for j in range(10):
        e=np.zeros(11);e[j]=1.
        out.append(speed*(d2(u+e)-d2(u-e))/4.)
    return np.asarray(out)

def accel_gradient_identity(y,fy,f,jac,scale,g,h):
    """H_F[e_j,F] via D_j[(D F_z)F_z] - (D F_z) (D_j F_z).

    Uses ORIGINAL pinned Jacobian AND field, hence shares errors with the
    Jacobian pathway; it is NOT an independent exact Hessian or physics model.
    """
    y=np.asarray(y,float);w=np.r_[np.asarray(scale,float),1.]
    v=normalized_full_speed(fy,scale);base=np.asarray(jac(y),float)
    B=base[:10,:]*w[None,:]/scale[:,None]
    if B.shape!=(10,11):raise RuntimeError('Malformed Q550 Jacobian')
    out=[]
    for j in range(10):
        e=np.zeros(11);e[j]=1.
        terms=[]
        for a,wa in zip(OFF,WT):
            yy=y+w*(a*h*e)
            Fj=np.asarray(f(0.,yy),float)
            Bj=np.asarray(jac(yy),float)[:10,:]*w[None,:]/scale[:,None]
            acceleration=Bj@normalized_full_speed(Fj,scale)
            terms.append(float(wa)*math.fsum(float(g[k])*float(acceleration[k]) for k in range(10)))
        gradient=math.fsum(terms)/(12*h)
        flow_column=base[10,j]*scale[j]
        dv=np.r_[B[:,j],flow_column]
        correction=math.fsum(float(g[k])*float((B@dv)[k]) for k in range(10))
        out.append(gradient-correction)
    return np.asarray(out)

def relative(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float)
    return float(np.linalg.norm(a-b)/max(np.linalg.norm(a),np.linalg.norm(b),1e-30))

def shares(a):
    a=np.asarray(a,float);v=a*a;s=float(np.sum(v));return v/max(s,1e-300)

def vector_columns(prefix,v):
    return {f'{prefix}{i}':float(v[i]) for i in range(10)}
