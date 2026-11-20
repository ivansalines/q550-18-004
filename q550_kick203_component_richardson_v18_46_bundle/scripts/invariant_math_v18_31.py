"""Explicit low-degree scalar candidate dictionary, independent derivative checks.

Only constant-coefficient polynomials of material coordinates of degree <=2.
Models here are empirical diagnostics; local conservation cannot certify an
intrinsic material first integral. Constant terms are excluded by design.
"""
import numpy as np
from itertools import combinations_with_replacement
N=10
PAIRS=list(combinations_with_replacement(range(N),2))
def feature(y,center,scale,degree):
    x=(np.asarray(y,float)[:10]-center)/scale
    if degree==1:return x.copy()
    return np.r_[x,np.array([x[i]*x[j] for i,j in PAIRS])]
def feature_derivative(y,f,center,scale,degree):
    x=(np.asarray(y,float)[:10]-center)/scale
    xd=np.asarray(f,float)[:10]/scale
    if degree==1:return xd.copy()
    return np.r_[xd,np.array([xd[i]*x[j]+x[i]*xd[j] for i,j in PAIRS])]
def gradient_hessian(y,center,scale,q,degree):
    x=(np.asarray(y,float)[:10]-center)/scale
    gx=np.asarray(q[:10],float).copy();hx=np.zeros((10,10))
    if degree==2:
        for c,(i,j) in zip(q[10:],PAIRS):
            gx[i]+=c*x[j];gx[j]+=c*x[i]
            hx[i,j]+=c;hx[j,i]+=c
    return gx/scale,hx/scale[:,None]/scale[None,:]
def rms(a):
    a=np.asarray(a,float)
    return float(np.sqrt(np.mean(a*a)))
def select_modes(A,cutoff=1e-8,max_candidates=3):
    """Whiten columns before SVD; do NOT identify rank with physical dimension."""
    A=np.asarray(A,float)
    sizes=np.sqrt(np.mean(A*A,axis=0))
    sizes=np.maximum(sizes,max(float(np.max(sizes))*1e-14,1e-30))
    uu,s,vh=np.linalg.svd(A/sizes,full_matrices=False)
    out=[]
    for k in range(len(s)-1,-1,-1):
        if s[k]/max(s[0],1e-30)>=cutoff:continue
        q=vh[k]/sizes;q=q/np.linalg.norm(q)
        out.append(dict(singular_index=int(k),sratio=float(s[k]/max(s[0],1e-30)),coeff=q))
        if len(out)>=max_candidates:break
    return s,out
def derivative_score(A,q):
    A=np.asarray(A,float);q=np.asarray(q,float)
    intrinsic=np.sqrt(np.mean((A*q)**2,axis=0).sum())
    return rms(A@q)/max(intrinsic,1e-30)
def tangent_normal_curvature(y,seed,center,scale,q,degree):
    G,H=gradient_hessian(y,center,scale,q,degree)
    gn=np.linalg.norm(G)
    if gn<1e-14:raise ValueError('Singular candidate gradient')
    v=np.asarray(seed,float)[:10]
    u=v-G*(G@v)/(gn*gn)
    if np.linalg.norm(u)<1e-12:raise ValueError('Candidate annihilates seed')
    u=u/np.linalg.norm(u)
    curvature=float(u@H@u)
    w=-G*curvature/(gn*gn)
    return u,w,dict(first_constraint_defect=float(abs(G@u)),second_constraint_defect=float(abs(G@w+curvature)),normal_curvature=float(np.linalg.norm(w)),curvature_scalar=curvature)
