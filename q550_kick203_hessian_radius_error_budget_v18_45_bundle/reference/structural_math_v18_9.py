#!/usr/bin/env python3
"""Q550 v18.9: coordinate-covariant tests; NEVER infer a mechanical metric from P.

All quantities labeled ``xi`` are kick-relative coordinates; the original support
ODE time is s = s0 + slope*xi.  A pair of independent vectors has NO canonical
sign of an inner product without a metric.
"""
from __future__ import annotations
import numpy as np
from scipy.interpolate import CubicHermiteSpline
from scipy.optimize import brentq

VERSION='2026-10-18-q550-v18.9-ode-structure-metric-candidate-audit'


def roots_bracketed(x, y, yd, start=1., end=8.):
    x=np.asarray(x,float); y=np.asarray(y,float); yd=np.asarray(yd,float)
    take=(x>=start-1e-12)&(x<=end+1e-12)
    x=x[take]; y=y[take]; yd=yd[take]
    if len(x)<2 or not (np.all(np.isfinite(y)) and np.all(np.isfinite(yd))):
        raise ValueError('nonfinite/empty root data')
    fn=CubicHermiteSpline(x,y,yd)
    return [float(brentq(fn,x[k],x[k+1],xtol=1e-13)) for k in range(len(x)-1) if y[k]*y[k+1]<0]


def cosines(a,b):
    aa=np.linalg.norm(a,axis=-1); bb=np.linalg.norm(b,axis=-1)
    if np.any(aa<=0) or np.any(bb<=0):raise ValueError('zero vector in pair')
    return np.clip(np.einsum('...i,...i->...',a,b)/(aa*bb),-1.,1.)


def covariant_pair_metric(E,K,W):
    return np.einsum('ni,nij,nj->n',E,W,K)


def transport_invariant(P, A, T, W0=None):
    """For W(x)=P(x)^-T W0 P(x)^-1, (PA)^T W (PT)=A^T W0 T.

    P and W0 require units and a chosen start inner product. Not mechanical.
    """
    if W0 is None: W0=np.eye(P.shape[-1])
    ea=np.einsum('nij,j->ni',P,A)
    kt=np.einsum('nij,nj->ni',P,T)
    z=np.linalg.solve(P,np.stack([ea,kt],axis=-1))
    return np.einsum('ni,ij,nj->n',z[:,:,0],W0,z[:,:,1])


def euler_strain_birth(x,E,K,J,P,g,slope):
    """N'=slope*E^T(J^T+J)K + slope*E^T P g for fixed Euclidean metric."""
    strain=slope*np.einsum('ni,nij,nj->n',E,J+np.transpose(J,(0,2,1)),K)
    source=slope*np.einsum('ni,nij,nj->n',E,P,g)
    return strain,source


def symmetry_basis(n=11):
    basis=[];pairs=[]
    for i in range(n):
        for j in range(i,n):
            b=np.zeros((n,n));b[i,j]=1.
            if i!=j:b[j,i]=1.
            b/=np.linalg.norm(b,'fro')
            basis.append(b);pairs.append((i,j))
    return np.asarray(basis),pairs


def lyapunov_null_screen(J, sample_stride=8):
    """Necessary screen for constant symmetric G with J^T G+G J=0.

    Does NOT test a state-dependent metric; DOES NOT rule out physical energy.
    """
    n=J.shape[-1]; B,pairs=symmetry_basis(n)
    blocks=[]
    for mat in J[::sample_stride]:
        images=np.einsum('ji,bjk->bik',mat,B)+np.einsum('bij,jk->bik',B,mat)
        blocks.append(images.reshape(len(B),-1).T)
    M=np.concatenate(blocks,axis=0)
    U,s,Vh=np.linalg.svd(M,full_matrices=False)
    G=np.einsum('b,bij->ij',Vh[-1],B)
    G/=np.linalg.norm(G,'fro')
    r=np.einsum('nji,jk->nik',J,G)+np.einsum('ij,njk->nik',G,J)
    rel=np.linalg.norm(r.reshape(len(J),-1),axis=1)/np.maximum(1e-30,2*np.linalg.norm(J,axis=(1,2))*np.linalg.norm(G,'fro'))
    evals=np.linalg.eigvalsh(G)
    return dict(min_singular=float(s[-1]),max_singular=float(s[0]),relative_min_singular=float(s[-1]/max(s[0],1e-30)),
                candidate_eigen_min=float(evals[0]),candidate_eigen_max=float(evals[-1]),
                candidate_definite=bool(np.all(evals>0) or np.all(evals<0)),
                candidate_residual_relative_max=float(np.max(rel)),candidate_residual_relative_median=float(np.median(rel)),
                tested_nodes=len(J[::sample_stride]),n_symmetric_unknowns=len(B),
                interpretation='necessary_screen_only; neither coordinate-intrinsic nor a search for state-dependent energy metric')


def affine_invariant_screen(J,sample_stride=8):
    """For linear first integral c^Ty, c^T J=0 necessarily; NOT sufficient."""
    M=np.concatenate([z.T for z in J[::sample_stride]],axis=0)
    s=np.linalg.svd(M,full_matrices=False,compute_uv=False)
    return dict(singular_min=float(s[-1]),singular_max=float(s[0]),
                min_to_max_ratio=float(s[-1]/max(s[0],1e-30)),
                interpretation='Necessary for constant affine first integral only; cannot exclude nonlinear constraints')


def unit_change_check(P,E,K,T,Gdiag):
    """Check coordinate tensor transformation of a metric; not Euclidean invariance."""
    n=P.shape[-1]; A=np.diag(Gdiag); Tmat=np.diag(T)
    Wprime=np.diag(Gdiag/(T*T))
    source=np.einsum('ni,ij,nj->n',E,A,K)
    dest=np.einsum('ni,ij,nj->n',E@Tmat,Wprime,K@Tmat)
    return float(np.max(np.abs(source-dest))/max(np.max(np.abs(source)),1e-300))
