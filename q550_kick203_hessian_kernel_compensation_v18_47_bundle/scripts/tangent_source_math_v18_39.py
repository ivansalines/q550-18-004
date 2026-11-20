"""Full 10D material residual source projected into a declared moving 9D gauge.

The normalized material Euclidean metric is a DIAGNOSTIC convention, not a
certified physical material metric or an intrinsic torsion measurement.
"""
import numpy as np
from residual_gradient_math_v18_37 import unit

def normal_frame(n):
    """Deterministic 10x9 orthonormal frame for normal n (Householder)."""
    n=unit(n)
    if n.shape!=(10,):raise ValueError('Expected ten-dimensional material normal')
    e=np.zeros(10);e[0]=1.
    v=e-n
    vnorm=np.linalg.norm(v)
    Q=np.eye(10) if vnorm<1.e-13 else np.eye(10)-2.*np.outer(v/vnorm,v/vnorm)
    basis=Q[:,1:]
    if np.linalg.norm(basis.T@n)>2.e-13:raise RuntimeError('Invalid tangent basis')
    return basis

def minimal_rotation(a,b):
    """Orthogonal minimal-plane rotation carrying unit a to b; no guessed axis."""
    a=unit(a);b=unit(b)
    if a.shape!=b.shape:raise ValueError('Vector size mismatch')
    c=float(np.clip(a@b,-1.,1.))
    if c<=-1.+1.e-10:raise ValueError('Antiparallel normal requires an unresolved transport gauge')
    K=np.outer(b,a)-np.outer(a,b)
    return np.eye(len(a))+K+K@K/(1.+c)

def common_vector(v,own_normal,base_normal,base_frame,divisor=1.):
    """Rotate each own tangent vector minimally to the SAME base normal and basis."""
    v=np.asarray(v,float);own_normal=unit(own_normal);base_normal=unit(base_normal)
    p=v-own_normal*float(own_normal@v)
    R=minimal_rotation(own_normal,base_normal)
    coords=np.asarray(base_frame).T@(R@p)/divisor
    return coords

def tilt_vector(hom,direct):
    """Stable signed angular displacement VECTOR in tangent plane of hom, radians.

    Preserves gauge orientation at angles much smaller than arccos resolution.
    """
    a=unit(hom);b=unit(direct)
    if float(a@b)<0:b=-b
    parallel=float(a@b);p=b-parallel*a
    sinval=float(np.linalg.norm(p));ang=float(np.arctan2(sinval,abs(parallel)))
    return np.zeros_like(a) if sinval==0 else p*(ang/sinval)

def vector_holdout(rows,field,center_key='alpha'):
    """Held-out odd vector response: inner +/-1/3 fit; outer +/-1 test.

    Returns both magnitude error and full-vector turn/gain. Never silently
    treats near-zero odd directions as established rotations.
    """
    by={round(float(r[center_key]),10):np.asarray(r[field],float) for r in rows}
    if set(by)!={-1.,-round(1/3,10),0.,round(1/3,10),1.}:raise ValueError('Incomplete symmetric five-amplitude family')
    h=round(1/3,10);odd_inner=(by[h]-by[-h])/(2./3.)
    odd_outer=(by[1.]-by[-1.])/2.
    center=by[0.];delta=np.r_[by[1.]-center,by[-1.]-center]
    expect=np.r_[odd_inner,-odd_inner]
    den=max(float(np.linalg.norm(delta)),float(np.linalg.norm(expect)),1.e-16)
    inn=float(np.linalg.norm(odd_inner));out=float(np.linalg.norm(odd_outer))
    quality=inn>1e-13 and out>1e-13
    cosine=float(np.clip(odd_inner@odd_outer/(inn*out),-1,1)) if quality else None
    return dict(holdout_relative_error=float(np.linalg.norm(delta-expect)/den),
        inner_odd_norm=inn,outer_odd_norm=out,outer_odd_gain=out/inn if quality else None,
        inner_outer_odd_cosine=cosine,inner_outer_odd_turn_deg=float(np.degrees(np.arccos(cosine))) if quality else None,
        inner_even_norm=float(np.linalg.norm((by[h]+by[-h]-2*center)/(2./9.))),
        outer_even_norm=float(np.linalg.norm((by[1.]+by[-1.]-2*center)/2.)))

def source_level_coherence(levels,sources):
    """Five-amplitude centered scalar/vector association; no universal-law claim."""
    x=np.asarray(levels,float);v=np.asarray(sources,float)
    xc=x-np.mean(x);vc=v-np.mean(v,axis=0)
    den=float(np.linalg.norm(xc)*np.linalg.norm(vc))
    return None if den<1.e-20 else float(np.linalg.norm(xc@vc)/den)

def rotation_angle(a,b,floor=1.e-13):
    a=np.asarray(a,float);b=np.asarray(b,float)
    if np.linalg.norm(a)<=floor or np.linalg.norm(b)<=floor:return None
    cosine=float(np.clip((a@b)/(np.linalg.norm(a)*np.linalg.norm(b)),-1,1))
    return float(np.degrees(np.arctan2(np.linalg.norm(b-(b@a)/(a@a)*a),float((b@a)/np.linalg.norm(a)))))
