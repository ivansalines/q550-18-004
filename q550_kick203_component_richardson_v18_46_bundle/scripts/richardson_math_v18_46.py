"""Component-aware Richardson diagnostics, NOT high-precision certification.
The original mixed RHS and directional Jacobian stencils are fourth-order
under smoothness and exact arithmetic; empirical order can fail with rounding.
All 10 components are kept. Component 0,2,8 are *diagnostic priorities*,
not physically privileged axes. No dynamic fitting of the model is performed.
"""
import numpy as np
RHS_H=(.001,.0015,.002,.003,.004,.006,.008,.012,.016,.024,.032,.048)
JAC_H=(.000125,.00025,.0005,.001,.002,.004,.008,.016)
TRIPLES_RHS=((.004,.002,.001),(.006,.003,.0015),(.008,.004,.002),(.012,.006,.003),(.016,.008,.004),(.024,.012,.006),(.032,.016,.008),(.048,.024,.012))
TRIPLES_JAC=((.0005,.00025,.000125),(.001,.0005,.00025),(.002,.001,.0005),(.004,.002,.001),(.008,.004,.002),(.016,.008,.004))
KEY_COMPONENTS=(0,2,8)
def relative(a,b):
    x=np.asarray(a,float);y=np.asarray(b,float)
    return float(np.linalg.norm(x-y)/max(float(np.linalg.norm(x)),float(np.linalg.norm(y)),1e-30))
def richardson_pair(coarse,fine,order=4):
    if order<=0:raise ValueError('Extrapolation order must be positive')
    return np.asarray(fine,float)+(np.asarray(fine,float)-np.asarray(coarse,float))/(2**order-1)
def triple_stats(v0,v1,v2,*,floor=5e-12):
    """Input coarse, medium, fine in ratio 4:2:1. Empirical order from delta norms.
    Flags are diagnostics, never mathematical error bars or proofs of convergence.
    """
    a=np.asarray(v0,float);b=np.asarray(v1,float);c=np.asarray(v2,float)
    d01=a-b;d12=b-c
    n01=float(np.linalg.norm(d01));n12=float(np.linalg.norm(d12))
    scale=max(float(np.linalg.norm(a)),float(np.linalg.norm(b)),float(np.linalg.norm(c)),1e-30)
    # Abs+relative guard prevents estimating an order from tiny cancellation noise.
    resolved=bool(n01>floor*scale and n12>floor*scale)
    p=float(np.log2(n01/n12)) if resolved else None
    high=richardson_pair(a,b);low=richardson_pair(b,c)
    cos=float((d01@d12)/max(n01*n12,1e-300))
    return dict(order=p,resolved=resolved,cos_differences=cos,
       order4_consistent=bool(resolved and abs(p-4)<1.0 and cos>0.8),
       richardson_between_pairs=relative(high,low),
       raw_neighbor_disagreement=relative(b,c),
       high=high,low=low,norm_delta_coarse=n01,norm_delta_fine=n12)
def diagnostic_references(j):
    """Three overlapping Jacobian internal references; none represents ground truth."""
    return {
       'small':np.median(np.stack([j[h] for h in (.00025,.0005,.001)]),axis=0),
       'parent':np.median(np.stack([j[h] for h in (.0005,.001,.002)]),axis=0),
       'large':np.median(np.stack([j[h] for h in (.001,.002,.004)]),axis=0),
    }
def error_shares(x):
    v=np.asarray(x,float)**2
    return v/max(float(np.sum(v)),1e-60)
def vec_fields(vec,prefix):return {f'{prefix}{k}':float(x) for k,x in enumerate(vec)}
