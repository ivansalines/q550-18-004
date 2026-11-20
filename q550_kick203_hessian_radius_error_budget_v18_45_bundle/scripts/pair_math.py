"""Reduced observer matching math. Off-constraint, no constitutive memory inference."""
import numpy as np

def orth_rows(columns,tol=1e-9):
    x=np.asarray(columns,float)
    u,s,vt=np.linalg.svd(x,full_matrices=False)
    if s[-1] < tol*s[0]:raise RuntimeError('Degenerate material observer basis')
    return u[:,:x.shape[1]].T

def null_basis(h,rtol=2e-11):
    h=np.asarray(h,float)
    u,s,vt=np.linalg.svd(h,full_matrices=True)
    rank=int(np.count_nonzero(s>max(s[0]*rtol,1e-12)))
    return vt[rank:].T,rank

def chosen_seed(h,jac,method='jet',second=False):
    """Exactly Hh=0; jet has H A h=0 too, to linear accuracy."""
    n,rank=null_basis(h)
    if rank != h.shape[0] or n.shape[1]<2:raise RuntimeError('Observer lacks hidden degrees')
    downstream=h@jac@n
    u,s,vt=np.linalg.svd(downstream,full_matrices=True)
    if method=='jet':
        rk=int(np.count_nonzero(s>max(s[0]*1e-10 if len(s) else 0,1e-11)))
        null=vt[rk:].T
        if null.shape[1]<1:raise RuntimeError('No nontrivial first-jet matching direction')
        v=n@null[:,1 if second and null.shape[1]>1 else 0]
    elif method=='state_only':
        v=n@vt[0]
    else:raise ValueError(method)
    v=v/np.linalg.norm(v)
    k=int(np.argmax(abs(v)))
    if v[k]<0:v=-v
    return v,{'observer_rank':int(rank),'hidden_dim':int(n.shape[1]),
              'jac_hidden_singular_values':s.tolist(),
              'state_match_defect':float(np.linalg.norm(h@v)),
              'jet_match_defect':float(np.linalg.norm(h@jac@v))}

def diagnostics(q_plus,q_minus,h,eps):
    dp=np.asarray(q_plus)-np.asarray(q_minus)
    return float(np.linalg.norm(dp)),float(np.linalg.norm(h@dp)/(2*eps))
