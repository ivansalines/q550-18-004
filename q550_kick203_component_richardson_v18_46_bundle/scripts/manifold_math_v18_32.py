"""Phase-covariant scalar hypersurface diagnostics for Q550.

The tested condition is L_F C = mu(theta)*C, not L_F C=0.
A numerical candidate is NOT a certified physical constraint. The dictionary
contains material-dependent terms only, excluding clock-only solutions.
"""
import numpy as np
from itertools import combinations_with_replacement
PAIRS=list(combinations_with_replacement(range(10),2))

def features(y,center,scale,theta0,degree=2,phase=True):
    y=np.asarray(y,dtype=float); x=(y[:10]-center)/scale
    a=y[10]-theta0; s=np.sin(a); c=np.cos(a)
    out=[x]
    if degree==2:out.append(np.array([x[i]*x[j] for i,j in PAIRS]))
    if phase:out += [x*s,x*c]
    return np.concatenate(out)

def features_flow(y,f,center,scale,theta0,degree=2,phase=True):
    y=np.asarray(y,dtype=float); f=np.asarray(f,float)
    x=(y[:10]-center)/scale; v=f[:10]/scale
    a=y[10]-theta0; s=np.sin(a); c=np.cos(a); rate=f[10]
    out=[v]
    if degree==2:out.append(np.array([v[i]*x[j]+x[i]*v[j] for i,j in PAIRS]))
    if phase:out += [v*s+x*c*rate,v*c-x*s*rate]
    return np.concatenate(out)

def features_material_gradient(y,center,scale,theta0,degree=2,phase=True):
    y=np.asarray(y,float);x=(y[:10]-center)/scale
    a=y[10]-theta0; s=np.sin(a);c=np.cos(a)
    out=np.zeros((len(features(y,center,scale,theta0,degree,phase)),10),float)
    k=0
    out[k:k+10]=np.eye(10);k+=10
    if degree==2:
        for i,j in PAIRS:
            out[k,i]+=x[j];out[k,j]+=x[i];k+=1
    if phase:
        out[k:k+10]=s*np.eye(10);k+=10
        out[k:k+10]=c*np.eye(10)
    return out/scale

def cofactor(y,theta0,lam,mod=0.):
    return lam+mod*np.sin(y[10]-theta0)

def score_residual(y,f,q,center,scale,theta0,lam,mod,degree,phase):
    phi=features(y,center,scale,theta0,degree,phase)
    dphi=features_flow(y,f,center,scale,theta0,degree,phase)
    return float((dphi-cofactor(y,theta0,lam,mod)*phi)@q)

def scaled_fit(states,fields,center,scale,theta0,lam,mod,degree,phase,anchors=None):
    states=np.asarray(states,float);fields=np.asarray(fields,float)
    ph=np.array([features(y,center,scale,theta0,degree,phase) for y in states])
    der=np.array([features_flow(y,f,center,scale,theta0,degree,phase) for y,f in zip(states,fields)])
    co=np.array([cofactor(y,theta0,lam,mod) for y in states])
    A=der-co[:,None]*ph
    # Column whitening and explicit anchor penalty; the intercept is excluded.
    spreads=np.maximum(np.sqrt(np.mean(ph*ph,axis=0)),1e-12)
    rr=A/spreads
    if anchors is not None:
        pa=np.array([features(y,center,scale,theta0,degree,phase) for y in anchors])
        # 'zero on the central trajectory' is not sufficient for invariance.
        rr=np.vstack([rr,0.2*np.sqrt(len(states)/len(pa))*pa/spreads])
    _,ss,vh=np.linalg.svd(rr,full_matrices=False)
    q=vh[-1]/spreads
    observed=float(np.sqrt(np.mean((ph@q)**2)))
    if not np.isfinite(observed) or observed<1e-14:raise RuntimeError('Degenerate manifold model')
    q=q/observed
    return q, dict(singular_min=float(ss[-1]),singular_max=float(ss[0]),relative_min=float(ss[-1]/ss[0]),feature_count=len(q))

def ensemble_score(states,fields,q,center,scale,theta0,lam,mod,degree,phase):
    states=np.asarray(states,float);fields=np.asarray(fields,float)
    cf=[];df=[];rr=[]
    for y,f in zip(states,fields):
        c=features(y,center,scale,theta0,degree,phase)@q
        d=features_flow(y,f,center,scale,theta0,degree,phase)@q
        cf.append(c);df.append(d);rr.append(d-cofactor(y,theta0,lam,mod)*c)
    cf=np.array(cf);df=np.array(df);rr=np.array(rr)
    typical=max(np.sqrt(np.mean(df**2)),np.sqrt(np.mean((rr-df)**2)),1e-20)
    return dict(residual_rms=float(np.sqrt(np.mean(rr**2))),relative_residual=float(np.sqrt(np.mean(rr**2))/typical),
                field_lie_rms=float(np.sqrt(np.mean(df**2))),field_candidate_rms=float(np.sqrt(np.mean(cf**2))),
                candidate_std=float(np.std(cf)),max_abs_residual=float(np.max(np.abs(rr))))

def project_to_level(y,q,center,scale,theta0,degree,phase,max_scaled_step=3.):
    """Project using a declared Euclidean metric in normalized material coordinates.

    This metric is a diagnostic choice and has no privileged material meaning.
    """
    z=np.asarray(y,float).copy()
    s=np.asarray(scale,float); step_norm=0.
    for _ in range(12):
        c=float(features(z,center,scale,theta0,degree,phase)@q)
        if abs(c)<1e-10:return z,step_norm,True
        grad=features_material_gradient(z,center,scale,theta0,degree,phase).T@q
        gx=grad*s; gnorm=float(np.linalg.norm(gx))
        if gnorm<1e-12:return z,step_norm,False
        dx=-c*gx/(gnorm*gnorm)
        dnorm=float(np.linalg.norm(dx));step_norm+=dnorm
        if step_norm>max_scaled_step:return z,step_norm,False
        z[:10]+=dx*s
    return z,step_norm,abs(float(features(z,center,scale,theta0,degree,phase)@q))<1e-8

def surface_test(states,f,q,center,scale,theta0,lam,mod,degree,phase):
    vals=[]; projected=[]; failed=0
    for y in states:
        z,shift,ok=project_to_level(y,q,center,scale,theta0,degree,phase)
        if not ok:failed+=1;continue
        try:
            field=np.asarray(f(0,z),float)
            if field.shape!=(11,) or not np.all(np.isfinite(field)):
                raise ValueError('Invalid projected Q550 field')
            d=features_flow(z,field,center,scale,theta0,degree,phase)@q
        except (ValueError,FloatingPointError,OverflowError,ZeroDivisionError):
            failed+=1
            continue
        vals.append(dict(surface_residual=float(abs(d)),scaled_projection_length=shift,
                         initial_C=float(features(z,center,scale,theta0,degree,phase)@q)))
        projected.append(z)
    return vals,projected,failed
