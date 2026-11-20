"""Independent RHS directional Hessian and second-order constraint geometry.

The fourth-order five-point formula differentiates the ORIGINAL vector field F,
not the Jacobian DF. These two methods share the physical ODE and arithmetic;
therefore agreement is numerical cross-validation, not independent physics.

Second-order constraint tangency for C(y)=constant:
 D C(y) u = 0,  D C(y) w + D2 C(y)[u,u] = 0.
A correction w0=-DC_material^+ D2C[u,u] gives a locally feasible initial jet.
No physical invariant is assumed if the external plugin is missing.
"""
import numpy as np

def rhs_hessian(f,t,y,u,h):
    """Fourth-order central 5-point second directional derivative along unit u."""
    y=np.asarray(y,float);u=np.asarray(u,float)
    norm=np.linalg.norm(u)
    if norm==0:return np.zeros_like(y)
    v=u/norm; h=float(h)
    if not np.isfinite(h) or h<=0:raise ValueError('h must be positive')
    return norm**2*(-f(t,y+2*h*v)+16*f(t,y+h*v)-30*f(t,y)+16*f(t,y-h*v)-f(t,y-2*h*v))/(12*h*h)

def jac_hessian(jac,y,u,h):
    """Fourth-order 4-point derivative of pinned DF, independent of RHS stencil."""
    y=np.asarray(y,float);u=np.asarray(u,float);n=np.linalg.norm(u)
    if n==0:return np.zeros_like(y)
    v=u/n;h=float(h)
    def g(s):return jac(y+s*v)@v
    return n*n*(g(-2*h)-8*g(-h)+8*g(h)-g(2*h))/(12*h)

def finite_jac_c(invariants,y,step=2e-5):
    y=np.asarray(y,float)
    val=np.atleast_1d(invariants(y)).astype(float)
    out=np.empty((len(val),len(y)))
    for j in range(len(y)):
        e=np.zeros(len(y));e[j]=step*max(1.,abs(y[j]))
        out[:,j]=(np.atleast_1d(invariants(y+e))-np.atleast_1d(invariants(y-e)))/(2*e[j])
    return out

def finite_c_hessian(invariants,y,u,h=.003):
    # Use the same 5-point fourth-order central stencil as RHS, but on C.
    fun=lambda _,v:np.atleast_1d(invariants(v)).astype(float)
    return rhs_hessian(fun,0.,y,u,h)

def constraint_initial_jet(invariants,y,v,h=.003):
    """Restrict the seed to material tangent, return second-order normal correction.

    This does NOT prove the constraint is invariant under the Q550 flow.
    """
    y=np.asarray(y,float);v=np.asarray(v,float)
    G=finite_jac_c(invariants,y)
    Gm=G[:,:10]
    if np.linalg.svd(Gm,compute_uv=False)[0]<1e-14:
        raise ValueError('Constraint derivative has no material rank')
    pm=np.eye(10)-np.linalg.pinv(Gm,rcond=1e-11)@Gm
    uv=pm@v[:10]
    if np.linalg.norm(uv)<1e-8:raise ValueError('Seed tangent projection is nearly zero')
    uv=uv/np.linalg.norm(uv)
    u=np.r_[uv,0.]
    d2=finite_c_hessian(invariants,y,u,h)
    w0=np.r_[-np.linalg.pinv(Gm,rcond=1e-11)@d2,0.]
    return u,w0,dict(tangent_defect=float(np.linalg.norm(G@u)),
                       second_tangent_defect=float(np.linalg.norm(G@w0+d2)),
                       normal_curvature_norm=float(np.linalg.norm(w0)),
                       constraint_rank=int(np.linalg.matrix_rank(Gm)))

def second_variational_rhs(f,jac,h,w0=None):
    def fun(t,z):
        y=z[:11];u=np.r_[z[11:21],0.];w=np.r_[z[21:31],0.]
        J=jac(y)
        src=rhs_hessian(f,t,y,u,h)
        return np.r_[f(t,y),(J@u)[:10],(J@w+src)[:10]]
    return fun

def rms(x):return float(np.sqrt(np.mean(np.asarray(x,float)**2)))
