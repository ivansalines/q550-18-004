"""Independent finite-directional variational operators in the full 11D Q550 ODE.

The Hessian here is NOT inferred from nonlinear +/- trajectories. Both Df and
D2f are evaluated from the pinned nonlinear RHS at finite *state* displacements
far above the epsilon-scale cancellation. This is a numerical variational
construction, NOT analytic automatic differentiation or an interval proof.
"""
from __future__ import annotations
import numpy as np

def directional_first(rhs, x, v, absolute_step=0.004, order=4):
    x=np.asarray(x,float); v=np.asarray(v,float)
    norm=float(np.linalg.norm(v[:10]))
    if norm==0:return np.zeros_like(x)
    direction=v/norm; h=float(absolute_step)
    fp=rhs(0.,x+h*direction);fm=rhs(0.,x-h*direction)
    if order==2:return norm*(fp-fm)/(2*h)
    fp2=rhs(0.,x+2*h*direction);fm2=rhs(0.,x-2*h*direction)
    return norm*(-fp2+8*fp-8*fm+fm2)/(12*h)

def directional_second(rhs, x, u, absolute_step=0.08, order=4):
    x=np.asarray(x,float);u=np.asarray(u,float)
    norm=float(np.linalg.norm(u[:10]))
    if norm==0:return np.zeros_like(x)
    direction=u/norm;h=float(absolute_step)
    f0=rhs(0.,x)
    fp=rhs(0.,x+h*direction);fm=rhs(0.,x-h*direction)
    if order==2:return norm**2*(fp-2*f0+fm)/h**2
    fp2=rhs(0.,x+2*h*direction);fm2=rhs(0.,x-2*h*direction)
    return norm**2*(-fp2+16*fp-30*f0+16*fm-fm2)/(12*h*h)

def second_variational_rhs(original_rhs, hessian_step=.08, first_step=.004,
                           hessian_order=4):
    """Autonomous 31-vector: base 11D + 10D material u + 10D material w.

    Fixed phase perturbation u_phase=w_phase=0 because phase is the identical
    clock for both nonlinear seeds. Variational equations include the full
    base clock when differentiating the original RHS.
    """
    def fun(t,z):
        y=np.asarray(z[:11]);u=np.r_[z[11:21],0.];w=np.r_[z[21:31],0.]
        du=directional_first(original_rhs,y,u,first_step)[:10]
        dw=(directional_first(original_rhs,y,w,first_step)
            +directional_second(original_rhs,y,u,hessian_step,hessian_order))[:10]
        return np.r_[original_rhs(t,y),du,dw]
    return fun

def paired_even_observation(base,plus,minus,H,eps):
    """Return measured normalized even and odd on material state components."""
    pd=((np.asarray(plus)[:,:10]-np.asarray(base)[:,:10])/eps)@H.T
    md=((np.asarray(minus)[:,:10]-np.asarray(base)[:,:10])/eps)@H.T
    return .5*(pd+md),.5*(pd-md)

def rms(x):
    x=np.asarray(x,float);return float(np.sqrt(np.mean(x*x)))


def masked_ratio(numer,denom, floor=1e-30):return float(numer/max(denom,floor))
