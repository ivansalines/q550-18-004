"""Cancellation-resistant exact-increment ODE using Jacobian line integrals.

Fundamental identity F(y+d)-F(y) = [integral_0^1 DF(y+t d)dt] d.
The identity is exact for smooth F; Gauss-Legendre numerical quadrature is an
approximation, and the Jacobian must be verified against the original RHS.
The method transports material differences, not two independently large states.
"""
import numpy as np
from numpy.polynomial.legendre import leggauss

def line_increment(jac,y,d,order=3):
    y=np.asarray(y,float);d=np.asarray(d,float)
    t,w=leggauss(int(order));t=(t+1.)*.5;w=w*.5
    return sum(wi*(np.asarray(jac(y+ti*d),float)@d) for ti,wi in zip(t,w))

def compensated_rhs(rhs,jac,eps,order=3):
    """ODE state [base11, delta_plus10, delta_minus10], same unperturbed clock."""
    def fun(t,state):
        y=state[:11];dp=np.r_[state[11:21],0.];dm=np.r_[state[21:31],0.]
        return np.r_[rhs(t,y),line_increment(jac,y,dp,order)[:10],
                        line_increment(jac,y,dm,order)[:10]]
    return fun

def direct_second_from_jac(jac,y,u,h,order=4):
    """D(DF)[u]u estimated from the independently pinned Jacobian.

    h is an absolute displacement along unit material tangent; outputs include
    tangent norm squared. Four-point derivative suppresses O(h^2) truncation.
    """
    y=np.asarray(y,float);u=np.asarray(u,float)
    norm=np.linalg.norm(u[:10]);
    if norm==0:return np.zeros_like(y)
    v=u/norm; h=float(h)
    def g(s):return np.asarray(jac(y+s*v),float)@v
    if order==2:return norm**2*(g(h)-g(-h))/(2*h)
    return norm**2*(g(-2*h)-8*g(-h)+8*g(h)-g(2*h))/(12*h)

def jacobian_variational_rhs(rhs,jac,h):
    """Independent 31D second variation using first derivative of pinned J."""
    def fun(t,z):
        y=z[:11]; u=np.r_[z[11:21],0.];w=np.r_[z[21:31],0.]
        J=jac(y)
        return np.r_[rhs(t,y),(J@u)[:10],(J@w+direct_second_from_jac(jac,y,u,h))[:10]]
    return fun

def even_odd_from_deltas(z,eps,H):
    plus=z[:,11:21];minus=z[:,21:31]
    return ((plus+minus)@H.T)/(2*eps),((plus-minus)@H.T)/(2*eps)

def rms(x):return float(np.sqrt(np.mean(np.asarray(x,float)**2)))
