"""SYNTHETIC SMOKE TEST ONLY. Replace with authenticated Q550 RHS decomposition."""
import numpy as np
A=np.diag(np.linspace(.3,1.3,11)); B=np.arange(1,12)/20.
def linear(x):return A@x
def quadratic(x):return B*x*x
def rhs(x):return linear(x)+quadratic(x)
components={'linear':linear,'quadratic':quadratic}
def analytic_bilinear(x,u,v):return 2*B*u*v
def analytic_component_bilinears(x,u,v):return {'linear':np.zeros(11),'quadratic':2*B*u*v}
