"""Synthetic regression suite. These are mathematical tests, NOT Q550 runs."""
import sys,unittest,math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from kernel_compensation_math_v18_47 import (mixed_compensated,polarization_rhs,accel_gradient_identity,relative,shares,vector_columns)
from dual_hessian_jet_math_v18_44 import mixed_rhs_hessian_contraction

class PolynomialModel:
    def __init__(self):
        rng=np.random.default_rng(1847)
        self.scale=np.exp(np.linspace(-.5,.7,10))
        self.H=rng.normal(0,.12,(10,11,11));self.H=.5*(self.H+np.swapaxes(self.H,1,2))
        self.L=rng.normal(0,.09,(10,11));self.b=np.linspace(.7,1.5,10)
        self.y=np.r_[np.linspace(-.5,.7,10)*self.scale,.3]
        self.g=rng.normal(size=10)
    def f(self,_,y):
        z=np.r_[np.asarray(y[:10],float)/self.scale,float(y[10])]
        vals=self.b+self.L@z+.5*np.einsum('ijk,j,k->i',self.H,z,z)
        return np.r_[self.scale*vals,.23]
    def jac(self,y):
        z=np.r_[np.asarray(y[:10],float)/self.scale,float(y[10])]
        B=self.L+np.einsum('ijk,k->ij',self.H,z)
        ans=np.zeros((11,11));ans[:10,:]=self.scale[:,None]*B/np.r_[self.scale,1.][None,:]
        return ans
    def truth(self):
        f=self.f(0,self.y);v=np.r_[f[:10]/self.scale,f[10]]
        return np.einsum('i,ijk,k->j',self.g,self.H,v)[:10]

class KernelTests(unittest.TestCase):
    def setUp(self):self.s=PolynomialModel();self.fy=self.s.f(0,self.s.y)
    def check(self,v,tol=1e-8):self.assertLess(relative(v,self.s.truth()),tol)
    def test_mixed_compensated(self):self.check(mixed_compensated(self.s.y,self.fy,self.s.f,self.s.scale,self.s.g,.008))
    def test_mixed_reverse(self):self.check(mixed_compensated(self.s.y,self.fy,self.s.f,self.s.scale,self.s.g,.008,True))
    def test_polarization(self):self.check(polarization_rhs(self.s.y,self.fy,self.s.f,self.s.scale,self.s.g,.008))
    def test_acceleration_identity(self):self.check(accel_gradient_identity(self.s.y,self.fy,self.s.f,self.s.jac,self.s.scale,self.s.g,.002))
    def test_historic_mixed_agrees(self):self.check(mixed_rhs_hessian_contraction(self.s.y,self.fy,self.s.f,self.s.scale,self.s.g,.008))
    def test_accel_and_polarization_not_same_kernel(self):
        import inspect
        self.assertNotEqual(inspect.getsource(accel_gradient_identity),inspect.getsource(polarization_rhs))
    def test_three_radii(self):
        for h in (.004,.008,.012):
            self.check(mixed_compensated(self.s.y,self.fy,self.s.f,self.s.scale,self.s.g,h))
            self.check(polarization_rhs(self.s.y,self.fy,self.s.f,self.s.scale,self.s.g,h))
    def test_accel_radii(self):
        for h in (.001,.002,.004):self.check(accel_gradient_identity(self.s.y,self.fy,self.s.f,self.s.jac,self.s.scale,self.s.g,h))
    def test_zero_hessian(self):
        s=self.s;s.H[:]=0
        for fun in (mixed_compensated,polarization_rhs):
            self.assertLess(np.linalg.norm(fun(s.y,s.f(0,s.y),s.f,s.scale,s.g,.008)),2e-9)
        self.assertLess(np.linalg.norm(accel_gradient_identity(s.y,s.f(0,s.y),s.f,s.jac,s.scale,s.g,.002)),2e-9)
    def test_basis_columns(self):self.assertEqual(set(vector_columns('v_',np.arange(10))),{f'v_{j}' for j in range(10)})
    def test_component_shares(self):
        z=shares(np.array([3.,4.]+[0.]*8));self.assertAlmostEqual(z[0],.36);self.assertAlmostEqual(z[1],.64)
    def test_reference_not_ground_truth(self):
        # Agreement of identically biased methods is not proof of accuracy.
        x=self.s.truth();bias=np.ones(10)*.5
        self.assertEqual(relative(x+bias,x+bias),0.)
        self.assertGreater(relative(x+bias,x),.01)
    def test_no_zero_step(self):
        with self.assertRaises(ZeroDivisionError):mixed_compensated(self.s.y,self.fy,self.s.f,self.s.scale,self.s.g,0.)

if __name__=='__main__':unittest.main()
