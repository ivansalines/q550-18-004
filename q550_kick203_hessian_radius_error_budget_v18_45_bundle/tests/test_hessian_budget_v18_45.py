"""v18.45 algebra tests independent of ORIGINAL Q550 server integration."""
import unittest,sys
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'scripts'))
from hessian_budget_math_v18_45 import *
from dual_hessian_jet_math_v18_44 import mixed_rhs_hessian_contraction,dot_B_transpose_g_from_jacobian
class HessianBudgetTests(unittest.TestCase):
    def test_grid_length(self):
        self.assertEqual(len(RHS_H),11);self.assertEqual(len(JAC_H),6)
    def test_anchored_parent_step(self):
        self.assertIn(.006,RHS_H);self.assertIn(.002,JAC_H)
    def test_positive_radii(self):
        self.assertTrue(all(0<h<.2 for h in RHS_H+JAC_H))
    def test_reference_median(self):
        self.assertTrue(np.array_equal(jac_reference({.0005:np.array([1.,2.]),.001:np.array([3.,4.]),.002:np.array([9.,6.])}),np.array([3.,4.])))
    def test_relative_zero(self):
        self.assertEqual(relative([0,0],[0,0]),0.)
    def test_relative_scale(self):
        self.assertAlmostEqual(relative(np.array([1.]),np.array([2.])),.5)
    def test_neighbor_instability(self):
        vs=[np.array([1.,0.]),np.array([2.,0.]),np.array([3.,0.])]
        self.assertAlmostEqual(neighbor_instability(vs,0),.5)
        self.assertAlmostEqual(neighbor_instability(vs,1),.5)
    def test_choose_descriptive_minimum(self):
        i,e=select_radius([.1,.2,.3],[np.array([3.]),np.array([1.]),np.array([2.])],np.array([1.]))
        self.assertEqual(i,1);self.assertEqual(e[i],0)
    def test_component_shares(self):
        d,s=component_budget(np.zeros(2),np.array([3.,4.]))
        self.assertAlmostEqual(sum(s),1);self.assertAlmostEqual(s[0],9/25)
    def test_signed_difference(self):
        d,_=component_budget(np.array([5.,2.]),np.array([3.,8.]))
        self.assertTrue(np.array_equal(d,[-2,6]))
    def test_richardson_constant(self):
        a=forecast_richardson(np.array([3.,4.]),np.array([3.,4.]))
        self.assertTrue(np.array_equal(a,[3.,4.]))
    def test_power_synthetic_quadratic(self):
        self.assertAlmostEqual(observed_power(.25,1.,2.),2.)
    def test_nonmonotone_probe_rejected(self):
        with self.assertRaises(ValueError):observed_power(-1.,1.,2.)
    def test_vector_columns(self):
        self.assertEqual(vec_cols([.1,.2],'p'),{'p0':.1,'p1':.2})
    def test_independent_quadratic_model_hessian(self):
        def f(t,y):
            z=np.zeros(11);z[0]=.5*y[0]**2+y[10]*y[1];z[1]=y[0]*y[1];z[10]=1.
            return z
        def jac(y):
            J=np.zeros((11,11));J[0,0]=y[0];J[0,1]=y[10];J[0,10]=y[1]
            J[1,0]=y[1];J[1,1]=y[0];return J
        y=np.zeros(11);y[0]=.4;y[1]=.2;y[10]=.3
        speed=f(0,y);g=np.array([1.,.3,0.,0.,0.,0.,0.,0.,0.,0.]);scale=np.ones(10)
        expected=np.zeros(10);expected[0]=speed[0]+.3*speed[1];expected[1]=1.+.3*speed[0]
        a=mixed_rhs_hessian_contraction(y,speed,f,scale,g,.01)
        b=dot_B_transpose_g_from_jacobian(y,speed,jac,scale,g,.001)
        self.assertTrue(np.allclose(a,expected,rtol=1e-8,atol=2e-9),(a,expected))
        self.assertTrue(np.allclose(b,expected,rtol=1e-9,atol=1e-11),(b,expected))
    def test_cancellation_unequal_probes(self):
        vs=[np.array([.01]),np.array([.1]),np.array([1.])]
        self.assertGreater(neighbor_instability(vs,0),0)
if __name__=='__main__':unittest.main()
