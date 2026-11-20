import unittest,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from richardson_math_v18_46 import (
    RHS_H,JAC_H,TRIPLES_RHS,TRIPLES_JAC,KEY_COMPONENTS,
    richardson_pair,triple_stats,diagnostic_references,error_shares,relative)
class RichardsonAuditTests(unittest.TestCase):
    def test_distinct_radii(self):
        self.assertEqual(len(set(RHS_H)),12);self.assertEqual(len(set(JAC_H)),8)
    def test_rhs_ladders(self):
        for a,b,c in TRIPLES_RHS:
            self.assertIn(a,RHS_H);self.assertIn(b,RHS_H);self.assertIn(c,RHS_H)
            self.assertAlmostEqual(a/b,2.);self.assertAlmostEqual(b/c,2.)
    def test_jac_ladders(self):
        for a,b,c in TRIPLES_JAC:
            self.assertIn(a,JAC_H);self.assertIn(b,JAC_H);self.assertIn(c,JAC_H)
            self.assertAlmostEqual(a/b,2.);self.assertAlmostEqual(b/c,2.)
    def test_exact_fourth_order(self):
        exact=np.arange(1.,11.)/13.;a=np.arange(10.)*.01+1.
        h=.008
        X=[exact+a*x**4 for x in (4*h,2*h,h)]
        v=triple_stats(*X)
        self.assertTrue(v['order4_consistent']);self.assertAlmostEqual(v['order'],4.,places=7)
        self.assertLess(relative(v['low'],exact),1e-14)
        self.assertLess(v['richardson_between_pairs'],1e-14)
    def test_fourth_order_richardson_vector(self):
        a=np.arange(10.)*0.12
        true=np.arange(10.)*0.5
        out=richardson_pair(true+a*0.2**4,true+a*0.1**4)
        np.testing.assert_allclose(out,true,atol=1e-15)
    def test_not_fourth_order(self):
        exact=np.arange(10.,dtype=float)
        x=[exact+np.ones(10)*h*h for h in (.08,.04,.02)]
        v=triple_stats(*x)
        self.assertFalse(v['order4_consistent'])
        self.assertAlmostEqual(v['order'],2,places=7)
    def test_oscillating_stencil_is_rejected(self):
        v=triple_stats(np.ones(10),np.zeros(10),np.ones(10)*0.9)
        self.assertFalse(v['order4_consistent'])
        self.assertLess(v['cos_differences'],0.)
    def test_unresolved_identical_vectors(self):
        v=triple_stats(np.ones(10),np.ones(10),np.ones(10))
        self.assertFalse(v['resolved']);self.assertIsNone(v['order'])
    def test_small_differences_flagged(self):
        a=np.ones(10)
        v=triple_stats(a,a+1e-13,a+2e-13)
        self.assertFalse(v['resolved'])
    def test_large_cancellation_detected(self):
        v=triple_stats(np.ones(10)*100, np.ones(10)*100.1,np.ones(10)*99.99)
        self.assertFalse(v['order4_consistent'])
    def test_three_jac_references(self):
        j={h:np.ones(10)*(1.+h**4) for h in JAC_H}
        out=diagnostic_references(j)
        self.assertEqual(set(out),{'small','parent','large'})
        self.assertTrue(out['small'][0]<out['parent'][0]<out['large'][0])
    def test_error_share_sum(self):
        v=np.array([3,0,4,0,0,0,0,0,0,0],float)
        self.assertAlmostEqual(float(error_shares(v).sum()),1.)
        self.assertAlmostEqual(float(error_shares(v)[0]),9/25)
    def test_error_share_zero_is_zero(self):
        self.assertTrue(np.array_equal(error_shares(np.zeros(10)),np.zeros(10)))
    def test_protected_components(self):
        self.assertEqual(KEY_COMPONENTS,(0,2,8))
        self.assertEqual(len(set(KEY_COMPONENTS)),3)
    def test_richardson_not_ground_truth(self):
        # Both paths may share a constant bias: extrapolation cannot remove it.
        exact=np.zeros(10);bias=np.ones(10)*3
        v=triple_stats(*(exact+bias+np.ones(10)*h**4 for h in (.08,.04,.02)))
        self.assertTrue(v['order4_consistent'])
        self.assertGreater(np.linalg.norm(v['low']-exact),9)
    def test_reference_dependence(self):
        r=np.array([1.,2.]);a=np.array([1.,2.1]);b=np.array([1.1,2.1]);self.assertNotAlmostEqual(relative(r,a),relative(r,b))
    def test_component_order_may_differ(self):
        h=[.04,.02,.01]
        x=[np.array([1+q**4,2+q**2]) for q in h]
        v=triple_stats(*x)
        self.assertAlmostEqual(v['order'],2,places=2)
        cp=np.log2(abs((x[0][0]-x[1][0])/(x[1][0]-x[2][0])))
        self.assertAlmostEqual(cp,4.,places=7)
if __name__=='__main__':unittest.main()
