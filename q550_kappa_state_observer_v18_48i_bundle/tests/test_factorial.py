import unittest,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from factorial_math import shapley,second_interaction,diagnostics
import numpy as np
class FactorialTests(unittest.TestCase):
 def test_two_factors_additive(self):
  vals=[0.,3.,5.,8.];a,b=shapley(vals,2)
  self.assertAlmostEqual(float(a),3.);self.assertAlmostEqual(float(b),5.)
  self.assertAlmostEqual(float(second_interaction(*vals)),0.)
 def test_two_factors_interaction(self):
  vals=[0.,1.,1.,8.];a,b=shapley(vals,2)
  self.assertAlmostEqual(float(a+b),8.);self.assertAlmostEqual(float(second_interaction(*vals)),6.)
 def test_three_factors(self):
  vals=[i for i in range(8)];c=shapley(vals,3)
  self.assertAlmostEqual(float(sum(c)),7.)
 def test_vector_closure(self):
  v=[np.array([i,(-1)**i*i],float) for i in range(8)]
  c=shapley(v,3);np.testing.assert_allclose(sum(c),v[-1]-v[0],atol=1e-13)
 def test_antiparallel(self):
  d=diagnostics(np.array([1.,0.]),np.array([-.99,0.]))
  self.assertAlmostEqual(d['cancellation_ratio'],199.,places=8)
  self.assertLess(d['opposition_cosine'],-.999)
 def test_null_not_promoted(self):
  d=diagnostics(np.array([1.,0.]),np.array([0.,0.]))
  self.assertTrue(d['is_coupling_null']);self.assertEqual(d['cancellation_ratio'],1.)
if __name__=='__main__':unittest.main()
