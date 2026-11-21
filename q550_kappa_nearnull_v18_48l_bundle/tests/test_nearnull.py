import sys,unittest
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parents[1];sys.path[:0]=[str(P/'scripts'),str(P/'reference')]
from subspace_math import source_active_mode,subspace_angle,full_null
from high_precision import Jet
class GeometryTests(unittest.TestCase):
 def test_second_jet_square(self):
  a=Jet.var(2,0);z=a*a
  self.assertEqual(float(z.x),4.)
  self.assertEqual(float(z.g[0]),4.)
  self.assertEqual(float(z.H[0][0]),2.)
 def test_null_vs_source(self):
  b=np.eye(10);c=-np.eye(10);c[0,0]=0.
  n=full_null(b,c)
  self.assertLess(n['sigma_min'],1e-12)
  self.assertGreater(n['source_norm'],.1)
 def test_active_cutoff_rank(self):
  b=np.diag([1.,1e-3]+[1e-8]*8);c=-b;c[0,0]=0.
  a=source_active_mode(b,c,1e-6)
  self.assertEqual(a['rank'],2)
 def test_line_angle_sign_invariant(self):
  x=np.arange(1,11.);self.assertAlmostEqual(subspace_angle(x,-x),0.,places=5)
 def test_no_false_rank_certificate(self):
  b=np.diag([1,1,0,0,0,0,0,0,0,0]);c=np.zeros((10,10))
  a=source_active_mode(b,c,1e-6)
  self.assertEqual(a['rank'],2)
if __name__=='__main__':unittest.main()
