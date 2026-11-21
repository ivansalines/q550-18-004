import unittest,math,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from run_audit import canonical,wrap,activity_gradient,matching_cost

class StructuralTests(unittest.TestCase):
 def setUp(self):
  self.scale=np.arange(1,11,dtype=float)*.1
  class P:phi0=.2
  self.p=P()
  self.y=np.array([.55,0.,4000.,.48,-2.1,0.,144.,.017,0.,0.,4361.])
 def test_gauge_relative_phase(self):
  y=canonical(self.y,0.)
  self.assertAlmostEqual(wrap(y[2]+y[6]-self.p.phi0-y[10]),wrap(self.y[2]+self.y[6]-self.p.phi0-self.y[10]),places=11)
 def test_gauge_clock(self):self.assertEqual(canonical(self.y,12.)[10],12.)
 def test_activity_observer_finite(self):
  for kind in ('sin','cos'):
   g=activity_gradient(self.y,self.p,self.scale,kind)
   self.assertEqual(g.shape,(11,));self.assertAlmostEqual(np.linalg.norm(g),1.,places=12)
 def test_modular_distance(self):
  y2=self.y.copy();y2[2]+=2*math.pi;y2[6]-=4*math.pi
  d,p=matching_cost(self.y,y2,self.scale,self.p)
  self.assertLess(d,1e-10);self.assertLess(p,1e-10)

if __name__=='__main__':unittest.main()
