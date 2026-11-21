"""Frozen-source and geometry tests for v18.48g."""
import hashlib,sys,unittest,math,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import run_audit as a
import run_audit_v18_48f as frozen
from structural_jet import coupling_jet,structural_jet

class GeometryTests(unittest.TestCase):
 def test_exact_pinned_inputs(self):
  p,_=frozen.verify_parent_inputs()
  self.assertEqual(p['version'],'v18.48e')
  z=np.load(ROOT/'inputs'/'structural_snapshot_states.npz')
  self.assertEqual(z['states'].shape,(36,11))
 def test_gauge_preserves_relative_phase(self):
  u=np.array([.54,1e-6,4209.,.48,-2.13,1e-6,143.,.017,1e-5,1e-6,4361.])
  v=u.copy();v[2]+=5.6;v[6]-=7.1;v[10]+=.6
  vv,meta=a.gauge_align(u,v)
  self.assertAlmostEqual(vv[10],u[10],places=10)
  self.assertAlmostEqual(a.wrap(a.phase(vv)-a.phase(v)),0,places=10)
  self.assertTrue(abs(vv[2]-u[2])<=math.pi)
 def test_kappa_coupling_zero_z(self):
  z=np.load(ROOT/'inputs'/'structural_snapshot_states.npz')
  y=z['states'][0]
  class P:phi0=0.
  _,_,H=coupling_jet(y,P())
  self.assertEqual(np.max(np.abs(H[:,8,:])),0)
  self.assertEqual(np.max(np.abs(H[:,:,8])),0)
 def test_lift_endpoints_are_equivalent(self):
  z=np.load(ROOT/'inputs'/'structural_snapshot_states.npz')
  y0,y1=z['states'][0],z['states'][3]
  a1,_=a.gauge_align(y0,y1)
  self.assertTrue(np.allclose(a.interpolate_state(y0,a1,1),a1))
  self.assertAlmostEqual(a.wrap(a.phase(y1)-a.phase(a1)),0,places=10)
 def test_frozen_histogram_pins(self):
  pin=json.loads((ROOT/'inputs'/'V18_48F_SHA256.json').read_text())['files']
  for name,digest in pin.items():self.assertEqual(a.digest(ROOT/'inputs'/name),digest)

if __name__=='__main__':unittest.main()
