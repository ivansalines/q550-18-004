"""Synthetic 11D tests using independent directional finite differences; not a Q550 server run."""
import math,unittest,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
from structural_jet import coupling_jet,structural_jet,analytic_delta
class P:
    phi0=0.0;e_reserve=.01;barrier_B=.3;mu_x=.05;gamma_x=.3;gamma_rho=.3;m_x=1.0
class Test(unittest.TestCase):
    def setUp(self):
        self.rng=np.random.default_rng(1848);self.p=P()
    def test_coupling_hessian_analytic(self):
        for _ in range(12):
            y=self.rng.normal(size=11)*.15
            u=self.rng.normal(size=11);u/=np.linalg.norm(u)
            v=self.rng.normal(size=11);v/=np.linalg.norm(v)
            k,j,h=coupling_jet(y,self.p)
            hh=1e-4
            numerical=(coupling_jet(y+hh*u,self.p)[1]@v-coupling_jet(y-hh*u,self.p)[1]@v)/(2*hh)
            exact=np.einsum('iab,a,b->i',h,u,v)
            self.assertLess(np.linalg.norm(numerical-exact),2e-7)
            self.assertEqual(float(np.max(np.abs(h[:,8,:]))),0.)
            self.assertEqual(float(np.max(np.abs(h[:,:,8]))),0.)
            self.assertLess(np.max(np.abs(h-h.swapaxes(1,2))),1e-13)
            self.assertTrue(all(i in (1,3,5,7) for i in np.where(np.max(np.abs(h),axis=(1,2))>0)[0]))
    def test_full_model_symbolic_jet(self):
        for _ in range(4):
            y=self.rng.normal(size=11)*.1
            a=self.rng.normal(size=11);a/=np.linalg.norm(a)
            b=self.rng.normal(size=11);b/=np.linalg.norm(b)
            f,j,h=structural_jet(y,1/6,self.p)
            t=1e-5
            j_num=(structural_jet(y+t*a,1/6,self.p)[0]-structural_jet(y-t*a,1/6,self.p)[0])/(2*t)
            self.assertLess(np.linalg.norm(j_num-j@a)/max(np.linalg.norm(j@a),1),1e-8)
            h_num=(structural_jet(y+t*a,1/6,self.p)[1]@b-structural_jet(y-t*a,1/6,self.p)[1]@b)/(2*t)
            self.assertLess(np.linalg.norm(h_num-np.einsum('ijk,j,k->i',h,a,b))/max(np.linalg.norm(h_num),1),1e-7)
            self.assertEqual(float(np.max(np.abs(h[10]))),0.)
    def test_affine_in_kappa(self):
        y=self.rng.normal(size=11)*.1
        ka=1/6;kb=35*math.pi*10/(12*550)
        a=structural_jet(y,ka,self.p);b=structural_jet(y,kb,self.p)
        d=analytic_delta(y,ka,kb,self.p)
        for aa,bb,dd in zip(a,b,d):
            self.assertLess(np.linalg.norm((bb-aa)-dd)/max(np.linalg.norm(dd),1e-30),1e-9)
if __name__=='__main__':unittest.main(verbosity=2)
