"""Tests on algebraic decomposition and frozen real Q550 parent data (not a server rerun)."""
import sys,unittest,math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from structural_jet import structural_jet,coupling_jet,chart_kernel
from run_audit import compute_row,cancellation_stats,case_meta,verify_parent_inputs

class Params:
    phi0=0.;e_reserve=.01;barrier_B=.3;mu_x=.05;gamma_x=.3;gamma_rho=.3;m_x=1.

class TransportTests(unittest.TestCase):
    def test_frozen_parent_provenance(self):
        r,d=verify_parent_inputs()
        self.assertEqual(r['original_snapshot_count'],36)
        self.assertEqual(len(d),4)
    def test_exact_algebraic_telescope_and_zero_z(self):
        z=np.load(ROOT/'inputs'/'structural_snapshot_states.npz')
        y=z['states'][0];scale=z['scale'];g=np.r_[z['g'][0],0.]
        kr=1/6;kg=35*math.pi*10/(12*550);s=float(z['slope'])
        yr=y;yg=y.copy();fy=structural_jet(y,kr,Params,s)[0]
        # Common state and flow isolate the direct kappa Hessian contribution.
        r,terms,krvec,kgvec,*_=compute_row(yr,yg,g,g,kr,kg,s,Params,scale,fy,fy)
        self.assertLess(r['telescope_budget_relative_error'],1e-10)
        self.assertLess(r['direct_analytic_identity_relative_error'],1e-8)
        self.assertLess(np.linalg.norm(terms[1]),1e-14)
        self.assertLess(np.linalg.norm(terms[2]),1e-14)
        self.assertLess(np.linalg.norm(terms[3]),1e-14)
        self.assertEqual(r['direct_z_kernel_column'],0.)
        coupling=coupling_jet(y,Params)[2]
        self.assertEqual(float(np.max(np.abs(coupling[:,8,:]))),0.)
    def test_cancellation_scaling_invariance(self):
        a=np.array([1.,2.,3.]);b=np.array([-1.,-1.9,-3.])
        a0=cancellation_stats(a,b)
        a1=cancellation_stats(7*a,7*b)
        self.assertAlmostEqual(a0['cancellation_ratio'],a1['cancellation_ratio'],places=12)
        self.assertTrue(a0['cancellation_ratio']>10)
        self.assertLess(a0['opposition_cosine'],-.99)
    def test_history_labels(self):
        z=np.load(ROOT/'inputs'/'structural_snapshot_states.npz')
        info=case_meta(str(z['labels'][0]))
        self.assertAlmostEqual(info[2],3.5)
        self.assertEqual(info[3],0)
        self.assertEqual(info[4],.125)
        self.assertEqual(len(z['labels']),36)

if __name__=='__main__':unittest.main()
